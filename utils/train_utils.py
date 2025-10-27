"""Training utils for One-D-Piece and TiTok.

Original code Copyright (2024) Bytedance Ltd. and/or its affiliates
Modified code Copyright (2024) Turing Inc. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""
import json
import os
import time
from pathlib import Path
import pprint
import glob
from collections import defaultdict

from data import SimpleImageDataset, SimpleVideoDataset
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import AdamW
from utils.lr_schedulers import get_scheduler
from modeling.modules import EMAModel, ReconstructionLoss_Stage1, ReconstructionLoss_Stage2, MLMLoss, ReconstructionLoss_Single_Stage, ReconstructionLoss_Reward
from modeling.titok import TiTok, PretrainedTokenizer as TiTokPretrainedTokenizer
from modeling.one_d_piece import OneDPiece, PretrainedTokenizer as OneDPiecePretrainedTokenizer
from modeling.quadtok import QuadTok, PolicyQuadTok
from modeling.maskgit import ImageBert, UViTBert
from eval.utils.evaluator import VQGANEvaluator
from demo_util import sample_fn
import torchvision
from torch.nn.utils.rnn import pad_sequence

from utils.viz_utils import make_viz_from_samples, make_viz_from_samples_generation
from torchinfo import summary


def get_config():
    """Reads configs from a yaml file and terminal."""
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


class AverageMeter(object):
    """Computes and stores the average and current value.
    
    This class is borrowed from
    https://github.com/pytorch/examples/blob/main/imagenet/main.py#L423
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_pretrained_tokenizer(config, logger, accelerator):
    if config.model.type == "titok":
        if config.model.vq_model.finetune_decoder:
            # No need of pretrained tokenizer at stage2
            pretrianed_tokenizer = None
        else:
            pretrianed_tokenizer = TiTokPretrainedTokenizer(config.model.vq_model.pretrained_tokenizer_weight)
            pretrianed_tokenizer.to(accelerator.device)
        return pretrianed_tokenizer
    elif config.model.type == "one_d_piece":
        if config.model.vq_model.finetune_decoder:
            # No need of pretrained tokenizer at stage2
            pretrianed_tokenizer = None
        else:
            pretrianed_tokenizer = OneDPiecePretrainedTokenizer(config.model.vq_model.pretrained_tokenizer_weight)
            pretrianed_tokenizer.to(accelerator.device)
        return pretrianed_tokenizer
    else:
        raise ValueError(f"Unsupported model type {config.model.type}")


def create_model(config, logger, accelerator,
                 model_type="titok"):
    """Creates TiTok model."""
    logger.info("Creating model.")
    if model_type == "titok":
        model_cls = TiTok
    elif model_type == "one_d_piece":
        model_cls = OneDPiece
    elif model_type == "quadtok":
        model_cls = QuadTok
    elif model_type == "maskgit":
        if config.model.generator.model_type == "ViT":
            model_cls = ImageBert
        elif config.model.generator.model_type == "UViT":
            model_cls == UViTBert
        else:
            raise ValueError(f"Unsupported generator model_type {config.model.generator.model_type}")
    else:
        raise ValueError(f"Unsupported model_type {model_type}")
    model = model_cls(config)

    if config.experiment.get("init_weight", ""):
        # If loading a pretrained weight
        model_weight = torch.load(config.experiment.init_weight, map_location="cpu")

        need_pretrained_tokenizer_load = config.model.vq_model.get("pretrained_tokenizer_weight", None) is not None
        need_pretrained_tokenizer_load = need_pretrained_tokenizer_load and ((model_type in ["titok", "one_d_piece"] and config.model.vq_model.finetune_decoder))
        if need_pretrained_tokenizer_load:
            # Add the MaskGIT-VQGAN's quantizer/decoder weight as well
            pretrained_tokenizer_weight = torch.load(
                config.model.vq_model.pretrained_tokenizer_weight, map_location="cpu"
            )
            # Only keep the quantize and decoder part
            pretrained_tokenizer_weight = {"pixel_" + k:v for k,v in pretrained_tokenizer_weight.items() if not "encoder." in k}
            model_weight.update(pretrained_tokenizer_weight)
        
        init_from_vae = config.experiment.get("init_from_vae", False)
        if init_from_vae:
            model_weight.pop('selector.out_proj.weight')
            model_weight.pop('selector.out_proj.bias')

        msg = model.load_state_dict(model_weight, strict=False)
        logger.info(f"loading weight from {config.experiment.init_weight}, msg: {msg}")

    # Create the EMA model.
    ema_model = None
    if config.training.use_ema:
        ema_model = EMAModel(model.parameters(), decay=0.999,
                            model_cls=model_cls, config=config)
        # Create custom saving and loading hooks so that `accelerator.save_state(...)` serializes in a nice format.
        def load_model_hook(models, input_dir):
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "ema_model"),
                                                  model_cls=model_cls, config=config)
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                ema_model.save_pretrained(os.path.join(output_dir, "ema_model"))

        accelerator.register_load_state_pre_hook(load_model_hook)
        accelerator.register_save_state_pre_hook(save_model_hook)

    # Print Model for sanity check.
    if accelerator.is_main_process:
        if model_type in ["titok", "one_d_piece", "quadtok"]:
            if not model.train_policy:
                input_size = (1, 3, config.dataset.preprocessing.crop_size, config.dataset.preprocessing.crop_size)
                model_summary_str = summary(model, input_size=input_size, depth=5,
                col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
                logger.info(model_summary_str)
        elif model_type in ["maskgit"]:
            input_size = (1, config.model.vq_model.num_latent_tokens)
            input_data = [
                torch.randint(0, config.model.vq_model.codebook_size, input_size),
                torch.ones(1, dtype=int)
            ]
            model_summary_str = summary(
                model, input_data=input_data, depth=7,
                col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
            logger.info(model_summary_str)
        else:
            raise ValueError(f"Unsupported model type {model_type}")
        
    return model, ema_model

def create_policy_model(config, logger, accelerator,
                 model_type="titok"):
    """Creates TiTok model."""
    logger.info("Creating model.")
    model_cls = PolicyQuadTok
    model = model_cls(config)


    # Create the EMA model.
    ema_model = None
    if config.training.use_ema:
        ema_model = EMAModel(model.parameters(), decay=0.999,
                            model_cls=model_cls, config=config)
        # Create custom saving and loading hooks so that `accelerator.save_state(...)` serializes in a nice format.
        def load_model_hook(models, input_dir):
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "ema_model"),
                                                  model_cls=model_cls, config=config)
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                ema_model.save_pretrained(os.path.join(output_dir, "ema_model"))

        accelerator.register_load_state_pre_hook(load_model_hook)
        accelerator.register_save_state_pre_hook(save_model_hook)
        
    return model, ema_model

def create_model_and_loss_module(config, logger, accelerator,
                                 model_type="titok"):
    """Creates TiTok model and loss module."""
    logger.info("Creating model and loss module.")

    # Create model.
    model, ema_model = create_model(config, logger, accelerator, model_type=model_type)

    if model_type in ["titok", "one_d_piece", "quadtok"] and model.train_policy == False:
        loss_cls = ReconstructionLoss_Single_Stage
    elif model_type == "maskgit":
        loss_cls = MLMLoss
    elif model_type in ["titok", "one_d_piece", "quadtok"] and model.train_policy:
        loss_cls = ReconstructionLoss_Reward
    else:
        raise ValueError(f"Unsupported model_type {model_type}")

    # Create loss module along with discrminator.
    loss_module = loss_cls(config=config)

    return model, ema_model, loss_module


def create_optimizer(config, logger, model, loss_module,
                     need_discrminator=True):
    """Creates optimizer for TiTok and discrminator."""
    logger.info("Creating optimizers.")
    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer_cls = AdamW
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Exclude terms we may not want to apply weight decay.
    exclude = (lambda n, p: p.ndim < 2 or "ln" in n or "bias" in n or 'latent_tokens' in n 
               or 'mask_token' in n or 'embedding' in n or 'norm' in n or 'gamma' in n)
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    optimizer = optimizer_cls(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": optimizer_config.weight_decay},
        ],
        lr=learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2)
    )

    if (config.model.type in ["titok", "one_d_piece", "quadtok"]) and need_discrminator:
        discriminator_learning_rate = optimizer_config.discriminator_learning_rate
        discriminator_named_parameters = list(loss_module.named_parameters())
        discriminator_gain_or_bias_params = [p for n, p in discriminator_named_parameters if exclude(n, p) and p.requires_grad]
        discriminator_rest_params = [p for n, p in discriminator_named_parameters if include(n, p) and p.requires_grad]

        discriminator_optimizer = optimizer_cls(
            [
                {"params": discriminator_gain_or_bias_params, "weight_decay": 0.},
                {"params": discriminator_rest_params, "weight_decay": optimizer_config.weight_decay},
            ],
            lr=discriminator_learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2)
        )
    else:
        discriminator_optimizer = None

    return optimizer, discriminator_optimizer


def create_lr_scheduler(config, logger, accelerator, optimizer, discriminator_optimizer=None):
    """Creates learning rate scheduler for TiTok and discrminator."""
    logger.info("Creating lr_schedulers.")
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps * accelerator.num_processes,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
        base_lr=config.lr_scheduler.params.learning_rate,
        end_lr=config.lr_scheduler.params.end_lr,
    )
    if discriminator_optimizer is not None:
        discriminator_lr_scheduler = get_scheduler(
            config.lr_scheduler.scheduler,
            optimizer=discriminator_optimizer,
            num_training_steps=config.training.max_train_steps * accelerator.num_processes - config.losses.discriminator_start,
            num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
            base_lr=config.lr_scheduler.params.learning_rate,
            end_lr=config.lr_scheduler.params.end_lr,
        )
    else:
        discriminator_lr_scheduler = None
    return lr_scheduler, discriminator_lr_scheduler


def create_dataloader(config, logger, accelerator):
    """Creates data loader for training and testing."""
    logger.info("Creating dataloaders.")
    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    total_batch_size = (
        config.training.per_gpu_batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
    )
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    dataset_type = config.dataset.get("type", "simple_image")
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # TODO: add support on pre-tokenization dataset
    base_params = dict(
        num_train_examples=config.experiment.max_train_examples,
        per_gpu_batch_size=config.training.per_gpu_batch_size,
        global_batch_size=total_batch_size_without_accum,
        num_workers_per_gpu=dataset_config.num_workers_per_gpu,
        resize_shorter_edge=preproc_config.resize_shorter_edge,
        crop_size=preproc_config.crop_size,
        random_crop=preproc_config.random_crop,
        random_flip=preproc_config.random_flip,
    )
    if dataset_type == "simple_image":
        dataset = SimpleImageDataset(
            train_shards_path=dataset_config.train_shards_path_or_url,
            eval_shards_path=dataset_config.eval_shards_path_or_url,
            **base_params,
        )
    elif dataset_type == "video_image":
        dataset = SimpleVideoDataset(
            train_shards_path=dataset_config.train_shards_path_or_url,
            eval_shards_path=dataset_config.eval_shards_path_or_url,
            **base_params,
        )
    train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader
    
    return train_dataloader, eval_dataloader


def create_evaluator(config, logger, accelerator):
    """Creates evaluator."""
    logger.info("Creating evaluator.")
    if config.model.vq_model.get("quantize_mode", "vq") == "vq":
        evaluator = VQGANEvaluator(
            device=accelerator.device,
            enable_rfid=True,
            enable_inception_score=True,
            enable_codebook_usage_measure=True,
            enable_codebook_entropy_measure=True,
            num_codebook_entries=config.model.vq_model.codebook_size
        )
    elif config.model.vq_model.get("quantize_mode", "vq") == "vae":
        evaluator = VQGANEvaluator(
            device=accelerator.device,
            enable_rfid=True,
            enable_inception_score=True,
            enable_codebook_usage_measure=False,
            enable_codebook_entropy_measure=False,
        )
    else:
        raise NotImplementedError
    return evaluator


def auto_resume(config, logger, accelerator, ema_model,
                num_update_steps_per_epoch, strict=True):
    """Auto resuming the training."""
    global_step = 0
    first_epoch = 0
    # If resuming training.
    if config.experiment.resume:            
        accelerator.wait_for_everyone()
        local_ckpt_list = list(glob.glob(os.path.join(
            config.experiment.output_dir, "checkpoint*")))
        logger.info(f"All globbed checkpoints are: {local_ckpt_list}")
        if len(local_ckpt_list) >= 1:
            if len(local_ckpt_list) > 1:
                fn = lambda x: int(x.split('/')[-1].split('-')[-1])
                checkpoint_paths = sorted(local_ckpt_list, key=fn, reverse=True)
            else:
                checkpoint_paths = local_ckpt_list
            global_step = load_checkpoint(
                Path(checkpoint_paths[0]),
                accelerator,
                logger=logger,
                strict=strict
            )
            if config.training.use_ema:
                ema_model.set_step(global_step)
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            logger.info("Training from scratch.")
    return global_step, first_epoch


def train_one_epoch(config, logger, accelerator,
                    model, ema_model, loss_module,
                    optimizer, discriminator_optimizer,
                    lr_scheduler, discriminator_lr_scheduler,
                    train_dataloader, eval_dataloader,
                    evaluator,
                    global_step,
                    pretrained_tokenizer=None):
    """One epoch training."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()

    autoencoder_logs = defaultdict(float)
    discriminator_logs = defaultdict(float)
    for i, batch in enumerate(train_dataloader):
        model.train()
        additional_args = {}
        if config.model.type in ["titok", "one_d_piece", "quadtok"]:
            if "image" in batch:
                images = batch["image"].to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )
                # Reconstruction
                expected_output_images = images
            else:
                raise ValueError(f"Not found valid keys: {batch.keys()}")
        else:
            raise ValueError(f"Unsupported model type {config.model.type}")

        fnames = batch["__key__"]
        data_time_meter.update(time.time() - end)

        # Obtain proxy codes
        if pretrained_tokenizer is not None:
            pretrained_tokenizer.eval()
            if config.model.type in ["titok", "one_d_piece"]:
                proxy_codes = pretrained_tokenizer.encode(images)
            else:
                raise ValueError(f"Unsupported model type {config.model.type}")
        else:
            proxy_codes = None

        with accelerator.accumulate([model, loss_module]):
            reconstructed_images, extra_results_dict = model(images, **additional_args)
            if proxy_codes is None:
                autoencoder_loss, loss_dict = loss_module(
                    expected_output_images,
                    reconstructed_images,
                    extra_results_dict,
                    global_step,
                    mode="generator",
                )
            else:
                autoencoder_loss, loss_dict = loss_module(
                    proxy_codes,
                    reconstructed_images,
                    quantizer_loss=extra_results_dict,
                )      

            # Gather the losses across all processes for logging.
            autoencoder_logs = {}
            for k, v in loss_dict.items():
                if k in ["discriminator_factor", "d_weight", "num_tokens"]:
                    if type(v) == torch.Tensor:
                        autoencoder_logs["train/" + k] = v.cpu().item()
                    else:
                        autoencoder_logs["train/" + k] = v
                else:
                    autoencoder_logs["train/" + k] = accelerator.gather(v).mean().item()

            accelerator.backward(autoencoder_loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

            # Train discriminator.
            discriminator_logs = defaultdict(float)
            if (config.model.type in ["titok", "one_d_piece", "quadtok"]) and accelerator.unwrap_model(loss_module).should_discriminator_be_trained(global_step):
                discriminator_logs = defaultdict(float)
                discriminator_loss, loss_dict_discriminator = loss_module(
                    images,
                    reconstructed_images,
                    extra_results_dict,
                    global_step=global_step,
                    mode="discriminator",
                )

                # Gather the losses across all processes for logging.
                for k, v in loss_dict_discriminator.items():
                    if k in ["logits_real", "logits_fake"]:
                        if type(v) == torch.Tensor:
                            discriminator_logs["train/" + k] = v.cpu().item()
                        else:
                            discriminator_logs["train/" + k] = v
                    else:
                        discriminator_logs["train/" + k] = accelerator.gather(v).mean().item()

                accelerator.backward(discriminator_loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(loss_module.parameters(), config.training.max_grad_norm)

                discriminator_optimizer.step()
                discriminator_lr_scheduler.step()
        
                # Log gradient norm before zeroing it.
                if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                ):
                    log_grad_norm(loss_module, accelerator, global_step + 1)
                
                discriminator_optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )

                lr = lr_scheduler.get_last_lr()[0]
                if not config.model.train_policy:
                    if config.model.vq_model.quantize_mode == "vq":
                        logger.info(
                            f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                            f"Batch (t): {batch_time_meter.val:0.4f} "
                            f"LR: {lr:0.6f} "
                            f"Step: {global_step + 1} "
                            f"Total Loss: {autoencoder_logs['train/total_loss']:0.4f} "
                            f"Quantizer Loss: {autoencoder_logs['train/quantizer_loss']:0.4f} "
                            f"Recon Loss: {autoencoder_logs['train/reconstruction_loss']:0.4f} "
                            + (f"Discriminator Loss: {autoencoder_logs['train/weighted_gan_loss']:0.4f} " if "train/weighted_gan_loss" in autoencoder_logs else "")
                            + (f"Perceptual Loss: {autoencoder_logs['train/perceptual_loss']:0.4f} " if "train/perceptual_loss" in autoencoder_logs else "")
                        )
                    elif config.model.vq_model.quantize_mode == "vae":
                        logger.info(
                            f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                            f"Batch (t): {batch_time_meter.val:0.4f} "
                            f"LR: {lr:0.6f} "
                            f"Step: {global_step + 1} "
                            f"Total Loss: {autoencoder_logs['train/total_loss']:0.4f} "
                            f"KL Loss: {autoencoder_logs['train/kl_loss']:0.4f} "
                            f"Recon Loss: {autoencoder_logs['train/reconstruction_loss']:0.4f} "
                            + (f"Discriminator Loss: {autoencoder_logs['train/weighted_gan_loss']:0.4f} " if "train/weighted_gan_loss" in autoencoder_logs else "")
                            + (f"Perceptual Loss: {autoencoder_logs['train/perceptual_loss']:0.4f} " if "train/perceptual_loss" in autoencoder_logs else "")
                        )
                else:
                    if config.model.vq_model.quantize_mode == "vae":
                        logger.info(
                            f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                            f"Batch (t): {batch_time_meter.val:0.4f} "
                            f"LR: {lr:0.6f} "
                            f"Step: {global_step + 1} "
                            f"Total Loss: {autoencoder_logs['train/total_loss']:0.4f} "
                            f"Policy Loss: {autoencoder_logs['train/policy_loss']:0.4f} "
                            f"Prob Number: {autoencoder_logs['train/prob_mean']:0.4f} "
                            f"Reward: {autoencoder_logs['train/reward']:0.4f} "
                            f"KL Loss: {autoencoder_logs['train/kl_loss']:0.4f} "
                            f"Token Number: {autoencoder_logs['train/num_tokens']:0.2f} "
                            f"Recon Loss: {autoencoder_logs['train/reconstruction_loss']:0.4f} "
                            + (f"Discriminator Loss: {autoencoder_logs['train/weighted_gan_loss']:0.4f} " if "train/weighted_gan_loss" in autoencoder_logs else "")
                            + (f"Perceptual Loss: {autoencoder_logs['train/perceptual_loss']:0.4f} " if "train/perceptual_loss" in autoencoder_logs else "")
                        )
                    else:
                        NotImplementedError
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(autoencoder_logs)
                logs.update(discriminator_logs)
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                if config.model.type in ["titok", "one_d_piece", "quadtok"]:
                    reconstruct_method = reconstruct_images
                    additional_args = {}

                reconstruct_method(
                    model,
                    images[:config.training.num_generated_images],
                    fnames[:config.training.num_generated_images],
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    config=config,
                    pretrained_tokenizer=pretrained_tokenizer,
                    **additional_args
                )

                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())


            # Evaluate reconstruction.
            if config.model.type in ["titok", "one_d_piece", "quadtok"]:
                eval_metrics = eval_reconstruction
            else:
                raise ValueError(f"Unsupported model type {config.model.type}")
            if eval_dataloader is not None and (global_step + 1) % config.experiment.eval_every == 0:
                logger.info(f"Computing metrics on the validation set.")
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                    # Eval for EMA.
                    eval_scores = eval_metrics(
                        model,
                        eval_dataloader,
                        accelerator,
                        evaluator,
                        pretrained_tokenizer=pretrained_tokenizer
                    )
                    logger.info(
                        f"EMA EVALUATION "
                        f"Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {f'ema_eval/'+k: v for k, v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)
                    if config.training.get("use_ema", False):
                        # Switch back to the original model parameters for training.
                        ema_model.restore(model.parameters())
                else:
                    # Eval for non-EMA.
                    eval_scores = eval_metrics(
                        model,
                        eval_dataloader,
                        accelerator,
                        evaluator,
                        pretrained_tokenizer=pretrained_tokenizer
                    )

                    logger.info(
                        f"Non-EMA EVALUATION "
                        f"Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {f'eval/'+k: v for k, v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)

                accelerator.wait_for_everyone()

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break


    return global_step

def train_one_epoch_stage2(config, logger, accelerator,
                    model, policy_model, ema_model, loss_module,
                    optimizer, discriminator_optimizer,
                    lr_scheduler, discriminator_lr_scheduler,
                    train_dataloader, eval_dataloader,
                    evaluator,
                    global_step,
                    pretrained_tokenizer=None):
    """One epoch training."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()

    autoencoder_logs = defaultdict(float)
    discriminator_logs = defaultdict(float)
    for i, batch in enumerate(train_dataloader):
        model.train()
        additional_args = {}
        if config.model.type in ["titok", "one_d_piece", "quadtok"]:
            if "image" in batch:
                images = batch["image"].to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )
                # Reconstruction
                expected_output_images = images
            else:
                raise ValueError(f"Not found valid keys: {batch.keys()}")
        else:
            raise ValueError(f"Unsupported model type {config.model.type}")

        fnames = batch["__key__"]
        data_time_meter.update(time.time() - end)

        with accelerator.accumulate([policy_model, loss_module]):

            with torch.no_grad():
                image_latent = model.encode(images)
            predicted_actions, predicted_probs = policy_model(image_latent)
            breakpoint()
            z = model.selector._forward_policy(image_latent, predicted_actions)
            z_quantized = model.quantize(z).sample()
            reconstruction = model.decoder._forward_policy(z_quantized, predicted_actions, predicted_probs)

            breakpoint()
            reconstructed_images, extra_results_dict = model(images, **additional_args)
            autoencoder_loss, loss_dict = loss_module(
                expected_output_images,
                reconstructed_images,
                extra_results_dict,
                global_step,
                mode="generator",
            )


            # Gather the losses across all processes for logging.
            autoencoder_logs = {}
            for k, v in loss_dict.items():
                if k in ["discriminator_factor", "d_weight", "num_tokens"]:
                    if type(v) == torch.Tensor:
                        autoencoder_logs["train/" + k] = v.cpu().item()
                    else:
                        autoencoder_logs["train/" + k] = v
                else:
                    autoencoder_logs["train/" + k] = accelerator.gather(v).mean().item()

            accelerator.backward(autoencoder_loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

            # Train discriminator.
            discriminator_logs = defaultdict(float)
            if (config.model.type in ["titok", "one_d_piece", "quadtok"]) and accelerator.unwrap_model(loss_module).should_discriminator_be_trained(global_step):
                discriminator_logs = defaultdict(float)
                discriminator_loss, loss_dict_discriminator = loss_module(
                    images,
                    reconstructed_images,
                    extra_results_dict,
                    global_step=global_step,
                    mode="discriminator",
                )

                # Gather the losses across all processes for logging.
                for k, v in loss_dict_discriminator.items():
                    if k in ["logits_real", "logits_fake"]:
                        if type(v) == torch.Tensor:
                            discriminator_logs["train/" + k] = v.cpu().item()
                        else:
                            discriminator_logs["train/" + k] = v
                    else:
                        discriminator_logs["train/" + k] = accelerator.gather(v).mean().item()

                accelerator.backward(discriminator_loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(loss_module.parameters(), config.training.max_grad_norm)

                discriminator_optimizer.step()
                discriminator_lr_scheduler.step()
        
                # Log gradient norm before zeroing it.
                if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                ):
                    log_grad_norm(loss_module, accelerator, global_step + 1)
                
                discriminator_optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(policy_model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )

                lr = lr_scheduler.get_last_lr()[0]
                if not config.model.train_policy:
                    if config.model.vq_model.quantize_mode == "vq":
                        logger.info(
                            f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                            f"Batch (t): {batch_time_meter.val:0.4f} "
                            f"LR: {lr:0.6f} "
                            f"Step: {global_step + 1} "
                            f"Total Loss: {autoencoder_logs['train/total_loss']:0.4f} "
                            f"Quantizer Loss: {autoencoder_logs['train/quantizer_loss']:0.4f} "
                            f"Recon Loss: {autoencoder_logs['train/reconstruction_loss']:0.4f} "
                            + (f"Discriminator Loss: {autoencoder_logs['train/weighted_gan_loss']:0.4f} " if "train/weighted_gan_loss" in autoencoder_logs else "")
                            + (f"Perceptual Loss: {autoencoder_logs['train/perceptual_loss']:0.4f} " if "train/perceptual_loss" in autoencoder_logs else "")
                        )
                    elif config.model.vq_model.quantize_mode == "vae":
                        logger.info(
                            f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                            f"Batch (t): {batch_time_meter.val:0.4f} "
                            f"LR: {lr:0.6f} "
                            f"Step: {global_step + 1} "
                            f"Total Loss: {autoencoder_logs['train/total_loss']:0.4f} "
                            f"KL Loss: {autoencoder_logs['train/kl_loss']:0.4f} "
                            f"Recon Loss: {autoencoder_logs['train/reconstruction_loss']:0.4f} "
                            + (f"Discriminator Loss: {autoencoder_logs['train/weighted_gan_loss']:0.4f} " if "train/weighted_gan_loss" in autoencoder_logs else "")
                            + (f"Perceptual Loss: {autoencoder_logs['train/perceptual_loss']:0.4f} " if "train/perceptual_loss" in autoencoder_logs else "")
                        )
                else:
                    if config.model.vq_model.quantize_mode == "vae":
                        logger.info(
                            f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                            f"Batch (t): {batch_time_meter.val:0.4f} "
                            f"LR: {lr:0.6f} "
                            f"Step: {global_step + 1} "
                            f"Total Loss: {autoencoder_logs['train/total_loss']:0.4f} "
                            f"Policy Loss: {autoencoder_logs['train/policy_loss']:0.4f} "
                            f"Prob Number: {autoencoder_logs['train/prob_mean']:0.4f} "
                            f"Reward: {autoencoder_logs['train/reward']:0.4f} "
                            f"KL Loss: {autoencoder_logs['train/kl_loss']:0.4f} "
                            f"Token Number: {autoencoder_logs['train/num_tokens']:0.2f} "
                            f"Recon Loss: {autoencoder_logs['train/reconstruction_loss']:0.4f} "
                            + (f"Discriminator Loss: {autoencoder_logs['train/weighted_gan_loss']:0.4f} " if "train/weighted_gan_loss" in autoencoder_logs else "")
                            + (f"Perceptual Loss: {autoencoder_logs['train/perceptual_loss']:0.4f} " if "train/perceptual_loss" in autoencoder_logs else "")
                        )
                    else:
                        NotImplementedError
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(autoencoder_logs)
                logs.update(discriminator_logs)
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
                if config.training.get("use_ema", False):
                    ema_model.store(policy_model.parameters())
                    ema_model.copy_to(policy_model.parameters())
                if config.model.type in ["titok", "one_d_piece", "quadtok"]:
                    reconstruct_method = reconstruct_images
                    additional_args = {}

                reconstruct_method(
                    model,
                    images[:config.training.num_generated_images],
                    fnames[:config.training.num_generated_images],
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    config=config,
                    pretrained_tokenizer=pretrained_tokenizer,
                    **additional_args
                )

                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(policy_model.parameters())


            # Evaluate reconstruction.
            if config.model.type in ["titok", "one_d_piece", "quadtok"]:
                eval_metrics = eval_reconstruction
            else:
                raise ValueError(f"Unsupported model type {config.model.type}")
            if eval_dataloader is not None and (global_step + 1) % config.experiment.eval_every == 0:
                logger.info(f"Computing metrics on the validation set.")
                if config.training.get("use_ema", False):
                    ema_model.store(policy_model.parameters())
                    ema_model.copy_to(policy_model.parameters())
                    # Eval for EMA.
                    eval_scores = eval_metrics(
                        model,
                        eval_dataloader,
                        accelerator,
                        evaluator,
                        pretrained_tokenizer=pretrained_tokenizer
                    )
                    logger.info(
                        f"EMA EVALUATION "
                        f"Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {f'ema_eval/'+k: v for k, v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)
                    if config.training.get("use_ema", False):
                        # Switch back to the original model parameters for training.
                        ema_model.restore(policy_model.parameters())
                else:
                    # Eval for non-EMA.
                    eval_scores = eval_metrics(
                        model,
                        eval_dataloader,
                        accelerator,
                        evaluator,
                        pretrained_tokenizer=pretrained_tokenizer
                    )

                    logger.info(
                        f"Non-EMA EVALUATION "
                        f"Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {f'eval/'+k: v for k, v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)

                accelerator.wait_for_everyone()

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break


    return global_step

def train_one_epoch_policy(config, logger, accelerator,
                    model, policy_model, ema_model, loss_module,
                    optimizer, discriminator_optimizer,
                    lr_scheduler, discriminator_lr_scheduler,
                    train_dataloader, eval_dataloader,
                    evaluator,
                    global_step,
                    pretrained_tokenizer=None):
    """One epoch training."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    policy_model.train()
    model.eval()

    autoencoder_logs = defaultdict(float)
    discriminator_logs = defaultdict(float)
    for i, batch in enumerate(train_dataloader):
        policy_model.train()
        model.eval()
        additional_args = {}
        if config.model.type in ["titok", "one_d_piece", "quadtok"]:
            if "image" in batch:
                images = batch["image"].to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )
                # Reconstruction
                bs = images.shape[0]
                expected_output_images = images
            else:
                raise ValueError(f"Not found valid keys: {batch.keys()}")
        else:
            raise ValueError(f"Unsupported model type {config.model.type}")

        fnames = batch["__key__"]
        data_time_meter.update(time.time() - end)

        policy_params = config.policy
        REWARD_SCALING = 100000.0

        with accelerator.accumulate([policy_model]):
            with torch.no_grad():
                image_latents = model.encode(images)
            image_latents_grouped = image_latents.repeat(
                policy_params.group_size, *([1] * (image_latents.dim() - 1))
            )
            images_grouped = images.repeat(policy_params.group_size, 1, 1, 1)
            b_g = images_grouped.shape[0]

            total_returns = torch.zeros(b_g, device=accelerator.device)
            total_quality_improvement = torch.zeros(b_g, device=accelerator.device)

            with torch.no_grad():
                trajectories = accelerator.unwrap_model(policy_model).sample_trajectories(
                    image_latents_grouped
                )
            all_lods_list = []
            all_indices_list = []
            all_num_tokens_list = []
            all_image_latents_list = []
            all_gt_images_list = []
            
            trajectory_map = [] 
            total_steps = 0

            with torch.no_grad():
                for i_traj in range(b_g):
                    traj = trajectories[i_traj]
                    image_latent_i = traj["image_latent"].unsqueeze(0)
                    gt_image_i = images_grouped[i_traj].unsqueeze(0)

                    if not traj["partial_trees_list"] or len(traj["partial_trees_list"]) <= config.policy_model.guaranteed_depth:
                        continue

                    for t in range(config.policy_model.guaranteed_depth, model.num_lod):
                        if t >= len(traj["partial_trees_list"]):
                            break 
                            
                        partial_tree_t = traj["partial_trees_list"][t]
                        
                        all_lods_list.append(partial_tree_t["pruned_lods_padded"])
                        all_indices_list.append(partial_tree_t["pruned_indices_padded"])
                        all_num_tokens_list.append(partial_tree_t["num_tokens"])
                        all_image_latents_list.append(image_latent_i)
                        all_gt_images_list.append(gt_image_i)
                        
                        trajectory_map.append((i_traj, t))
                        total_steps += 1

            max_len_batch = max(t.shape[1] for t in all_lods_list)

            batched_lods = pad_sequence(
                [t.squeeze(0) for t in all_lods_list], batch_first=True, padding_value=-1
            )
            batched_indices = pad_sequence(
                [t.squeeze(0) for t in all_indices_list], batch_first=True, padding_value=-1
            )
            
            if batched_lods.shape[1] < max_len_batch:
                batched_lods = F.pad(batched_lods, (0, max_len_batch - batched_lods.shape[1]), 'constant', -1)
            if batched_indices.shape[1] < max_len_batch:
                batched_indices = F.pad(batched_indices, (0, max_len_batch - batched_indices.shape[1]), 'constant', -1)

            batched_num_tokens = torch.cat(all_num_tokens_list, dim=0)
            
            batched_policy_result = {
                "pruned_lods_padded": batched_lods,
                "pruned_indices_padded": batched_indices,
                "num_tokens": batched_num_tokens
            }
            
            batched_image_latents = torch.cat(all_image_latents_list, dim=0)
            batched_gt_images = torch.cat(all_gt_images_list, dim=0)

            with torch.no_grad():
                model.train_policy = True
                reconstructions, _ = model.decoding_pre_selector(
                    batched_image_latents, 
                    policy_output=batched_policy_result
                )
                
                total_losses_per_item, loss_dict = loss_module(
                    batched_gt_images,
                    reconstructions,
                )

            total_returns = torch.zeros(b_g, device=accelerator.device)
            total_quality_improvement = torch.zeros(b_g, device=accelerator.device)
            
            traj_losses = defaultdict(dict)

            for i in range(total_steps):
                i_traj, t = trajectory_map[i]
                traj_losses[i_traj][t] = total_losses_per_item[i]
                
            for i_traj in range(b_g):
                if not traj_losses[i_traj]: continue
                
                loss_G = traj_losses[i_traj][config.policy_model.guaranteed_depth]

                final_t_for_traj = max(traj_losses[i_traj].keys())
                loss_T = traj_losses[i_traj][final_t_for_traj]
                total_improvement = (loss_G - loss_T)
                total_quality_improvement[i_traj] = total_improvement 

                final_global_reward = -loss_T
                total_returns[i_traj] = (total_improvement * policy_params.improvement_scaling) + \
                                        (final_global_reward * policy_params.final_loss_scaling)

            returns_grouped = total_returns.view(bs, policy_params.group_size)
            group_mean = returns_grouped.mean(dim=1, keepdim=True)
            group_std = returns_grouped.std(dim=1, keepdim=True)
            advantages_normalized = (returns_grouped - group_mean) / (group_std + 1e-8)
            advantages_flat = advantages_normalized.flatten().detach()

            mean_policy_loss = torch.tensor(0.0, device=accelerator.device)

            valid_trajectories = []
            valid_advantages = []
            valid_indices = []
            
            for i_traj in range(b_g):
                if trajectories[i_traj]["actions_list"]: 
                    valid_trajectories.append(trajectories[i_traj])
                    valid_advantages.append(advantages_flat[i_traj])
                    valid_indices.append(i_traj)
            
            if not valid_trajectories:
                if accelerator.is_main_process:
                     logger.warning(f"Step {global_step}: No valid trajectories for PPO update.")
                continue #
            
            num_valid_trajectories = len(valid_trajectories)
            
 
            batched_image_latents = torch.cat(
                [traj["image_latent"].unsqueeze(0) for traj in valid_trajectories], 
                dim=0
            ) # (B_valid, L, D)
            
            batched_advantages = advantages_flat[valid_indices] # (B_valid,)
            
            max_T = max(len(traj["actions_list"]) for traj in valid_trajectories)
            
            batched_old_logprobs = torch.zeros(num_valid_trajectories, max_T, device=accelerator.device)
            batched_actions_full_tensor = torch.zeros(num_valid_trajectories, accelerator.unwrap_model(policy_model).num_total_nodes, 1, device=accelerator.device)
            timesteps_mask = torch.zeros(num_valid_trajectories, max_T, device=accelerator.device)

            for i, traj in enumerate(valid_trajectories):
                T_i = len(traj["actions_list"])
                timesteps_mask[i, :T_i] = 1.0
                
                logprobs_i = torch.stack(traj["action_logprobs_list"]) # (T_i,)
                batched_old_logprobs[i, :T_i] = logprobs_i
                
                lod_start_idx = 0
                for t in range(T_i):
                    actions_t = traj["actions_list"][t] # (num_nodes_t, 1)
                    num_nodes_at_lod = actions_t.shape[0]
                    if num_nodes_at_lod == 0: continue
                    
                    batched_actions_full_tensor[i, lod_start_idx : lod_start_idx + num_nodes_at_lod, :] = actions_t
                    lod_start_idx += num_nodes_at_lod
            
            for ppo_epoch in range(policy_params.ppo_epochs):
 
                current_logprobs_per_lod_batch = accelerator.unwrap_model(policy_model).reevaluate_logprobs(  #accelerator.unwrap_model
                    batched_image_latents,
                    batched_actions_full_tensor
                )
                
                advantages_expanded = batched_advantages.unsqueeze(-1) # (B_valid, 1)
                
                # (B_valid, T)
                ratios = torch.exp(current_logprobs_per_lod_batch - batched_old_logprobs)
                
                surr1 = ratios * advantages_expanded
                surr2 = torch.clamp(
                    ratios, 
                    1.0 - policy_params.clip_epsilon, 
                    1.0 + policy_params.clip_epsilon
                ) * advantages_expanded

                policy_loss = -torch.min(surr1, surr2) # (B_valid, T)
                
                policy_loss = policy_loss * timesteps_mask
                
                mean_policy_loss = policy_loss.sum() / timesteps_mask.sum()

                optimizer.zero_grad()
                accelerator.backward(mean_policy_loss)
                
                if accelerator.sync_gradients and config.get("clip_grad_norm", None):
                   accelerator.clip_grad_norm_(policy_model.parameters(), config.clip_grad_norm)

                optimizer.step()

            lr_scheduler.step()
            policy_logs = {}
            # if accelerator.is_main_process:
            policy_logs["train/lr"] = lr_scheduler.get_last_lr()[0]
            policy_logs["train/total_return"] = accelerator.gather(total_returns).mean().item()
            policy_logs["train/ppo_loss"] = accelerator.gather(mean_policy_loss).mean().item()
            policy_logs["train/adv_mean"] = accelerator.gather(advantages_flat).mean().item()
            policy_logs["train/adv_std"] = accelerator.gather(advantages_flat).std().item()

            for k, v in loss_dict.items():
                policy_logs["train/" + k] = accelerator.gather(v).mean().item()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(policy_model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)


        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(policy_model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )

                lr = lr_scheduler.get_last_lr()[0]

                logger.info(
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"PPO Loss: {policy_logs['train/ppo_loss']:.4f}, "
                    f"Return (Quality Impr.): {policy_logs['train/total_return']:.4f}, "
                    f"Adv Mean: {policy_logs['train/adv_mean']:.4f}, "
                    f"Adv Std: {policy_logs['train/adv_std']:.4f}, "
                    f"Recon Loss: {policy_logs['train/reconstruction_loss']:.4f}, "
                    f"Perceptual Loss: {policy_logs['train/perceptual_loss']:.4f}, "
                    f"LR: {lr:.2e}"
                )

                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(policy_logs)

                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    policy_model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images.

            ####### NOT DONE YET!!
            # if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
            #     # Store the model parameters temporarily and load the EMA parameters to perform inference.
            #     if config.training.get("use_ema", False):
            #         ema_model.store(model.parameters())
            #         ema_model.copy_to(model.parameters())
            #     if config.model.type in ["titok", "one_d_piece", "quadtok"]:
            #         reconstruct_method = reconstruct_images
            #         additional_args = {}

            #     reconstruct_method(
            #         model,
            #         images[:config.training.num_generated_images],
            #         fnames[:config.training.num_generated_images],
            #         accelerator,
            #         global_step + 1,
            #         config.experiment.output_dir,
            #         logger=logger,
            #         config=config,
            #         pretrained_tokenizer=pretrained_tokenizer,
            #         **additional_args
            #     )

            #     if config.training.get("use_ema", False):
            #         # Switch back to the original model parameters for training.
            #         ema_model.restore(model.parameters())


            # # Evaluate reconstruction.
            # if config.model.type in ["titok", "one_d_piece", "quadtok"]:
            #     eval_metrics = eval_reconstruction
            # else:
            #     raise ValueError(f"Unsupported model type {config.model.type}")
            # if eval_dataloader is not None and (global_step + 1) % config.experiment.eval_every == 0:
            #     logger.info(f"Computing metrics on the validation set.")
            #     if config.training.get("use_ema", False):
            #         ema_model.store(model.parameters())
            #         ema_model.copy_to(model.parameters())
            #         # Eval for EMA.
            #         eval_scores = eval_metrics(
            #             model,
            #             eval_dataloader,
            #             accelerator,
            #             evaluator,
            #             pretrained_tokenizer=pretrained_tokenizer
            #         )
            #         logger.info(
            #             f"EMA EVALUATION "
            #             f"Step: {global_step + 1} "
            #         )
            #         logger.info(pprint.pformat(eval_scores))
            #         if accelerator.is_main_process:
            #             eval_log = {f'ema_eval/'+k: v for k, v in eval_scores.items()}
            #             accelerator.log(eval_log, step=global_step + 1)
            #         if config.training.get("use_ema", False):
            #             # Switch back to the original model parameters for training.
            #             ema_model.restore(model.parameters())
            #     else:
            #         # Eval for non-EMA.
            #         eval_scores = eval_metrics(
            #             model,
            #             eval_dataloader,
            #             accelerator,
            #             evaluator,
            #             pretrained_tokenizer=pretrained_tokenizer
            #         )

            #         logger.info(
            #             f"Non-EMA EVALUATION "
            #             f"Step: {global_step + 1} "
            #         )
            #         logger.info(pprint.pformat(eval_scores))
            #         if accelerator.is_main_process:
            #             eval_log = {f'eval/'+k: v for k, v in eval_scores.items()}
            #             accelerator.log(eval_log, step=global_step + 1)

                accelerator.wait_for_everyone()

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break


    return global_step


def train_one_epoch_generator(
                    config, logger, accelerator,
                    model, ema_model, loss_module,
                    optimizer,
                    lr_scheduler,
                    train_dataloader,
                    tokenizer,
                    global_step,):
    """One epoch training."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()

    for i, batch in enumerate(train_dataloader):
        model.train()
        if "image" in batch:
            images = batch["image"].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
            conditions = batch["class_id"].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )

            # Encode images on the flight.
            with torch.no_grad():
                tokenizer.eval()
                if config.model.generator_type in ["maskgit"]:
                    length = config.model.generator.image_seq_len
                input_tokens = tokenizer.encode(images)[1]["min_encoding_indices"]
                input_tokens = input_tokens[:,:,:length]
                assert input_tokens.shape[2] == length, f"Expected input tokens shape {length}, got {input_tokens.shape[2]}"
                input_tokens = input_tokens.reshape(images.shape[0], -1)
        else:
            raise ValueError(f"Not found valid keys: {batch.keys()}")

        fnames = batch["__key__"]
        data_time_meter.update(time.time() - end)

        unwrap_model = accelerator.unwrap_model(model)

        # Randomly masking out input tokens.
        if config.model.generator_type in ["maskgit"]:
            masked_tokens, masks = unwrap_model.masking_input_tokens(
                input_tokens)
            
        with accelerator.accumulate([model]):
            if config.model.generator_type in ["maskgit"]:
                logits = model(masked_tokens, conditions,
                            cond_drop_prob=config.model.generator.class_label_dropout)
            loss, loss_dict= loss_module(logits, input_tokens, weights=masks)
            # Gather the losses across all processes for logging.
            mlm_logs = {}
            for k, v in loss_dict.items():
                mlm_logs["train/" + k] = accelerator.gather(v).mean().item()
            accelerator.backward(loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )

                lr = lr_scheduler.get_last_lr()[0]
                logger.info(
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Loss: {mlm_logs['train/loss']:0.4f} "
                    f"Accuracy: {mlm_logs['train/correct_tokens']:0.4f} "
                )
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(mlm_logs)
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                generate_images(
                    model,
                    tokenizer,
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    config=config
                )

                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break


    return global_step


@torch.no_grad()
def eval_reconstruction(
    model,
    eval_loader,
    accelerator,
    evaluator,
    pretrained_tokenizer=None,
    length=None,
):
    model.eval()
    evaluator.reset_metrics()
    local_model = accelerator.unwrap_model(model)

    for batch in eval_loader:
        images = batch["image"].to(
            accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
        )
        original_images = torch.clone(images)
        additional_args = {}
        if length is not None:
            additional_args["length"] = length
        reconstructed_images, model_dict = local_model(images, **additional_args)
        if pretrained_tokenizer is not None:
            reconstructed_images = pretrained_tokenizer.decode(reconstructed_images.argmax(1))
        reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
        # Quantize to uint8
        reconstructed_images = torch.round(reconstructed_images * 255.0) / 255.0
        original_images = torch.clamp(original_images, 0.0, 1.0)
        # # For VQ model.
        # token_indices = model_dict["min_encoding_indices"]
        # if length is not None:
        #     token_indices = token_indices[:, :, :length]
        #     assert token_indices.shape[2] == length, f"Expected token indices shape {length}, got {token_indices.shape[2]}, where the original shape is {model_dict['min_encoding_indices'].shape}"
        if local_model.quantize_mode == 'vq': 
            # For VQ model.
            evaluator.update(original_images, reconstructed_images.squeeze(2), model_dict["min_encoding_indices"])
        else:
            # For VAE model.
            evaluator.update(original_images, reconstructed_images.squeeze(2), None)
    model.train()
    return evaluator.result()


@torch.no_grad()
def reconstruct_images(model, original_images, fnames, accelerator, 
                    global_step, output_dir, logger, config=None,
                    pretrained_tokenizer=None):
    logger.info("Reconstructing images...")
    original_images = torch.clone(original_images)
    model.eval()
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
        reconstructed_images, encoder_dict = accelerator.unwrap_model(model)(original_images)


    images_for_saving, images_for_logging = make_viz_from_samples(
        original_images,
        reconstructed_images
    )
    # Log images.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {f"Train Reconstruction": images_for_saving},
            step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Reconstruction": images_for_logging}, step=global_step
        )
    # Log locally.
    root = Path(output_dir) / "train_images"
    os.makedirs(root, exist_ok=True)
    for i,img in enumerate(images_for_saving):
        filename = f"{global_step:08}_s-{i:03}-{fnames[i]}.png"
        path = os.path.join(root, filename)
        img.save(path)

    model.train()


@torch.no_grad()
def generate_images(model, tokenizer, accelerator, 
                    global_step, output_dir, logger, config=None):
    model.eval()
    tokenizer.eval()
    logger.info("Generating images...")
    generated_image = sample_fn(
        accelerator.unwrap_model(model),
        tokenizer,
        guidance_scale=config.model.generator.get("guidance_scale", 3.0),
        guidance_decay=config.model.generator.get("guidance_decay", "constant"),
        guidance_scale_pow=config.model.generator.get("guidance_scale_pow", 3.0),
        randomize_temperature=config.model.generator.get("randomize_temperature", 2.0),
        softmax_temperature_annealing=config.model.generator.get("softmax_temperature_annealing", False),
        num_sample_steps=config.model.generator.get("num_steps", 8),
        device=accelerator.device,
        return_tensor=True
    )
    images_for_saving, images_for_logging = make_viz_from_samples_generation(
        generated_image)

    # Log images.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {"Train Generated": [images_for_saving]}, step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Generated": images_for_logging}, step=global_step
        )
    # Log locally.
    root = Path(output_dir) / "train_generated_images"
    os.makedirs(root, exist_ok=True)
    filename = f"{global_step:08}_s-generated.png"
    path = os.path.join(root, filename)
    images_for_saving.save(path)

    model.train()
    return


def save_checkpoint(model, output_dir, accelerator, global_step, logger) -> Path:
    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained_weight(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")

    accelerator.save_state(save_path)
    return save_path


def load_checkpoint(checkpoint_path: Path, accelerator, logger, strict=True):
    logger.info(f"Load checkpoint from {checkpoint_path}")

    accelerator.load_state(checkpoint_path, strict=strict)
    
    with open(checkpoint_path / "metadata.json", "r") as f:
        global_step = int(json.load(f)["global_step"])

    logger.info(f"Resuming at global_step {global_step}")
    return global_step


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)

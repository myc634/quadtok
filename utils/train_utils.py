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
import shutil
import time
from pathlib import Path
import pprint
import glob
from collections import defaultdict
import pickle
from data import SimpleImageDataset, PretokenizedDataset
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import AdamW
from utils.lr_schedulers import get_scheduler
from modeling.modules import EMAModel, ReconstructionLoss_Stage1, ReconstructionLoss_Single_Stage_Repa, MLMLoss, ReconstructionLoss_Single_Stage, ReconstructionLoss_Stage_Multi_Scale, DiffLoss
from modeling.titok import TiTok, PretrainedTokenizer as TiTokPretrainedTokenizer
from modeling.one_d_piece import OneDPiece, PretrainedTokenizer as OneDPiecePretrainedTokenizer
from modeling.quadtok import QuadTok, PolicyQuadTok
from modeling.maskgit import ImageBert, UViTBert
from modeling.mar import MAR, CausalMAR, QuadtreeMAR, QuadtreeGPT
from modeling.dit import DiT
from eval.utils.evaluator import VQGANEvaluator
from demo_util import sample_fn
import torchvision
from torch.nn.utils.rnn import pad_sequence

from utils.viz_utils import make_viz_from_samples, make_viz_from_samples_generation
from torchinfo import summary
import copy
from modeling.utils import build_quadtree, get_ordered_nodes, build_tree_from_decision_nodes, build_probabilistic_quadtree, QuadTreeNode
from concurrent.futures import ThreadPoolExecutor, as_completed

LOD_PROB_MAPPING = {2: 0.7, 3: 0.6, 4: 0.5}

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

def create_generater_tokenizer(config, logger, accelerator):
    if config.tokenizer.type == "mar-vae":
        from external_models.vae import AutoencoderKL
        tokenizer = AutoencoderKL(embed_dim=config.tokenizer.vae_embed_dim, ch_mult=config.tokenizer.ch_mult, ckpt_path=config.tokenizer.ckpt_dir)
        tokenizer.eval()
        tokenizer.requires_grad_(False)
        tokenizer.to(accelerator.device)
        return tokenizer
    elif config.tokenizer.type == "quadtok":
        tokenizer = QuadTok(config)
        tokenizer.eval()
        tokenizer.requires_grad_(False)
        tokenizer.to(accelerator.device)
        sd = torch.load(config.tokenizer.tokenizer_ckpt_dir, map_location="cpu")
        msg = tokenizer.load_state_dict(sd, strict=False)
        return tokenizer
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
            model_cls = UViTBert
        else:
            raise ValueError(f"Unsupported generator model_type {config.model.generator.model_type}")
    elif model_type == "mar":
        model_cls = MAR
    elif model_type == "mar-causal":
        model_cls = CausalMAR
    elif model_type == "mar-quadtree":
        model_cls = QuadtreeMAR #QuadtreeMARBox
    elif model_type == "gpt-quadtree":
        model_cls = QuadtreeGPT
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
            pass
            # if not model.train_policy:
            #     input_size = (1, 3, config.dataset.preprocessing.crop_size, config.dataset.preprocessing.crop_size)
            #     model_summary_str = summary(model, input_size=input_size, depth=5,
            #     col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
            #     logger.info(model_summary_str)
        elif model_type in ["maskgit"]:
            pass
            # input_size = (1, config.model.vq_model.num_latent_tokens)
            # input_data = [
            #     torch.randint(0, config.model.vq_model.codebook_size, input_size),
            #     torch.ones(1, dtype=int)
            # ]
            # model_summary_str = summary(
            #     model, input_data=input_data, depth=7,
            #     col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
            # logger.info(model_summary_str)
        elif model_type in ["mar", "mar-causal", "mar-quadtree", "gpt-quadtree"]:
            pass
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
        if config.losses.get("repa_param", None) is not None:
            loss_cls = ReconstructionLoss_Single_Stage_Repa
        else:
            loss_cls = ReconstructionLoss_Single_Stage
    elif model_type == "maskgit":
        loss_cls = MLMLoss
    elif model_type in ["titok", "one_d_piece", "quadtok"] and model.train_policy:
        loss_cls = ReconstructionLoss_Reward
    elif model_type in ["mar", "mar-causal", "mar-quadtree", "gpt-quadtree"]:
        loss_cls = DiffLoss # fake loss here
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

    if need_discrminator:
        if (config.model.type in ["titok", "one_d_piece", "quadtok"]):
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
    # logger.info("Creating dataloaders.")
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
        train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader
    
        return train_dataloader, eval_dataloader
    elif dataset_type == "pre_tokenized":
        pretokenized_params = dict(
            num_train_examples=config.experiment.max_train_examples,
            per_gpu_batch_size=config.training.per_gpu_batch_size,
            global_batch_size=total_batch_size_without_accum,
            num_workers_per_gpu=dataset_config.num_workers_per_gpu,
        )
        dataset = PretokenizedDataset(
            shards_path=dataset_config.shards_path,
            **pretokenized_params,
        )
    
        return dataset.dataloader
    


def create_evaluator(config, logger, accelerator):
    """Creates evaluator."""
    logger.info("Creating evaluator.")
    if config.model.vq_model.get("quantize_mode", "vq") == "vq":
        evaluator = VQGANEvaluator(
            device=accelerator.device,
            enable_rfid=True,
            enable_inception_score=True,
            enable_codebook_usage_measure=True,
            enable_codebook_entropy_measure=False,
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
                        # f"Repa Loss: {autoencoder_logs['train/repa_loss']:0.4f} "
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
                    # length = config.model.generator.image_seq_len
                    # input_tokens = tokenizer.encode(images)[1]["min_encoding_indices"]
                    # input_tokens = input_tokens[:,:,:length]
                    # assert input_tokens.shape[2] == length, f"Expected input tokens shape {length}, got {input_tokens.shape[2]}"
                    # input_tokens = input_tokens.reshape(images.shape[0], -1)
                    with open("/mnt/petrelfs/jianglihan/my_code/quadtok/fixed_quadtree_low.json", 'r') as f:
                        tree_dict_json = json.load(f)
                    final_tree_dict = {}
                    num_patch_side_list = config.model.selector.num_patch_side_list
                    patch_size_list = config.model.selector.patch_size_list
                    resolution = 256
                    node_positions = []
                    node_incides = []
                    for lod_idx, node_dict in tree_dict_json['final_tree'].items():
                        patch_size = patch_size_list[int(lod_idx)]
                        nodes = []
                        for node_info in node_dict:
                            node = QuadTreeNode(
                                patch_index=node_info['patch_index'],
                                lod_level=node_info['lod_level']
                            )
                            
                            row, col = divmod(node.patch_index, num_patch_side_list[int(lod_idx)])
                            y_start, x_start = row * patch_size, col * patch_size
                            y_end, x_end = y_start + patch_size, x_start + patch_size
                            node_position = [x_start / resolution, y_start / resolution, x_end / resolution, y_end / resolution]
                            node_positions.append(node_position)
                            nodes.append(node)
                            node_incides.append(int(lod_idx))
                        final_tree_dict[int(lod_idx)] = nodes
                    node_positions = torch.tensor(node_positions, device=accelerator.device, dtype=images.dtype)
                    node_incides = torch.tensor(node_incides, device=accelerator.device, dtype=torch.long)[1:]
                    tree_list = [copy.deepcopy(final_tree_dict) for _ in range(images.shape[0])]
                    image_latent = tokenizer.encode(images)
                    z_batch = tokenizer.selector._forward_optimize(image_latent, tree_list)
                    z_quantized_batch, result_dict = tokenizer.quantize(z_batch)
                    input_tokens = result_dict['min_encoding_indices'].squeeze(1)

                elif config.model.generator_type in ["mar", "mar-causal"]:
                    posterior = tokenizer.encode_generation(images)
                    breakpoint()
                    input_tokens = posterior.sample().mul_(0.2325)
                elif config.model.generator_type in ["mar-quadtree", "gpt-quadtree"]:
                    with open("/mnt/petrelfs/jianglihan/my_code/quadtok/fixed_quadtree_low.json", 'r') as f:
                        tree_dict_json = json.load(f)
                    final_tree_dict = {}
                    num_patch_side_list = config.model.selector.num_patch_side_list
                    patch_size_list = config.model.selector.patch_size_list
                    resolution = 256
                    node_positions = []
                    node_incides = []
                    for lod_idx, node_dict in tree_dict_json['final_tree'].items():
                        patch_size = patch_size_list[int(lod_idx)]
                        nodes = []
                        for node_info in node_dict:
                            node = QuadTreeNode(
                                patch_index=node_info['patch_index'],
                                lod_level=node_info['lod_level']
                            )
                            
                            row, col = divmod(node.patch_index, num_patch_side_list[int(lod_idx)])
                            y_start, x_start = row * patch_size, col * patch_size
                            y_end, x_end = y_start + patch_size, x_start + patch_size
                            node_position = [x_start / resolution, y_start / resolution, x_end / resolution, y_end / resolution]
                            node_positions.append(node_position)
                            nodes.append(node)
                            node_incides.append(int(lod_idx))
                        final_tree_dict[int(lod_idx)] = nodes
                    node_positions = torch.tensor(node_positions, device=accelerator.device, dtype=images.dtype)
                    node_incides = torch.tensor(node_incides, device=accelerator.device, dtype=torch.long)[1:]
                    tree_list = [copy.deepcopy(final_tree_dict) for _ in range(images.shape[0])]
                    
                    # tree_root = build_probabilistic_quadtree(
                    #     config.model.selector.num_patch_side_list, 
                    #     guaranteed_depth=3, 
                    #     expansion_probs=[0.3, 0.3]
                    # )
                    # final_tree = tree_to_decision_nodes_dict(tree_root, 6)
                    # tree_list = [copy.deepcopy(final_tree) for _ in range(images.shape[0])]
                    image_latent = tokenizer.encode(images)
                    z_batch = tokenizer.selector._forward_optimize(image_latent, tree_list)

                    if tokenizer.quantize_mode == "vae":
                        z_quantized_batch = tokenizer.quantize(z_batch).sample()
                        target_tokens =  z_quantized_batch.permute(0, 3, 2, 1).squeeze(2).to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True)
                        
                    elif tokenizer.quantize_mode == "vq":
                        z_quantized_batch, result_dict = tokenizer.quantize(z_batch)
                        target_tokens = result_dict['min_encoding_indices'].squeeze(1)
                    input_tokens = target_tokens[:, :-1].clone()
         
        elif "code_indices" in batch:
            # use batch 0 for debugging:
            target_tokens = batch["code_indices"].to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True)
            input_tokens = target_tokens[:, :-1] # remove the last token

            conditions = batch["class_id"].to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True)
            tree_dict = dict(lod_indices=batch["lod_indices"].to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True), patch_indices=batch["patch_indices"].to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True))
        else:
            raise ValueError(f"Not found valid keys: {batch.keys()}")
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
            elif config.model.generator_type in ["mar", "mar-causal"]:
                loss, loss_dict = model(input_tokens, conditions)
            elif config.model.generator_type in ["mar-quadtree", "gpt-quadtree"]:
                loss, loss_dict = model(input_tokens, target_tokens, tree_dict, conditions)
            elif config.model.generator_type in ["dit"]:
                loss, loss_dict = model(target_tokens, tree_dict, conditions)
            # Gather the losses across all processes for logging.
            loss_logs = {}
            for k, v in loss_dict.items():
                if k not in ["z", "token_logits"]:
                    loss_logs["train/" + k] = accelerator.gather(v).mean().item()
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
                if config.model.generator_type in ["mar-quadtree"]:
                    logger.info(
                        f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_meter.val:0.4f} "
                        f"LR: {lr:0.6f} "
                        f"Step: {global_step + 1} "
                        f"Total Loss: {loss_logs['train/total_loss']:0.4f} "
                        f"Diff Loss: {loss_logs['train/diff_loss']:0.4f} "
                        # f"Expand Acc: {loss_logs['train/expand_acc']:0.4f} "
                        # f"Expand Loss: {loss_logs['train/expand_loss']:0.4f} "
                    )
                else:
                    if "train/correct_tokens" in loss_logs.keys():
                        logger.info(
                            f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                            f"Batch (t): {batch_time_meter.val:0.4f} "
                            f"LR: {lr:0.6f} "
                            f"Step: {global_step + 1} "
                            f"Loss: {loss_logs['train/total_loss']:0.4f} "
                            f"Token Acc: {loss_logs['train/correct_tokens']:0.4f} "
                        )
                    else:
                        logger.info(
                            f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                            f"Batch (t): {batch_time_meter.val:0.4f} "
                            f"LR: {lr:0.6f} "
                            f"Step: {global_step + 1} "
                            f"Loss: {loss_logs['train/total_loss']:0.4f} "
                            f"Token Acc: {loss_logs['train/acc']:0.4f} "
                        )
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(loss_logs)
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
                # try:
                generate_images(
                    model,
                    tokenizer,
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    config=config
                )
                # except Exception as e:
                #     logger.error(f"Error generating images: {e}")
                #     continue
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
        if isinstance(reconstructed_images, dict):
            reconstructed_images = reconstructed_images[5]
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

    if isinstance(reconstructed_images, dict):
        reconstructed_images = reconstructed_images[5]

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


@torch.no_grad()
def prediction_images(model_output, tokenizer, input_tokens, accelerator, 
                    global_step, output_dir, logger, config=None):
    tokenizer.eval()
    predicted_token, tree = model_output
    with torch.no_grad():
        decoded_imgs = tokenizer.decoder._forward_optimize(predicted_token, tree)
        reconstructed_images = tokenizer.decoder._forward_optimize(input_tokens, tree)

    images_for_saving, images_for_logging = make_viz_from_samples(
        reconstructed_images,
        reconstructed_images=decoded_imgs
    )

    # Log images.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {f"Train Prediction": images_for_saving},
            step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {f"Train Prediction": images_for_logging}, step=global_step
        )
    # Log locally.
    root = Path(output_dir) / "train_prediction_images"
    os.makedirs(root, exist_ok=True)
    for i,img in enumerate(images_for_saving):
        filename = f"{global_step:08}_s-{i:03}-prediction.png"
        path = os.path.join(root, filename)
        img.save(path)
    return


def cleanup_old_checkpoints(output_dir, keep_num=5, logger=None):
    """Keep only the latest N checkpoints and delete the rest.
    
    Args:
        output_dir: Directory containing checkpoints
        keep_num: Number of latest checkpoints to keep (default: 5)
        logger: Optional logger for logging deletion info
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return
    
    # Find all checkpoint directories
    checkpoint_dirs = list(output_path.glob("checkpoint-*"))
    
    if len(checkpoint_dirs) <= keep_num:
        return
    
    # Sort checkpoints by global_step (extracted from directory name)
    def get_step(checkpoint_dir):
        try:
            return int(checkpoint_dir.name.split("-")[-1])
        except (ValueError, IndexError):
            return -1
    
    checkpoint_dirs.sort(key=get_step, reverse=True)
    
    # Delete old checkpoints, keeping only the latest keep_num
    checkpoints_to_delete = checkpoint_dirs[keep_num:]
    for checkpoint_dir in checkpoints_to_delete:
        try:
            shutil.rmtree(checkpoint_dir)
            if logger:
                logger.info(f"Deleted old checkpoint: {checkpoint_dir}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to delete checkpoint {checkpoint_dir}: {e}")


def save_checkpoint(model, output_dir, accelerator, global_step, logger) -> Path:
    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # Cleanup old checkpoints after saving (only on main process)
    if accelerator.is_main_process:
        cleanup_old_checkpoints(output_dir, keep_num=1, logger=logger)

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

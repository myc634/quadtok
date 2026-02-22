import glob
import json
import math
import os
import pickle
import random
import warnings
from hashlib import md5
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
import torch.utils.data as data
# from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.video_utils import VideoClips
from tqdm import tqdm


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L161
def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())

# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L400
def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


def calc_dataset_md5(dataset):
    try:
        md5_val =  md5(json.dumps(dataset.__dict__, sort_keys=True).encode('utf-8')).hexdigest()
    except Exception as e:
        print(f'Failed to calculate md5 for dataset: {e}. Using pickle instead.')
        data_bytes = pickle.dumps(dataset)
        md5_val = md5(data_bytes).hexdigest()
    return md5_val

    
class FeatureStats:

    def __init__(
        self,
        capture_all=False,
        capture_mean_cov=False,
        max_items=None,
        only_stats_mode=False,
        loaded_mean=None,
        loaded_cov=None,
    ):
        self.only_stats_mode = only_stats_mode
        if only_stats_mode:
            # load pre-computed mean and cov
            assert loaded_mean is not None and loaded_cov is not None, 'loaded_mean and loaded_cov must be provided in only_stats_mode'
            self.loaded_mean = loaded_mean
            self.loaded_cov = loaded_cov
        else:
            assert loaded_mean is None and loaded_cov is None, 'loaded_mean and loaded_cov must be None if only_stats_mode is False'
            self.loaded_mean = self.loaded_cov = None
            self.capture_all = capture_all
            self.capture_mean_cov = capture_mean_cov
            self.max_items = max_items
            self.num_items = 0
            self.num_features = None
            self.all_features = None
            self.raw_mean = None
            self.raw_cov = None

    def set_num_features(self, num_features):
        if self.only_stats_mode:
            raise ValueError('Cannot set num_features in only_stats_mode')

        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        if self.only_stats_mode:
            return True
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        if self.only_stats_mode:
            raise ValueError('Cannot append in only_stats_mode')

        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1):
        if self.only_stats_mode:
            raise ValueError('Cannot append in only_stats_mode')

        assert isinstance(x, torch.Tensor) and x.ndim == 2
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.float().cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        if self.only_stats_mode:
            return self.loaded_mean, self.loaded_cov
        else:
            if self.capture_mean_cov:
                mean = self.raw_mean / self.num_items
                cov = self.raw_cov / self.num_items
                cov = cov - np.outer(mean, mean)
                
            elif self.capture_all:
                features = self.get_all()
                mean = np.mean(features, axis=0)
                cov = np.cov(features, rowvar=False)

            else:
                raise ValueError('No stats captured')
            
            return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(stats_path):
        stats_path = Path(stats_path)
        if stats_path.suffix == '.pkl': # pickle file, load as FeatureStats
            with open(stats_path, 'rb') as f:
                s = edict(pickle.load(f))
            obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
            obj.__dict__.update(s)
        elif stats_path.suffix == '.npz': # npz file, ADM's precomputed mean and cov
            data = np.load(stats_path)
            obj = FeatureStats(only_stats_mode=True, loaded_mean=data['mu'], loaded_cov=data['sigma'])
        else:
            raise ValueError(f'Unknown file extension: {stats_path}')
        return obj


class FVDCalculator:
    def __init__(self, i3d_path=None, device='cuda'):
        if i3d_path is None:
            # https://github.com/universome/fvd-comparison/blob/master/compare_models.py#L34
            # Downloaded from https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1
            i3d_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'i3d_torchscript.pt')
        assert os.path.exists(i3d_path), f'Could not find i3d model at {i3d_path}'
        self.device = device
        self.i3d = torch.jit.load(i3d_path).eval().to(device)
        if torch.distributed.is_initialized():
            self.num_gpus = torch.distributed.get_world_size() # use world_size as num_gpus
        else:
            self.num_gpus = torch.cuda.device_count() # use local device count as num_gpus

    @torch.inference_mode()
    def get_feature_stats_for_batch(self, batch, feats=None, num_gpus=None):
        if num_gpus is None:
            num_gpus = self.num_gpus
        if feats is None:
            feats = FeatureStats(capture_mean_cov=True)

        if isinstance(batch, Dict):
            if 'gt' in batch:
                data = batch['gt']
            elif 'video' in batch:
                data = batch['video']
            else:
                raise ValueError('Expected key `gt` or `video` in the batch dict.')
        else:
            data = batch
            
        # Note: the used i3d torchscript model expects input in [-1, 1] and size=224x224.
        # if setting resize=True, the input will be resized to 224x224
        # if setting rescale=True, model expects input in [0, 255] and the model will rescale 
        # the input to [-1, 1] internally.
            
        # Here we assume data is in [0, 1] and BCTHW
        # so wee need to rescale data to [-1, 1] without setting rescale        
        assert isinstance(data, torch.Tensor) and data.ndim == 5 # BCTHW
        data = (data - 0.5) * 2
        features = self.i3d(data.to('cuda'), resize=True, return_features=True)
        feats.append_torch(features, num_gpus=num_gpus)
        return feats
    
    def get_feature_stats_for_dataset(
            self, 
            dataset: Dataset, 
            bs=32, 
            cache_stats=True,
            num_workers=4,
            stats_pkl_path=None
        ): # always using a single gpu
        assert isinstance(dataset, Dataset), f'Expected a torch Dataset, but got {type(dataset)}'
        
        if hasattr(dataset, 'csv_file'):
            dataset_name = Path(dataset.csv_file).stem 
        else:
            dataset_name = 'unknown'

        if cache_stats:
            if stats_pkl_path is None:
                dataset_md5 = calc_dataset_md5(dataset)
                stats_pkl = f'fvd_stats_{dataset_name}_{dataset_md5}.pkl'
                stats_cache_path = Path(__file__).resolve().parent / 'stats_cache'
                stats_cache_path.mkdir(exist_ok=True)
                stats_pkl_path = stats_cache_path / stats_pkl

            if stats_pkl_path.exists():
                return FeatureStats.load(stats_pkl_path)
        
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            if rank == 0:
                feats = self.calculate_stats_for_dataset(dataset, bs, num_workers)
                if cache_stats:
                    feats.save(stats_pkl_path)
            torch.distributed.barrier()
            if rank != 0:
                assert stats_pkl_path.exists()
                feats = FeatureStats.load(stats_pkl_path)

        else:
            feats = self.calculate_stats_for_dataset(dataset, bs, num_workers)
            if cache_stats:
                feats.save(stats_pkl_path)
        
        return feats
    
    def calculate_stats_for_dataset(self, dataset, bs=32, num_workers=4):
        feats = FeatureStats(capture_mean_cov=True)
        loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)
        for batch in tqdm(loader, desc='Extracting features'):
            feats = self.get_feature_stats_for_batch(batch, feats, num_gpus=1)
        return feats

    def calculate_fvd(self, feats_gen, feats_real):
        mu_gen, cov_gen = feats_gen.get_mean_cov()
        mu_real, cov_real = feats_real.get_mean_cov()

        # updated for better numerical stability
        mu_gen = torch.from_numpy(mu_gen)
        cov_gen = torch.from_numpy(cov_gen)
        mu_real = torch.from_numpy(mu_real)
        cov_real = torch.from_numpy(cov_real)

        mean = torch.sum((mu_gen - mu_real) ** 2)
        sqrt_trace_component = trace_sqrt_product(cov_gen, cov_real)
        trace = torch.trace(cov_gen + cov_real) - 2.0 * sqrt_trace_component
        fvd = trace + mean
            
        return fvd

    def calculate_fvd_with_dataset(self, feats_gen, dataset_real, bs=32, cache_stats=True):
        feats_real = self.get_feature_stats_for_dataset(dataset_real, bs, cache_stats)
        return self.calculate_fvd(feats_gen, feats_real)
    
    def calculate_fvd_with_video_folder(
        self, 
        feats_real, 
        video_folder, 
        bs=32,
        num_workers=4,
        sequence_length=16, 
        resolution=128,
        cache_stats=False
    ):
        dataset_gen = VideoDataset(
            data_folder=video_folder,
            sequence_length=sequence_length,
            resolution=resolution,
            sample_every_n_frames=1
        )
        feats_gen = self.get_feature_stats_for_dataset(dataset_gen, bs, cache_stats, num_workers=num_workers)
        return self.calculate_fvd(feats_gen, feats_real)
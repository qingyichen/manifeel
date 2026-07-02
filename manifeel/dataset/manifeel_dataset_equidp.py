from typing import Dict
import torch
import numpy as np
import os
import copy
import cv2

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class ManifeelDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            zarr_path: str, 
            horizon = 1,
            pad_before = 0,
            pad_after = 0,
            n_obs_steps=2,
            seed=42,
            val_ratio = 0.0,
            max_train_episodes=None
            ):
        
        print('Data path:', zarr_path)
        assert os.path.isdir(zarr_path)
        
        print(f"pad_before={pad_before}, pad_after={pad_after}, horizon={horizon}")

        super().__init__()

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            # 'tactile' force grids share the rgb load path (grid array -> CHW) and the
            # per-channel min-max normalizer keyed on the 'tactile_force_field' name in
            # get_normalizer; only the obs encoder treats them differently.
            if type in ('rgb', 'tactile'):
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        print("rgb_keys", rgb_keys)
        print("lowdim_keys", lowdim_keys)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        #TODO: Remove "_img" suffix from the camera view in upcoming demo data
        data_keys = rgb_keys + lowdim_keys
        data_keys.append('action')
        
        print("data_keys", data_keys)
    
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=data_keys)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer= self.replay_buffer, 
            sequence_length= horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.train_mask = train_mask
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        
        # normalizer for action and state
        data = {
            'action': self.replay_buffer['action'],
            'state': self.replay_buffer['state']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # normalizer for image / tactile force field
        for key in self.rgb_keys:
            attr = self.shape_meta['obs'].get(key, {})
            # `norm: image` (baseline only) forces the legacy [0,1]->[-1,1] image map;
            # on a ~1e-3 N force grid that collapses every taxel to ~-1, i.e. a dead
            # channel -- this reproduces the original faulty TacFF baseline. Without it
            # TacFF gets the correct per-channel min-max [-1,1] normalizer (one (min,max)
            # per channel shared across taxels, equalising the channels' ~4x scale gap
            # while preserving each channel's spatial contrast).
            if 'tactile_force_field' in key and attr.get('norm') != 'image':
                normalizer[key] = self._tacff_normalizer(key)
            else:
                normalizer[key] = get_image_range_normalizer()

        return normalizer

    def _tacff_normalizer(self, key):
        """Per-channel min-max [-1,1] normalizer for a tactile force field.

        The policy applies obs in (..., C, H, W) layout, and LinearNormalizer maps
        each field's flattened trailing dims via reshape(-1, C*H*W). So per-channel
        stats are broadcast to a full (C, H, W) scale/offset whose row-major flatten
        matches that layout — giving every taxel in a channel the same scale.
        """
        C, H, W = self.shape_meta['obs'][key]['shape']
        ff = self.replay_buffer[key][:].astype(np.float32)   # (N, H, W, C) as stored
        ch = ff.reshape(-1, ff.shape[-1])                    # (N*H*W, C)
        ch_min = ch.min(0)
        ch_max = ch.max(0)
        rng = np.clip(ch_max - ch_min, 1e-4, None)           # guard near-flat channels
        scale_c  = 2.0 / rng                                 # (C,)
        offset_c = -1.0 - ch_min * scale_c
        bc = lambda v: np.broadcast_to(
            v[:, None, None], (C, H, W)).astype(np.float32)  # (C,) -> (C,H,W)
        return SingleFieldLinearNormalizer.create_manual(
            scale=bc(scale_c), offset=bc(offset_c),
            input_stats_dict={
                'min':  bc(ch_min),         'max':  bc(ch_max),
                'mean': bc(ch.mean(0)),     'std':  bc(ch.std(0)),
            })

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):     
        # print(f"🚀 Inside _sample_to_data()")
        # print(f"✅ sample keys: {sample.keys()}")

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            if key in self.shape_meta['obs']:
           
                image_seq = sample[key] # horizon, H, W, C, np.float32 [0, 1]
                
                # resize the img to target shape in shape_meta
                target_shape = self.shape_meta['obs'][key]['shape']
                target_h, target_w = target_shape[1], target_shape[2]
                resized_image_seq = np.array([
                    cv2.resize(image, (target_w, target_h))
                    for image in image_seq
                ], dtype=np.float32)
                
                obs_dict[key] = np.moveaxis(resized_image_seq[T_slice], -1, 1) # image data [0, 1], (horizon, 3, 256, 256)
                
                # delete to save RAM
                del sample[key]

        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][T_slice].astype(np.float32) # (horizon, 7)
            # delete to save RAM
            del sample[key]

        data = {
            'obs': obs_dict,
            'action': sample['action'].astype(np.float32) # T, 6
        }

        # print(f"[degug1] Data structure: {data.keys()}")
        # print(f"[degug2] obs keys: {data['obs'].keys()}") 
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)

        # print(f"🔍 obs_dict keys before return: {torch_data['obs'].keys()}")

        return torch_data

if __name__=='__main__':
    import hydra
    from omegaconf import OmegaConf
    import pathlib

    # allows arbitrary python code execution in configs using the ${eval:''} resolver
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    @hydra.main(
        version_base=None,
        config_path=str(pathlib.Path(__file__).resolve().parents[1].joinpath('config')),
        config_name='train_diffusion_workspace.yaml'
    )
    def main(cfg: OmegaConf):
        OmegaConf.resolve(cfg)
        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        print("🚀Dataset length: ", len(dataset))

        for test_id in range(0, 10):
            side_img = dataset[test_id]['obs']['wrist']
            state = dataset[test_id]['obs']['state']
            action = dataset[test_id]['action']
            print("Obs shapes: ", side_img.shape, state.shape)
            print("Img Obs range: ",torch.min(side_img), torch.max(side_img))
            print("Action shape: ", action.shape)
            print("✅Action range: ", torch.min(action), torch.max(action))
        print("Finished")

        # normalizer = dataset.get_normalizer()
        # print(normalizer)

        # from matplotlib import pyplot as plt
        # normalizer = dataset.get_normalizer()
        # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
        # diff = np.diff(nactions, axis=0)
        # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)

    main()
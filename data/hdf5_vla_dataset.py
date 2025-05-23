import os
import fnmatch
import json

import yaml
import cv2
import numpy as np

from configs.state_vec import STATE_VEC_IDX_MAPPING

import torch
import random

class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """
    def __init__(self) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        self.DATASET_NAME = "human_mani"

        self.METALIST = "/data/home/tanhengkai/cobot-magic-vm/assets/metadatas/rdt_training_human_meta_file.list"

        self.first_train = True
        
        self.data_preload()

        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
    
        # Get each episode's len
        episode_lens = []
        for json_file in self.file_paths:
            item_name = self.get_item_name(json_file)
            valid, res = self.parse_tensor_file_state_only(item_name['tensor_path'])
            # valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)


    def get_item_name(self,json_file):
        with open(json_file, 'r') as f:
            meta_dat = json.load(f)

        return {
            "video_path": meta_dat['video_path'],
            "tensor_path": meta_dat['video_path'].replace('.mp4', '_qpos.pt'),
            "caption": meta_dat['raw_caption']['long caption'],
        }


    def data_preload(self):
        
        first_load = True
        with open(self.METALIST,'r') as f:
            lines = f.readlines()
            self.file_paths = [line.strip() for line in lines]

            crop_param = {
                'cam_high': (0, 0, 480, 640),
                'cam_left_wrist': (480, 0, 240, 320),
                'cam_right_wrist': (480, 320, 240, 320)
            }

            # crate large dataset dict
            self.dataset_dict = {}

            for file_path in self.file_paths:
                item_name = self.get_item_name(file_path)
                key = item_name['video_path'].split('/')[-2] + '/' + item_name['video_path'].split('/')[-1].replace('.mp4', '')

                assert(key not in self.dataset_dict and "warning key duplicated")

                video_path = item_name['video_path']
                tensor_path = item_name['tensor_path']

                cam_left_wrist = []
                cam_right_wrist = []
                cam_high = []
                cap = cv2.VideoCapture(video_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    for key, (x, y, h, w) in crop_param.items():
                        if key == 'cam_high':
                            cam_high.append(frame[x:x+h, y:y+w])
                        elif key == 'cam_left_wrist':
                            cam_left_wrist.append(frame[x:x+h, y:y+w])
                        elif key == 'cam_right_wrist':
                            cam_right_wrist.append(frame[x:x+h, y:y+w])

                    if first_load:
                        # save image to check
                        from PIL import Image
                        from torchvision.transforms.functional import to_pil_image
                        image = to_pil_image(frame)
                        image.save(f"all.jpg")
                        image = to_pil_image(cam_left_wrist[-1])
                        image.save(f"left.jpg")
                        image = to_pil_image(cam_right_wrist[-1])
                        image.save(f"right.jpg")
                        image = to_pil_image(cam_high[-1])
                        image.save(f"high.jpg")
                        first_load = False

                cam_left_wrist = np.array(cam_left_wrist)
                cam_high = np.array(cam_high)
                cam_right_wrist = np.array(cam_right_wrist)

                qpos = torch.load(tensor_path).cpu().numpy()

                self.dataset_dict[key] = {
                    "caption": item_name['caption'],
                    "cam_left_wrist": cam_left_wrist,
                    "cam_high": cam_high,
                    "cam_right_wrist": cam_right_wrist,
                    "qpos": qpos,
                    "tensor_path": tensor_path,
                }

        return 
            
    
    def __len__(self):
        return 100000000
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                key,value = random.choice(list(self.dataset_dict.items()))
            else:
                key,value = list(self.dataset_dict.items())[index]

            valid, sample = self.parse_tensor_file(value['cam_high'], value['cam_left_wrist'], value['cam_right_wrist'], value['qpos'], value['caption']) \
                if not state_only else self.parse_tensor_file_state_only(value['tensor_path'])
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))


    def parse_tensor_file(self, cam_high, cam_left_wrist, cam_right_wrist, qpos, caption):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        
        # with open(json_path,'r') as f:
        #     data = json.load(f)
        # caption = data['raw_caption']['long caption']
        # video_path = data['video_path']

        first_idx = 1

        #qpos  (N, 14)

        num_steps = qpos.shape[0]
        # [Optional] We drop too-short episode
        if num_steps < 128:
            return False, None
        
        # We randomly sample a timestep
        step_id = np.random.randint(first_idx-1, num_steps)
        
        # Assemble the meta
        meta = {
            "dataset_name": self.DATASET_NAME,
            "#steps": num_steps,
            "step_id": step_id,
            "instruction": caption,
        }

        # state = qpos / 4
        # action = qpos / 44

        # Rescale gripper to [0, 1]

        # 6  # 6.54230118e-02  5.26302719
        # 13  # 7.57226944e-02  5.31681633


        target_qpos = qpos[step_id:step_id+self.CHUNK_SIZE] / np.array(
            [[1, 1, 1, 1, 1, 1, 5.26302719, 1, 1, 1, 1, 1, 1, 5.31681633]] 
        )

        qpos = qpos / np.array(
            [[1, 1, 1, 1, 1, 1, 5.26302719, 1, 1, 1, 1, 1, 1, 5.31681633]]
        )
        
        # Parse the state and action
        state = qpos[step_id:step_id+1]
        state_std = np.std(qpos, axis=0)
        state_mean = np.mean(qpos, axis=0)
        state_norm = np.sqrt(np.mean(qpos**2, axis=0))
        actions = target_qpos
        if actions.shape[0] < self.CHUNK_SIZE:
            # Pad the actions using the last action
            actions = np.concatenate([
                actions,
                np.tile(actions[-1:], (self.CHUNK_SIZE-actions.shape[0], 1))
            ], axis=0)
        
        # Fill the state/action into the unified vector
        def fill_in_state(values):
            # Target indices corresponding to your state space
            # In this example: 6 joints + 1 gripper for each arm
            UNI_STATE_INDICES = [
                STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
            ] + [
                STATE_VEC_IDX_MAPPING["left_gripper_open"]
            ] + [
                STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
            ] + [
                STATE_VEC_IDX_MAPPING["right_gripper_open"]
            ]
            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec
        state = fill_in_state(state)
        state_indicator = fill_in_state(np.ones_like(state_std))
        state_std = fill_in_state(state_std)
        state_mean = fill_in_state(state_mean)
        state_norm = fill_in_state(state_norm)
        # If action's format is different from state's,
        # you may implement fill_in_action()
        actions = fill_in_state(actions)

        def parse_img(images):
            imgs = images[max(step_id-self.IMG_HISORY_SIZE+1, 0):step_id+1]
            if imgs.shape[0] < self.IMG_HISORY_SIZE:
                # Pad the images using the first image
                imgs = np.concatenate([
                    np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                    imgs
                ], axis=0)
            return imgs

        # Parse the images
        # `cam_high` is the external camera image
        cam_high = parse_img(cam_high)
        # For step_id = first_idx - 1, the valid_len should be one
        valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
        cam_high_mask = np.array(
            [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
        )

        cam_left_wrist = parse_img(cam_left_wrist)
        cam_right_wrist = parse_img(cam_right_wrist)
        cam_left_wrist_mask = cam_high_mask.copy()
        cam_right_wrist_mask = cam_high_mask.copy()
        
        # Return the resulting sample
        # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
        # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
        # if the left-wrist camera is unavailable on your robot
        return True, {
            "meta": meta,
            "state": state,
            "state_std": state_std,
            "state_mean": state_mean,
            "state_norm": state_norm,
            "actions": actions,
            "state_indicator": state_indicator,
            "cam_high": cam_high,
            "cam_high_mask": cam_high_mask,
            "cam_left_wrist": cam_left_wrist,
            "cam_left_wrist_mask": cam_left_wrist_mask,
            "cam_right_wrist": cam_right_wrist,
            "cam_right_wrist_mask": cam_right_wrist_mask
        }

        
    def parse_tensor_file_state_only(self, tensor_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        qpos = torch.load(tensor_path) #(N, 14)
        num_steps = qpos.shape[0]
        # [Optional] We drop too-short episode
        if num_steps < 128:
            return False, None
        
        first_idx = 1

        target_qpos = qpos[1:] / np.array(
            [[1, 1, 1, 1, 1, 1, 5.26302719, 1, 1, 1, 1, 1, 1, 5.31681633]]
        )

        # Rescale gripper to [0, 1]
        qpos = qpos[:-1] / np.array(
            [[1, 1, 1, 1, 1, 1, 5.26302719, 1, 1, 1, 1, 1, 1, 5.31681633]]
        )

        # assert(False and "warn:state only")
        
        # Parse the state and action
        state = qpos[first_idx-1:]
        action = target_qpos[first_idx-1:]
        
        # Fill the state/action into the unified vector
        def fill_in_state(values):
            # Target indices corresponding to your state space
            # In this example: 6 joints + 1 gripper for each arm
            UNI_STATE_INDICES = [
                STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
            ] + [
                STATE_VEC_IDX_MAPPING["left_gripper_open"]
            ] + [
                STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
            ] + [
                STATE_VEC_IDX_MAPPING["right_gripper_open"]
            ]
            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec
        state = fill_in_state(state)
        action = fill_in_state(action)
        
        # Return the resulting sample
        return True, {
            "state": state,
            "action": action
        }

if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)

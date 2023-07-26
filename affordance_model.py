from typing import Tuple, Optional, Dict

import numpy as np
from matplotlib import cm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import torchvision.transforms.functional as TF # for rotating images by 22.5 degrees angles in predict_grasp

from common import draw_grasp
from collections import deque

def get_gaussian_scoremap(
        shape: Tuple[int, int],
        keypoint: np.ndarray,
        sigma: float=1, dtype=np.float32) -> np.ndarray:
    """
    Generate a image of shape=:shape:, generate a Gaussian distribtuion
    centered at :keypoint: with standard deviation :sigma: pixels.
    keypoint: shape=(2,)
    """
    coord_img = np.moveaxis(np.indices(shape),0,-1).astype(dtype)
    sqrt_dist_img = np.square(np.linalg.norm(
        coord_img - keypoint[::-1].astype(dtype), axis=-1))
    scoremap = np.exp(-0.5/np.square(sigma)*sqrt_dist_img)
    return scoremap

class AffordanceDataset(Dataset):
    """
    Transformational dataset.
    raw_dataset is of type train.RGBDataset
    """
    def __init__(self, raw_dataset: Dataset):
        super().__init__()
        self.raw_dataset = raw_dataset
    
    def __len__(self) -> int:
        return len(self.raw_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Transform the raw RGB dataset element into
        training targets for AffordanceModel.
        return:
        {
            'input': torch.Tensor (3,H,W), torch.float32, range [0,1]
            'target': torch.Tensor (1,H,W), torch.float32, range [0,1]
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx]
        # TODO: (problem 2) complete this method and return the correct input and target as data dict
        # Hint: Use get_gaussian_scoremap
        # Hint: https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html
        # ===============================================================================
        
        # convert data torch values back to np and retrieve rgb image, center_point, and angle
        img = data['rgb'].numpy() / 255.0 # scaling to [0,1]
        center = data['center_point'].numpy()
        old_angle = data['angle'].numpy() # might just be a float not numpy list
        
        #discretizing angle into one of 8 bins
        bins = list()
        for i in range(9):
            bins.append(i*22.5)
        bin_diff = bins - old_angle # may need [0,0] to access the angle value
        angle = bins[np.argmin(bin_diff)] # angle is the factor of 22.5 that is closest to old_angle
        
        # use the center point we retrieved from data as the keypoint
        kps = KeypointsOnImage([
            Keypoint(x=center[0], y=center[1])
        ], shape=img.shape) # may also need to add left and right as keypoints
        
        # rotate rgb image and keypoint by the discretized angle
        rot = iaa.Rotate(angle)
        image_aug, kps_aug = rot(image=img, keypoints=kps)
    
        # use get_gaussian_scoremap with the augmented centerpoint as the key point and the image's H and W as the shape to determine the target
        score_map = get_gaussian_scoremap((img.shape[0], img.shape[1]), np.reshape(kps_aug[0].coords, (2,)))
        #new_score_map = torch.unsqueeze(score_map, 0) # change score_map size from (128,128) to (1,128,128)
        #score_map.reshape((1, img.shape[0], img.shape[1]))
        #torch.permute(torch.from_numpy(score_map,(0,3,1,2))
        
        # format output into dict with input and target as keys, then converting np arrays into torch tensors before returning
        data = {
            'input': image_aug,
            'target': np.array(score_map, dtype=np.float32),
        }
        
        data_torch = dict()
        for key, value in data.items():
            data_torch[key] = torch.from_numpy(value)
            
        data_torch['input'] = (torch.permute(data_torch['input'], (2,0,1))).float()
        data_torch['target'] = torch.unsqueeze(data_torch['target'], 0)
        
        #print('input shape:', data_torch['input'].shape) # should be (3,128,128)
        #print('target shape:', data_torch['target'].shape) # should be (1,128,128)
            
        return data_torch


class AffordanceModel(nn.Module):
    def __init__(self, n_channels: int=3, n_classes: int=1, n_past_actions: int=0, **kwargs):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Sequential(
            # For simplicity:
            #     use padding 1 for 3*3 conv to keep the same Width and Height and ease concatenation
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.outc = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.past_actions = deque(maxlen=n_past_actions)

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_inc = self.inc(x)  # N * 64 * H * W
        x_down1 = self.down1(x_inc)  # N * 128 * H/2 * W/2
        x_down2 = self.down2(x_down1)  # N * 256 * H/4 * W/4
        x_up1 = self.upconv1(x_down2)  # N * 128 * H/2 * W/2
        x_up1 = torch.cat([x_up1, x_down1], dim=1)  # N * 256 * H/2 * W/2
        x_up1 = self.conv1(x_up1)  # N * 128 * H/2 * W/2
        x_up2 = self.upconv2(x_up1)  # N * 64 * H * W
        x_up2 = torch.cat([x_up2, x_inc], dim=1)  # N * 128 * H * W
        x_up2 = self.conv2(x_up2)  # N * 64 * H * W
        x_outc = self.outc(x_up2)  # N * n_classes * H * W
        return x_outc

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict affordance using this model.
        This is required due to BCEWithLogitsLoss
        """
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def get_criterion() -> torch.nn.Module:
        """
        Return the Loss object needed for training.
        Hint: why does nn.BCELoss does not work well?
        """
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray,
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """
        cmap = cm.get_cmap('viridis')
        in_img = np.moveaxis(input, 0, -1)
        pred_img = cmap(output[0])[...,:3]
        row = [in_img, pred_img]
        if target is not None:
            gt_img = cmap(target[0])[...,:3]
            row.append(gt_img)
        img = (np.concatenate(row, axis=1)*255).astype(np.uint8)
        return img


    def predict_grasp(
        self,
        rgb_obs: np.ndarray,
    ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        """
        Given an RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_img
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Note: torchvision's rotation is counter clockwise, while imgaug,OpenCV's rotation are clockwise.
        """
        device = self.device
        # TODO: (problem 2) complete this method (prediction)
        # Hint: why do we provide the model's device here?
        # ===============================================================================
        coord, angle = None, None
        
        # 1: Take RGB input
            # note: rgb_obs's values range from 0 to 255
        rgb_obs = rgb_obs/255.0
        
        # 2: Rotate to [0-7] * 22.5
        stack = np.empty((8,3,rgb_obs.shape[0], rgb_obs.shape[1]))
        for i in range(8):
            rot = iaa.Rotate(i*-22.5)
            rotated_rgb = rot(image = rgb_obs) # shape of rotated rgb is (H,W,3)
            
            rotated_rgb = np.transpose(rotated_rgb, (2,0,1)) # after this line shape of rotated rgb is (3,H,W)
            
            stack[i] = rotated_rgb
            
        # at this point stack's shape is: (8,3,H,W)
        
        # 3: stack batch wise
        stack = stack.astype(np.float32)
        #stack01 = stack#/255.0
        #stack1 = stack1.astype(np.float32)
        stacked_stack = torch.from_numpy(stack)
        stacked_stack.to(device)#, dtype=torch.float64)
        
        # 4: Feed into network
        affordance_map = self.predict(stacked_stack) # predict's returned tensor's values range from 0-1
        affordance_map = torch.clip(affordance_map,0,1) # affordance_map's values are from 0-1
        a_map = affordance_map.detach().numpy()
        
        # 5: Find the max affordance pixel across all 8 images
        arg_max = torch.argmax(affordance_map)
        coord_max = np.unravel_index(arg_max, affordance_map.shape)
        
        # ===============================================================================

        # TODO: (problem 3, skip when finishing problem 2) avoid selecting the same failed actions
        # ===============================================================================
        curr_action_val = float(a_map[coord_max])
        curr_bin = coord_max[0]
        larger_than_all = True
        for max_coord in list(self.past_actions): # max coord should be of shape (bin, c, y, x)
            bin_ = max_coord[0]
            if max_coord == coord_max: # if the current predicted action has already been seen we suppress it in affordance map
                c = np.array([max_coord[3],max_coord[2]])
                suppression_map = get_gaussian_scoremap((rgb_obs.shape[0], rgb_obs.shape[1]), np.reshape(c, (2,)), sigma = 4)
                a_map[bin_] -= suppression_map
                larger_than_all = False
            else:
                if curr_action_val < float(a_map[max_coord]):
                    larger_than_all = False
                
                #if curr_action_val > float(affordance_map[max_coord]):
                #    self.past_actions.append((coord_max[0], coord_max[3], coord_max[2]))
                    
        if larger_than_all:
            max_coord = coord_max
        else:
            aff_map = torch.from_numpy(a_map)
            arg_max = torch.argmax(aff_map)
            max_coord = np.unravel_index(arg_max, aff_map.shape)
            
        self.past_actions.append(max_coord)
        
        coord_max = max_coord
            
            # supress past actions and select next-best action
        # ===============================================================================
        
        # TODO: (problem 2) complete this method (visualization)
        # :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.
        # Hint: use common.draw_grasp
        # Hint: see self.visualize
        # Hint: draw a grey (127,127,127) line on the bottom row of each image.
        # ===============================================================================
        vis_img = None
        # ===============================================================================

        # 6: Recover grasping location and angle
        #angle = float(-22.5 * coord_max[0])
        angle = float(22.5 * coord_max[0])
        
        rot2 = iaa.Rotate(angle)
        kps = KeypointsOnImage([
            Keypoint(x=coord_max[3], y=coord_max[2])
        ], shape=rgb_obs.shape)
        rotated_kps = rot2(keypoints = kps)
        
        x = int(rotated_kps[0].coords[0,0])
        y = int(rotated_kps[0].coords[0,1])
        #coord = (int(rotated_kps[0].coords[0,0]),int(rotated_kps[0].coords[0,1]))
        if x > 127:
            x = 127
        if x < 0:
            x = 0
        if y > 127:
            y = 127
        if y < 0:
            y = 0
        coord = (x,y)
        
        draw_coord = (coord_max[3], coord_max[2])
        
        
        # 1: Iterate through all 8 pairs of (img, pred) and np.concatenate them together
        predictions = a_map
        
        left = list()
        right = list()
        for i in range(8):
            img = np.ascontiguousarray(stack[i].transpose((1,2,0))*255, dtype=np.uint8)
            # img shape is now (H,W,3)
            # img was recast as contiguous array to avoid 'polylines' error from draw_grasp below
            if i == coord_max[0]:
                draw_grasp(img, draw_coord, 0) # requires img to be shape (H,W,3)
                
            cmap = cm.get_cmap('viridis')
            pred = cmap(predictions[i][0])[...,:3]
            pred = np.ascontiguousarray(pred*255, dtype=np.uint8)
            # pred shape is now (H,W,3)
            
            pair = np.concatenate((img, pred), axis = 1)
            
            if i % 2 == 0: # 0,2,4,6
                if len(left) == 0:
                    left = pair
                else:
                    left = np.concatenate((left, pair), axis = 0)
            else: # 1,3,5,7
                if len(right) == 0:
                    right = pair
                else:
                    right = np.concatenate((right, pair), axis = 0)
                    
        vis_img = np.concatenate((left, right), axis = 1).astype(np.uint8)
        
        return coord, angle, vis_img


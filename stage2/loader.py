from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import cv2
import math
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from Affine_3d import AffineTransformation_3d as af3d

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2)
    b, c, d = -axis*math.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def get_new_u(u,n):
    """
    :param u: three column vector in svd u
    :param n: theta = n * 2 * pi / 5
    :return: new_u
    """
    theta = n * 2 * np.pi / 5
    u2,u1,u0 = u[:, 2].reshape(3, 1), u[:, 1].reshape(3, 1), u[:, 0].reshape(3, 1)
    matrix = rotation_matrix(u0.reshape(3,), theta)   # (3,3)
    new_u1 = np.dot(matrix,u1)
    new_u2 = np.dot(matrix,u2)
    new_u = np.hstack((u0,new_u1,new_u2))
    return new_u

def get_2d_img(df,ct_combination_2_path,ct_challenge_path,idx, threshold=120, pca=True):
    name = df.iloc[idx,0].split('-')
    ct_name, size, n = name[0], int(name[4]), int(name[5])
    # print(ct_name)
    if os.path.exists( os.path.join(ct_challenge_path,ct_name[:-4]+"_clean.npy") ):
        HUs = np.load(os.path.join(ct_challenge_path,ct_name[:-4]+"_clean.npy"))
    elif os.path.exists(os.path.join(ct_combination_2_path,ct_name[:-4]+"_clean.npy")):
        HUs = np.load(os.path.join(ct_combination_2_path,ct_name[:-4]+"_clean.npy"))

    z_center, y_center, x_center = df.iloc[idx,3], df.iloc[idx,2], df.iloc[idx,1]
    z_size, y_size, x_size = size, size, size

    if pca:
        af = af3d(HUs, (z_center, y_center, x_center), (z_size, y_size, x_size))
        u = af.crop_img_and_do_pca(threshold)
        if u is None:
            M = af.compute_matrix((z_center, y_center, x_center), u)
        else:
            new_u = get_new_u(u, n)
            M = af.compute_matrix((z_center, y_center, x_center), new_u)
        out = af.affine_transform(M)
        assert out.shape == (z_size, y_size, x_size)
    else:
        x_min, x_max = x_center - x_size // 2, x_center + x_size // 2 + 1
        y_min, y_max = y_center - y_size // 2, y_center + y_size // 2 + 1
        z_min, z_max = z_center - z_size // 2, z_center + z_size // 2 + 1
        out = HUs[0, z_min:z_max, y_min:y_max, x_min:x_max]
        assert out.shape == (z_size, y_size, x_size)
    mid = out.shape[0] // 2
    slice0 = cv2.resize(out[mid, :, :], dsize=(32, 32))
    slice1 = cv2.resize(out[:, mid, :], dsize=(32, 32))
    slice2 = cv2.resize(out[:, :, mid], dsize=(32, 32))
    slice0 = np.expand_dims(slice0, 2)
    slice1 = np.expand_dims(slice1, 2)
    slice2 = np.expand_dims(slice2, 2)
    out_2d = np.concatenate((slice0, slice1, slice2), axis=2)
    return out_2d


class PeDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, ct_combination_2_path,ct_challenge_path, threshold=120, pca=True,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.frame = df
        self.ct_combination_2_path = ct_combination_2_path
        self.ct_challenge_path = ct_challenge_path
        self.transform = transform
        self.threshold = threshold
        self.pca = pca

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_name = self.frame.iloc[idx, 0]
        image = get_2d_img(self.frame,ct_combination_2_path,ct_challenge_path,idx, self.threshold, self.pca)
        image = image.astype(np.float32) / 255
        label = self.frame.loc[idx, 'on_mask']
        sample = {'image': image, 'label': np.array(label),'file_name':img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, label, file_name = sample['image'], sample['label'], sample['file_name']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label),
                'file_name': file_name}


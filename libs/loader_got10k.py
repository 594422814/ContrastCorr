import numpy as np
import time
import random
import scipy.io
from PIL import Image
import cv2
import os
import torch
from os.path import exists, join, split
import libs.transforms_multi as transforms
from torchvision import datasets

import pdb


def framepair_loader(video_path, frame_start, frame_end):
	# read frame
	pair = []
	id_ = np.zeros(2)
	if frame_end > 50:
		id_[0] = random.randint(frame_start, frame_end-50)
		id_[1] = id_[0] + random.randint(1, 50)
	else:
		id_[0] = random.randint(frame_start, frame_end)
		id_[1] = random.randint(frame_start, frame_end)

	for ii in range(2):
		image_path = os.path.join(video_path, '{:08d}.jpg'.format(int(id_[ii])))
		# print(image_path)
		image = cv2.imread(image_path)
		h, w, _ = image.shape
		h = max(64, (h // 64) * 64)
		w = max(64, (w // 64) * 64)
		image = cv2.resize(image, (w,h))
		image = image.astype(np.uint8)
		pil_im = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
		pair.append(pil_im)
			
	return pair


class VidListv1(torch.utils.data.Dataset):
	# for warm up, random crop both
	def __init__(self, video_path, patch_size, rotate = 10, scale=1.2, is_train=True, moreaug= True):
		super(VidListv1, self).__init__()
		self.data_dir = video_path
		normalize = transforms.Normalize(mean = (128, 128, 128), std = (128, 128, 128))

		t = []
		if rotate > 0:
			t.append(transforms.RandomRotate(rotate))
		if scale > 0:
			t.append(transforms.RandomScale(scale))
		t.extend([transforms.RandomCrop(patch_size, seperate =moreaug), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
			  normalize])

		self.transforms = transforms.Compose(t)
		
		self.is_train = is_train
		self.read_list()

	def __getitem__(self, idx):
		while True:
			video_ = self.list[idx]
			frame_end = len(os.listdir(video_)) - 6   # 5 additional files in the folder
			if frame_end <=0:
				print("Empty video {}, skip to the next".format(self.list[idx]))
				idx += 1
			else:
				break

		pair_ = framepair_loader(video_, 1, frame_end)  # start frame: 1
		data = list(self.transforms(*pair_))
		return tuple(data)

	def __len__(self):
		return len(self.list)

	def read_list(self):
		video_list_path = self.data_dir + 'train/list.txt'
		filenames = open(video_list_path).readlines()
		self.list = [os.path.join(self.data_dir, 'train', video[:-1]) for video in filenames]


class VidListv2(torch.utils.data.Dataset):
	# for localization, random crop frame1
	def __init__(self, video_path, patch_size, window_len, rotate = 10, scale = 1.2, full_size = 640, is_train=True):
		super(VidListv2, self).__init__()
		self.data_dir = video_path
		self.window_len = window_len
		normalize = transforms.Normalize(mean = (128, 128, 128), std = (128, 128, 128))
		self.transforms1 = transforms.Compose([
						   transforms.RandomRotate(rotate),
						   # transforms.RandomScale(scale),
						   # transforms.RandomHorizontalFlip(p=0.1)
						   # transforms.RandomVerticalFlip(p=0.1)
						   transforms.ResizeandPad(full_size),
						   transforms.RandomCrop(patch_size),
						   transforms.ToTensor(),
						   normalize])			
		self.transforms2 = transforms.Compose([
						   transforms.ResizeandPad(full_size),
						   transforms.ToTensor(),
						   normalize])
		self.is_train = is_train
		self.read_list()

	def __getitem__(self, idx):
		while True:
			video_ = self.list[idx]
			frame_end = len(os.listdir(video_)) - 6   # 5 additional files in the folder
			if frame_end <=0:
				print("Empty video {}, skip to the next".format(self.list[idx]))
				idx += 1
			else:
				break

		pair_ = framepair_loader(video_, 1, frame_end)  # start frame: 1
		data1 = list(self.transforms1(*pair_))
		data2 = list(self.transforms2(*pair_))
		if self.window_len == 2:
			data = [data1[0],data2[1]]
		else:
			data = [data1[0],data2[1], data2[2]]
		return tuple(data)

	def __len__(self):
		return len(self.list)

	def read_list(self):
		video_list_path = self.data_dir + 'train/list.txt'
		filenames = open(video_list_path).readlines()
		self.list = [os.path.join(self.data_dir, 'train', video[:-1]) for video in filenames]

import numpy as np
import time
import random
import scipy.io
from PIL import Image
import cv2
import os
import torch
import torchvision
from os.path import exists, join, split
import libs.transforms_multi as transforms
from torchvision import datasets

import pdb


def list_sequences(root, set_ids):
    """ Lists all the videos in the input set_ids. Returns a list of tuples (set_id, video_name)

    args:
        root: Root directory to TrackingNet
        set_ids: Sets (0-11) which are to be used

    returns:
        list - list of tuples (set_id, video_name) containing the set_id and video_name for each sequence
    """
    sequence_list = []

    for s in set_ids:
        anno_dir = os.path.join(root, "TRAIN_" + str(s), "anno")

        sequences_cur_set = [(s, os.path.splitext(f)[0]) for f in os.listdir(anno_dir) if f.endswith('.txt')]
        sequence_list += sequences_cur_set

    return sequence_list


def framepair_loader(video_path, frame_start, frame_end):
	# read frame
	pair = []
	id_ = np.zeros(2)
	frame_num = frame_end - frame_start
	if frame_end > 50:
		id_[0] = random.randint(frame_start, frame_end - 50)
		id_[1] = id_[0] + random.randint(1, 50)
	else:
		id_[0] = random.randint(frame_start, frame_end)
		id_[1] = random.randint(frame_start, frame_end)

	for ii in range(2):
		image_path = os.path.join(video_path, '{}.jpg'.format(int(id_[ii])))
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

		t.extend([transforms.RandomCrop(patch_size, seperate = moreaug), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
			  normalize])

		self.transforms = transforms.Compose(t)
		self.list = list_sequences(video_path, set_ids=list(range(12))) # training sets: 0~11
		self.is_train = is_train

	def __getitem__(self, idx):
		while True:
			set_id = self.list[idx][0]
			vid_name = self.list[idx][1]
			frame_path = os.path.join(self.data_dir, "TRAIN_" + str(set_id), "zips", vid_name) 

			frame_end = len(os.listdir(frame_path)) - 1
			if frame_end <=0:
				print("Empty video {}, skip to the next".format(self.list[idx]))
				idx += 1
			else:
				break

		pair_ = framepair_loader(frame_path, 0, frame_end)
		data = list(self.transforms(*pair_))
		return tuple(data)

	def __len__(self):
		return len(self.list)


class VidListv2(torch.utils.data.Dataset):
	# for localization, random crop frame1
	def __init__(self, video_path, patch_size, window_len, rotate = 10, scale = 1.2, full_size = 640, is_train=True):
		super(VidListv2, self).__init__()
		self.data_dir = video_path
		self.window_len = window_len
		normalize = transforms.Normalize(mean = (128, 128, 128), std = (128, 128, 128))
		self.transforms1 = transforms.Compose([
						   transforms.RandomRotate(rotate),
						   transforms.ResizeandPad(full_size),
						   transforms.RandomCrop(patch_size),
						   transforms.ToTensor(),
						   normalize])			
		self.transforms2 = transforms.Compose([
						   transforms.ResizeandPad(full_size),
						   transforms.ToTensor(),
						   normalize])
		self.is_train = is_train
		self.list = list_sequences(video_path, set_ids=list(range(12))) # training sets: 0~11
			

	def __getitem__(self, idx):
		while True:
			set_id = self.list[idx][0]
			vid_name = self.list[idx][1]
			frame_path = os.path.join(self.data_dir, "TRAIN_" + str(set_id), "zips", vid_name) 
			frame_end = len(os.listdir(frame_path)) - 1
			if frame_end <=0:
				print("Empty video {}, skip to the next".format(self.list[idx]))
				idx += 1
			else:
				break

		pair_ = framepair_loader(frame_path, 0, frame_end)
		data1 = list(self.transforms1(*pair_))
		data2 = list(self.transforms2(*pair_))
		if self.window_len == 2:
			data = [data1[0],data2[1]]
		else:
			data = [data1[0],data2[1], data2[2]]
		return tuple(data)

	def __len__(self):
		return len(self.list)


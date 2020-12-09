import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.net_utils import NLM_NC_woSoft
from libs.utils import *
from libs.autoencoder import encoder3, decoder3, encoder_res18, encoder_res50

import pdb

class NLM_woSoft(nn.Module):
	"""
	Non-local mean layer w/o softmax on affinity
	"""
	def __init__(self):
		super(NLM_woSoft, self).__init__()

	def forward(self, in1, in2):
		n,c,h,w = in1.size()
		in1 = in1.view(n,c,-1)
		in2 = in2.view(n,c,-1)

		affinity = torch.bmm(in1.permute(0,2,1), in2)
		return affinity

def transform(aff, frame1):
	"""
	Given aff, copy from frame1 to construct frame2.
	INPUTS:
	 - aff: (h*w)*(h*w) affinity matrix
	 - frame1: n*c*h*w feature map
	"""
	b,c,h,w = frame1.size()
	frame1 = frame1.view(b,c,-1)
	frame2 = torch.bmm(frame1, aff)
	return frame2.view(b,c,h,w)

class normalize(nn.Module):
	"""Given mean: (R, G, B) and std: (R, G, B),
	will normalize each channel of the torch.*Tensor, i.e.
	channel = (channel - mean) / std
	"""

	def __init__(self, mean, std = (1.0,1.0,1.0)):
		super(normalize, self).__init__()
		self.mean = nn.Parameter(torch.FloatTensor(mean).cuda(), requires_grad=False)
		self.std = nn.Parameter(torch.FloatTensor(std).cuda(), requires_grad=False)

	def forward(self, frames):
		b,c,h,w = frames.size()
		frames = (frames - self.mean.view(1,3,1,1).repeat(b,1,h,w))/self.std.view(1,3,1,1).repeat(b,1,h,w)
		return frames

def create_flat_grid(F_size, GPU=True):
	"""
	INPUTS:
	 - F_size: feature size
	OUTPUT:
	 - return a standard grid coordinate
	"""
	b, c, h, w = F_size
	theta = torch.tensor([[1,0,0],[0,1,0]])
	theta = theta.unsqueeze(0).repeat(b,1,1)
	theta = theta.float()

	# grid is a uniform grid with left top (-1,1) and right bottom (1,1)
	# b * (h*w) * 2
	grid = torch.nn.functional.affine_grid(theta, F_size)
	grid[:,:,:,0] = (grid[:,:,:,0]+1)/2 * w
	grid[:,:,:,1] = (grid[:,:,:,1]+1)/2 * h
	grid_flat = grid.view(b,-1,2)
	if(GPU):
		grid_flat = grid_flat.cuda()
	return grid_flat


class Model_switchGTfixdot_swCC_Res(nn.Module):
	def __init__(self, encoder_dir = None, decoder_dir = None,
					   temp = None, pretrainRes = False, uselayer=4):
		'''
		For switchable concenration loss
		Using Resnet18
		'''
		super(Model_switchGTfixdot_swCC_Res, self).__init__()
		self.gray_encoder = encoder_res18(pretrained = pretrainRes, uselayer=uselayer)
		# self.gray_encoder = encoder_res50(pretrained = pretrainRes, uselayer=uselayer)
		
		self.rgb_encoder = encoder3(reduce = True)

		# self.nlm = NLM_woSoft()
		# testing stage: mutual correlation for affinity computation
		self.nlm = NLM_NC_woSoft()

		self.decoder = decoder3(reduce = True)
		
		self.temp = temp
		self.softmax = nn.Softmax(dim=1)
		self.cos_window = torch.Tensor(np.outer(np.hanning(40), np.hanning(40))).cuda()
		self.normalize = normalize(mean=[0.485, 0.456, 0.406],
								   std=[0.229, 0.224, 0.225])

		self.R = 8  # window size
		self.P = self.R * 2 + 1
		self.topk = 5

		if(not encoder_dir is None):
			print("Using pretrained encoders: %s."%encoder_dir)
			self.rgb_encoder.load_state_dict(torch.load(encoder_dir))
		if(not decoder_dir is None):
			print("Using pretrained decoders: %s."%decoder_dir)
			self.decoder.load_state_dict(torch.load(decoder_dir))

		for param in self.decoder.parameters():
			param.requires_grad = False
		for param in self.rgb_encoder.parameters():
			param.requires_grad = False


	def forward(self, gray1, gray2, color1=None, color2=None):
		gray1 = (gray1 + 1) / 2
		gray2 = (gray2 + 1) / 2

		gray1 = self.normalize(gray1)
		gray2 = self.normalize(gray2)

		Fgray1 = self.gray_encoder(gray1)
		Fgray2 = self.gray_encoder(gray2)

		aff = self.nlm(Fgray1, Fgray2)
		aff_norm = self.softmax(aff * self.temp)

		if(color1 is None):
			# for testing
			return aff_norm, Fgray1, Fgray2

		Fcolor1 = self.rgb_encoder(color1)
		Fcolor2 = self.rgb_encoder(color2)
		Fcolor2_est = transform(aff_norm, Fcolor1)
		pred2 = self.decoder(Fcolor2_est)

		Fcolor1_est = transform(aff_norm.transpose(1,2), Fcolor2)
		pred1 = self.decoder(Fcolor1_est)

		return pred1, pred2, aff_norm, aff, Fgray1, Fgray2


	def propagate_neighbor_frames(self, gray2, gray1, mask):
		# propagate the mask of gray1 to gray2
		gray1 = (gray1 + 1) / 2
		gray2 = (gray2 + 1) / 2

		gray1 = self.normalize(gray1)
		gray2 = self.normalize(gray2)

		Fgray1 = self.gray_encoder(gray1)
		Fgray2 = self.gray_encoder(gray2)
		
		# prepare mask
		b, feat_c, feat_h, feat_w = Fgray1.size()
		_, mask_c, _, _ = mask.size()

		pad_mask = F.pad(mask, (self.R, self.R, self.R, self.R), mode='replicate')  
		window_mask = F.unfold(pad_mask, kernel_size=self.P)
		window_mask = window_mask.reshape([b, mask_c, self.P*self.P, feat_h*feat_w])

		# affinity
		pad_Fgray1 = F.pad(Fgray1, (self.R, self.R, self.R, self.R), mode='constant', value=0) 
		window_Fgray1 = F.unfold(pad_Fgray1, kernel_size=self.P)
		window_Fgray1 = window_Fgray1.reshape([b, feat_c, self.P*self.P, feat_h*feat_w]) 
		Fgray2 = Fgray2.reshape([b, feat_c, 1, -1])  # [B, C, 1, window_num]

		aff = (Fgray2 * window_Fgray1).sum(dim=1)
		aff[aff == 0] = -1e10  # discount padding at edge for softmax
		aff = F.softmax(aff*self.temp, dim=1)  # temp: 1
		
		# top-k selection
		b, N1, N2 = aff.size()
		tk_val, tk_idx = torch.topk(aff, dim = 1, k = self.topk)
		tk_val_min, _ = torch.min(tk_val, dim=1)
		tk_val_min = tk_val_min.view(b, 1, N2)
		aff[tk_val_min > aff] = 0
		
		aff = aff.unsqueeze(1)
		# predicted mask of gray2
		out = (aff * window_mask).sum(dim=2).reshape([b, mask_c, feat_h, feat_w])
		
		return out


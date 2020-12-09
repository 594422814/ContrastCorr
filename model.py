import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.net_utils import NLM, NLM_dot, NLM_woSoft, NLM_NC_woSoft, Batch_Contrastive
from torchvision.models import resnet18
from libs.autoencoder import encoder3, decoder3, encoder_res18, encoder_res50

from libs.utils import *

import pdb

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


def coords2bbox(coords, patch_size, h_tar, w_tar):
	"""
	INPUTS:
	 - coords: coordinates of pixels in the next frame
	 - patch_size: patch size
	 - h_tar: target image height
	 - w_tar: target image widthg
	"""
	b = coords.size(0)
	center = torch.mean(coords, dim=1) # b * 2
	center_repeat = center.unsqueeze(1).repeat(1,coords.size(1),1)
	dis_x = torch.sqrt(torch.pow(coords[:,:,0] - center_repeat[:,:,0], 2))
	dis_x = torch.mean(dis_x, dim=1).detach()
	dis_y = torch.sqrt(torch.pow(coords[:,:,1] - center_repeat[:,:,1], 2))
	dis_y = torch.mean(dis_y, dim=1).detach()
	left = (center[:,0] - dis_x*2).view(b,1)
	left[left < 0] = 0
	right = (center[:,0] + dis_x*2).view(b,1)
	right[right > w_tar] = w_tar
	top = (center[:,1] - dis_y*2).view(b,1)
	top[top < 0] = 0
	bottom = (center[:,1] + dis_y*2).view(b,1)
	bottom[bottom > h_tar] = h_tar
	new_center = torch.cat((left,right,top,bottom),dim=1)
	return new_center


def dropout2d(img1, img2): 
	# drop same layers for all images
	if np.random.random() < 0.3:
		return img1, img2

	drop_ch_num = int(np.random.choice(np.arange(1, 3), 1)) 
	drop_ch_ind = np.random.choice(np.arange(3), drop_ch_num, replace=False)

	for dropout_ch in drop_ch_ind:
		img1[:, dropout_ch] = 0
		img2[:, dropout_ch] = 0

	img1 *= (3 / (3 - drop_ch_num))
	img2 *= (3 / (3 - drop_ch_num))
	return img1, img2


class track_match_comb(nn.Module):
	def __init__(self, pretrained, encoder_dir = None, decoder_dir = None, temp=1, Resnet = "r18", color_switch=True, coord_switch=True, contrastive=True):
		super(track_match_comb, self).__init__()

		if Resnet in "r18":
			self.gray_encoder = encoder_res18(pretrained=pretrained, uselayer=4)
		elif Resnet in "r50":
			self.gray_encoder = encoder_res50(pretrained=pretrained, uselayer=4)
		self.rgb_encoder = encoder3(reduce=True)
		self.decoder = decoder3(reduce=True)

		self.rgb_encoder.load_state_dict(torch.load(encoder_dir))
		self.decoder.load_state_dict(torch.load(decoder_dir))
		for param in self.decoder.parameters():
			param.requires_grad = False
		for param in self.rgb_encoder.parameters():
			param.requires_grad = False

		self.nlm = NLM_woSoft()
		self.cont_model = Batch_Contrastive(temp=temp)
		self.normalize = normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])
		self.softmax = nn.Softmax(dim=1)
		self.temp = temp
		self.grid_flat = None
		self.grid_flat_crop = None
		self.color_switch = color_switch
		self.coord_switch = coord_switch
		self.contrastive_flag = contrastive
		self.self_reconstruction = True


	def forward(self, img_ref, img_tar, warm_up=True, patch_size=None):
		n, c, h_ref, w_ref = img_ref.size()   # [b, 3, 256, 256]
		n, c, h_tar, w_tar = img_tar.size()
		# UVC algorithm uses the gray-scale images as the inputs. 
		# We do not use the gray-scale image but randomly drop the image channel, which slightly outperforms the gray-scale baseline.
		gray_ref = copy.deepcopy(img_ref)
		gray_tar = copy.deepcopy(img_tar)

		gray_ref = (gray_ref + 1) / 2
		gray_tar = (gray_tar + 1) / 2

		gray_ref = self.normalize(gray_ref)
		gray_tar = self.normalize(gray_tar)

		# following CorrFlow method, channel dropout
		gray_ref, gray_tar = dropout2d(gray_ref, gray_tar)

		Fgray1 = self.gray_encoder(gray_ref)
		Fgray2 = self.gray_encoder(gray_tar)
		Fcolor1 = self.rgb_encoder(img_ref)

		output = []

		if warm_up:
			aff = self.nlm(Fgray1, Fgray2)
			aff_norm = self.softmax(aff * self.temp)
			Fcolor2_est = transform(aff_norm, Fcolor1)
			color2_est = self.decoder(Fcolor2_est)
	
			output.append(color2_est)
			output.append(aff)

			if self.color_switch:
				Fcolor2 = self.rgb_encoder(img_tar)
				Fcolor1_est = transform(aff_norm.transpose(1,2), Fcolor2)
				color1_est = self.decoder(Fcolor1_est)
				output.append(color1_est)
		
			if self.self_reconstruction:
				self_aff1 = self.nlm(Fgray1, Fgray1)
				self_aff1 -= (torch.eye(self_aff1.size(-1)).unsqueeze(0) * 1e10).cuda()
				self_aff1_norm = self.softmax(self_aff1)
				Fcolor1_reconstruct = transform(self_aff1_norm, Fcolor1)
				color1_reconstruct = self.decoder(Fcolor1_reconstruct)
				output.append(color1_reconstruct)

		else:
			if(self.grid_flat is None):
				self.grid_flat = create_flat_grid(Fgray2.size())
			aff_ref_tar = self.nlm(Fgray1, Fgray2)
			aff_ref_tar = torch.nn.functional.softmax(aff_ref_tar * self.temp, dim=2)
			coords = torch.bmm(aff_ref_tar, self.grid_flat)
			center = torch.mean(coords, dim=1) # b * 2
			# new_c = center2bbox(center, patch_size, h_tar, w_tar)
			new_c = center2bbox(center, patch_size, Fgray2.size(2), Fgray2.size(3))
			# print("center2bbox:", new_c, h_tar, w_tar)

			Fgray2_crop = diff_crop(Fgray2, new_c[:,0], new_c[:,2], new_c[:,1], new_c[:,3], patch_size[1], patch_size[0])
			# print("HERE: ", Fgray2.size(), Fgray1.size(), Fgray2_crop.size())
			
			aff_p = self.nlm(Fgray1, Fgray2_crop)
			aff_norm = self.softmax(aff_p * self.temp)
			Fcolor2_est = transform(aff_norm, Fcolor1)
			color2_est = self.decoder(Fcolor2_est)
		
			Fcolor2_full = self.rgb_encoder(img_tar)
			Fcolor2_crop = diff_crop(Fcolor2_full, new_c[:,0], new_c[:,2], new_c[:,1], new_c[:,3], patch_size[1], patch_size[0])

			output.append(color2_est)
			output.append(aff_p)
			output.append(new_c*8) 
			output.append(coords)

			# color orthorganal
			if self.color_switch:
				Fcolor1_est = transform(aff_norm.transpose(1,2), Fcolor2_crop)
				color1_est = self.decoder(Fcolor1_est)
				output.append(color1_est)

			# coord orthorganal
			if self.coord_switch:
				aff_norm_tran = self.softmax(aff_p.permute(0,2,1) * self.temp)
				if self.grid_flat_crop is None:
					self.grid_flat_crop = create_flat_grid(Fp_tar.size()).permute(0,2,1).detach()
				C12 = torch.bmm(self.grid_flat_crop, aff_norm)
				C11 = torch.bmm(C12, aff_norm_tran)
				output.append(self.grid_flat_crop)
				output.append(C11)

			if self.self_reconstruction:
				self_aff1 = self.nlm(Fgray1, Fgray1)
				self_aff1 -= (torch.eye(self_aff1.size(-1)).unsqueeze(0) * 1e10).cuda()
				self_aff1_norm = self.softmax(self_aff1)
				Fcolor1_reconstruct = transform(self_aff1_norm, Fcolor1)
				color1_reconstruct = self.decoder(Fcolor1_reconstruct)
				output.append(color1_reconstruct)

			if self.contrastive_flag:
				# contrastive loss on a pair of features
				Fcolor2_est_batch, sparse_loss = self.cont_model(Fgray2_crop, Fgray1, Fcolor1)
				Fcolor2_est_batch = self.decoder(Fcolor2_est_batch)
				output.append(Fcolor2_est_batch)
				output.append(sparse_loss)

		return output


import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class LayerNorm(nn.Module):
	def __init__(self, eps=1e-5):
		super().__init__()
		self.register_parameter('gamma', None)
		self.register_parameter('beta', None)
		self.eps = eps

	def forward(self, x):
		if self.gamma is None:
			self.gamma = nn.Parameter(torch.ones(x.size()).cuda())
		if self.beta is None:
			self.beta = nn.Parameter(torch.zeros(x.size()).cuda())
		mean = torch.min(x, 1, keepdim=True)[0]
		std = x.std(1, keepdim=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta

class NLM(nn.Module):
	"""NLM layer, output affinity"""
	def __init__(self, is_norm = False, iso1 = True):
		super(NLM, self).__init__()
		self.is_norm = is_norm
		if is_norm:
			self.norm = LayerNorm()
		self.softmax = nn.Softmax(dim=1)
		self.iso1 = iso1

	def forward(self, in1, in2, return_unorm=False):
		n,c,h,w = in1.size()
		N = h*w
		in1 = in1.view(n,c,N)
		in2 = in2.view(n,c,N)
		affinity = torch.bmm(in1.permute(0,2,1), in2)

		for ii in range(n):
			if self.iso1:
				affinity[ii] = affinity[ii] - 0.5*torch.diag(affinity[ii]).view(-1,1).repeat(1,N) - 0.5*torch.diag(affinity[ii]).view(1,-1).repeat(N,1)
			else:
				diag_ = torch.diag(affinity[ii])
				for xx in range(N):
					affinity[ii,xx] -= 0.5 * diag_
				for yy in range(N):
					affinity[ii, :, yy] -= 0.5 * diag_
		aff = self.softmax(affinity)
		if(return_unorm):
			return aff,affinity
		else:
			return aff

def featureL2Norm(feature):
	epsilon = 1e-6
	norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
	return torch.div(feature,norm)


class NLM_dot(nn.Module):
	"""NLM layer, output affinity"""
	def __init__(self, is_norm = False, temp = 1, l2_norm = False):
		super(NLM_dot, self).__init__()
		self.is_norm = is_norm
		self.l2_norm = l2_norm
		if is_norm:
			self.norm = LayerNorm()
		self.softmax = nn.Softmax(dim=1)
		self.temp = temp

	def forward(self, in1, in2):
		n,c,h,w = in1.size()
		N = h*w
		in1 = in1.view(n,c,N)
		in2 = in2.view(n,c,N)
		if self.is_norm:
			in1 = self.norm(in1)
			in2 = self.norm(in2)

		if self.l2_norm:
			in1 = featureL2Norm(in1)
			in2 = featureL2Norm(in2)

		affinity = torch.bmm(in1.permute(0,2,1), in2)
		affinity = self.softmax(affinity*self.temp) # n*N*N
		return affinity


class NLM_woSoft(nn.Module):
	"""NLM layer, output affinity, no softmax"""
	def __init__(self, is_norm = False, l2_norm = False):
		super(NLM_woSoft, self).__init__()
		self.is_norm = is_norm
		self.l2_norm = l2_norm
		if is_norm:
			self.norm = LayerNorm()

	def forward(self, in1, in2):
		n,c,h,w = in1.size()
		N = h*w
		in1 = in1.view(n,c,-1)
		in2 = in2.view(n,c,-1)
		if self.is_norm:
			in1 = self.norm(in1)
			in2 = self.norm(in2)

		if self.l2_norm:
			in1 = featureL2Norm(in1)
			in2 = featureL2Norm(in2)

		affinity = torch.bmm(in1.permute(0,2,1), in2)
		return affinity


def MutualMatching(affinity):
    # mutual matching
	batch_size, h, w = affinity.size()
	# get max
	affinity_B_max, _ = torch.max(affinity, dim=1, keepdim=True)
	affinity_A_max, _ = torch.max(affinity, dim=2, keepdim=True)
	eps = 1e-5
	affinity_A = affinity/(affinity_A_max + eps)
	affinity_B = affinity/(affinity_B_max + eps)
	affinity = affinity*(affinity_A*affinity_B)
	return affinity


class NLM_NC_woSoft(nn.Module):
	"""NLM layer, output affinity, no softmax"""
	def __init__(self, is_norm = False, l2_norm = False):
		super(NLM_NC_woSoft, self).__init__()
		self.is_norm = is_norm
		self.l2_norm = l2_norm
		if is_norm:
			self.norm = LayerNorm()

	def forward(self, in1, in2):
		b,c,h1,w1 = in1.size()
		b,c,h2,w2 = in2.size()
		# reshape features for matrix multiplication
		in1 = in1.view(b,c,h1*w1).transpose(1,2) # size [b,c,h*w]
		in2 = in2.view(b,c,h2*w2) # size [b,c,h*w]
		# perform matrix mult.
		feature_mul = torch.bmm(in1, in2)
		affinity = MutualMatching(feature_mul)
		return affinity


class Batch_Contrastive(nn.Module):
	""" Feaure contrastive loss on batch """
	def __init__(self, temp = 1, is_norm = False, l2_norm = False):
		super(Batch_Contrastive, self).__init__()
		self.is_norm = is_norm
		self.l2_norm = l2_norm
		self.temp = temp
		self.MSE_Loss = torch.nn.MSELoss(reduction = 'mean')
		self.L1_Loss = torch.nn.L1Loss(reduction = 'mean')
		if is_norm:
			self.norm = LayerNorm()

	def forward(self, feat_2, feat_1, Fcolor1):
		# feat_2: target feature to be reconstructed     feat_1 and Fcolor1: source features
		b, feat_c, feat_h, feat_w = feat_1.size()

		# contrastive learning
		feat_1 = feat_1.permute(0,2,3,1).contiguous()  # dim: [B, Dim, H, W] -> [B, H, W, Dim]
		feat_1 = feat_1.view(-1, feat_c)               # [Num_embedding (B*H*W), dim]
		feat_2 = feat_2.permute(0,2,3,1).contiguous()  
		feat_2 = feat_2.view(-1, feat_c)  

		_, color_c, _, _ = Fcolor1.size() 
		Fcolor1 = Fcolor1.permute(0,2,3,1).contiguous()  
		Fcolor1 = Fcolor1.view(-1, color_c) 

		batch_affinity = feat_2.mm(feat_1.t())
		norm_batch_affinity = F.softmax(batch_affinity * self.temp, dim=-1)
		color_2_est = norm_batch_affinity.mm(Fcolor1)
		color_2_est = color_2_est.view(b, feat_h, feat_w, color_c).permute(0,3,1,2)

		# the mask to ignore the correlations of other batches
		mask = torch.zeros([b*feat_h*feat_w, b*feat_h*feat_w]).cuda()
		batch_len = feat_h * feat_w
		for i in range(b):
			start_x = i * batch_len
			start_y = i * batch_len
			mask[start_x:(start_x + batch_len), start_y:(start_y + batch_len)] = 1

		batch_sim = (norm_batch_affinity * mask).sum(-1)
		batch_loss = self.L1_Loss(batch_sim, torch.ones(b*feat_h*feat_w).cuda())
		return color_2_est, batch_loss


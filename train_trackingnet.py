# a combination of track and match
# 1. load fullres images, resize to 640**2
# 2. warmup: set random location for crop
# 3. loc-match: add attention
import os
import cv2
import sys
import time
import copy
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from libs.loader_trackingnet import VidListv1, VidListv2
import torch.backends.cudnn as cudnn
import libs.transforms_multi as transforms
from model import track_match_comb as Model
from libs.loss import L1_loss
from libs.concentration_loss import ConcentrationSwitchLoss as ConcentrationLoss
from libs.train_utils import save_vis, AverageMeter, save_checkpoint, log_current
from libs.utils import diff_crop

import pdb

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

############################## helper functions ##############################

def parse_args():
	parser = argparse.ArgumentParser(description='')

	# file/folder pathes
	parser.add_argument("--videoRoot", type=str, default="/data1/trackingnet/", help='train video path')  ### change the dataset path
	parser.add_argument("--encoder_dir",type=str, default='weights/encoder_single_gpu.pth', help="pretrained encoder")
	parser.add_argument("--decoder_dir",type=str, default='weights/decoder_single_gpu.pth', help="pretrained decoder")
	parser.add_argument('--resume', type=str, default='', metavar='PATH', help='path to latest checkpoint (default: none)')
	parser.add_argument("-c","--savedir",type=str,default="trackingnet_contrastive/", help='checkpoints path')
	parser.add_argument("--Resnet", type=str, default="r18", help="choose from r18 or r50")

	# main parameters
	parser.add_argument("--pretrainRes",action="store_true")
	parser.add_argument("--batchsize",type=int, default=16, help="batchsize")
	parser.add_argument('--workers', type=int, default=16)
	parser.add_argument("--patch_size", type=int, default=256, help="crop size for localization.")  
	parser.add_argument("--full_size", type=int, default=640, help="full size for one frame.")      
	parser.add_argument("--rotate",type=int,default=10, help='degree to rotate training images')
	parser.add_argument("--scale",type=float,default=1.2, help='random scale')
	parser.add_argument("--lr",type=float,default=1e-4, help='learning rate')
	parser.add_argument('--lr-mode', type=str, default='step')   # poly step 
	parser.add_argument("--window_len",type=int,default=2, help='number of images (2 for pair and 3 for triple)')
	parser.add_argument("--log_interval",type=int,default=10, help='')
	parser.add_argument("--save_interval",type=int,default=200, help='save model every x epoch')
	parser.add_argument("--momentum",type=float,default=0.9, help='momentum')
	parser.add_argument("--weight_decay",type=float,default=0.005, help='weight decay')
	parser.add_argument("--device", type=int, default=2, help="0~device_count-1 for single GPU, device_count for dataparallel.")
	parser.add_argument("--temp", type=int, default=1, help="temprature for softmax.")

	# set epoches
	parser.add_argument("--wepoch",type=int,default=100, help='warmup epoch') 
	parser.add_argument("--nepoch",type=int,default=200, help='max epoch') 
	parser.add_argument("--step",type=int,default=40, help='step for lr') 

	# concenration regularization
	parser.add_argument("--lc",type=float,default=0.3, help='weight of concentration loss')
	parser.add_argument("--lc_win",type=int,default=8, help='win_len for concentration loss')

	# orthorganal regularization
	parser.add_argument("--color_switch",type=float,default=0.1, help='weight of color switch loss')
	parser.add_argument("--coord_switch",type=float,default=0, help='weight of color switch loss')

	# contrastive loss
	parser.add_argument("--contrastive",type=float,default=1, help='weight of contrastive loss')

	
	print("Begin parser arguments.")
	args = parser.parse_args()
	assert args.videoRoot is not None
	if not os.path.exists(args.savedir):
		os.mkdir(args.savedir)
	args.savepatch = os.path.join(args.savedir,'savepatch')
	args.logfile = open(os.path.join(args.savedir,"logargs.txt"),"w") 
	args.multiGPU = args.device == torch.cuda.device_count()

	if not args.multiGPU:
		torch.cuda.set_device(args.device)
	if not os.path.exists(args.savepatch):
		os.mkdir(args.savepatch)

	args.vis = True
	if args.color_switch > 0:
		args.color_switch_flag = True
	else:
		args.color_switch_flag = False

	if args.coord_switch > 0:
		args.coord_switch_flag = True
	else:
		args.coord_switch_flag = False
	
	if args.contrastive > 0:
		args.contrastive_flag = True
	else:
		args.contrastive_flag = False

	try:
		from tensorboardX import SummaryWriter
		global writer
		writer = SummaryWriter()
	except ImportError:
		args.vis = False
	print(' '.join(sys.argv))
	print('\n')
	args.logfile.write(' '.join(sys.argv))
	args.logfile.write('\n')
	
	for k, v in args.__dict__.items():
		print(k, ':', v)
		args.logfile.write('{}:{}\n'.format(k,v))
	args.logfile.close()
	return args
	

def adjust_learning_rate(args, optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	if args.lr_mode == 'step':
		lr = args.lr * (0.5 ** (epoch // args.step))
	elif args.lr_mode == 'poly':
		lr = args.lr * (1 - epoch / args.nepoch) ** 0.9
	else:
		raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr
	

def create_loader(args):
	dataset_train_warm = VidListv1(args.videoRoot, args.patch_size, args.rotate, args.scale)
	dataset_train = VidListv2(args.videoRoot, args.patch_size, args.window_len, args.rotate, args.scale, args.full_size)

	if args.multiGPU:
		train_loader_warm = torch.utils.data.DataLoader(
			dataset_train_warm, batch_size=args.batchsize, shuffle = True, num_workers=args.workers, pin_memory=True, drop_last=True)
		train_loader = torch.utils.data.DataLoader(
			dataset_train, batch_size=args.batchsize, shuffle = True, num_workers=args.workers, pin_memory=True, drop_last=True)
	else:
		train_loader_warm = torch.utils.data.DataLoader(
			dataset_train_warm, batch_size=args.batchsize, shuffle = True, num_workers=0, drop_last=True)
		train_loader = torch.utils.data.DataLoader(
			dataset_train, batch_size=args.batchsize, shuffle = True, num_workers=0, drop_last=True)
	return train_loader_warm, train_loader


def train(args):
	loader_warm, loader = create_loader(args)
	cudnn.benchmark = True
	best_loss = 1e10
	start_epoch = 0

	model = Model(args.pretrainRes, args.encoder_dir, args.decoder_dir, temp = args.temp, Resnet = args.Resnet, 
				   color_switch = args.color_switch_flag, coord_switch = args.coord_switch_flag, contrastive = args.contrastive_flag)

	if args.multiGPU:
		model = torch.nn.DataParallel(model).cuda()
		closs = ConcentrationLoss(win_len=args.lc_win, stride=args.lc_win,
								   F_size=torch.Size((args.batchsize//torch.cuda.device_count(),2, args.patch_size//8, args.patch_size//8)), temp = args.temp)   
		closs = nn.DataParallel(closs).cuda()
		optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model._modules['module'].parameters()),args.lr)
	else:
		closs = ConcentrationLoss(win_len=args.lc_win, stride=args.lc_win,
								   F_size=torch.Size((args.batchsize,2,
													  args.patch_size//8,
													  args.patch_size//8)), temp = args.temp)     
		model.cuda()
		closs.cuda()
		optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),args.lr)

	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			start_epoch = checkpoint['epoch']
			best_loss = checkpoint['best_loss']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{} ({})' (epoch {})"
				  .format(args.resume, best_loss, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))		
			
	for epoch in range(start_epoch, args.nepoch):
		if epoch < args.wepoch:
			lr = adjust_learning_rate(args, optimizer, epoch)
			print("Base lr for epoch {}: {}.".format(epoch, optimizer.param_groups[0]['lr']))
			best_loss = train_iter(args, loader_warm, model, closs, optimizer, epoch, best_loss)
		else:
			lr = adjust_learning_rate(args, optimizer, epoch-args.wepoch)
			print("Base lr for epoch {}: {}.".format(epoch, optimizer.param_groups[0]['lr']))
			best_loss = train_iter(args, loader, model, closs, optimizer, epoch, best_loss)


def forward(frame1, frame2, model, warm_up, patch_size=None):
	n, c, h, w = frame1.size()
	if warm_up:
		output = model(frame1, frame2)
	else:
		output = model(frame1, frame2, warm_up=False, patch_size=[patch_size//8, patch_size//8])  
		new_c = output[2]
		color2_gt = diff_crop(frame2, new_c[:,0], new_c[:,2], new_c[:,1], new_c[:,3],
							  patch_size, patch_size)
		output.append(color2_gt)
	return output


def train_iter(args, loader, model, closs, optimizer, epoch, best_loss):
	losses = AverageMeter()
	batch_time = AverageMeter()
	c_losses = AverageMeter()

	MSE_Loss = torch.nn.MSELoss(reduction = 'mean')
	p_losses = AverageMeter()

	model.train()
	end = time.time()
	if args.coord_switch_flag:
		coord_switch_loss = nn.L1Loss()
		sc_losses = AverageMeter()
	if args.contrastive_flag:
		cont_losses = AverageMeter()
		s_losses = AverageMeter()

	if epoch < 1 or (epoch>=args.wepoch and epoch< args.wepoch+2):
		thr = None
	else:
		thr = 2.5

	for i,frames in enumerate(loader):
		frame1_var = frames[0].cuda()
		frame2_var = frames[1].cuda()
		
		if epoch < args.wepoch:
			output = forward(frame1_var, frame2_var, model, warm_up=True)
			color2_est = output[0]
			aff = output[1]
			b,x,_ = aff.size()
			color1_est = None
			if args.color_switch_flag:
				color1_est = output[2]
			loss_ = L1_loss(color2_est, frame2_var, 10, 10, thr=thr, pred1=color1_est, frame1_var = frame1_var)
			# frame self reconstruction 
			reconstruction_loss = L1_loss(output[-1], frame1_var, 10, 10, thr=thr)

			if epoch >= 10 and args.lc > 0:
				constraint_loss = torch.sum(closs(aff.view(b,1,x,x))) * args.lc
				c_losses.update(constraint_loss.item(), frame1_var.size(0))
				loss = loss_ + reconstruction_loss + constraint_loss
			else:
				loss = loss_ + reconstruction_loss

			if(i % args.log_interval == 0):
				save_vis(color2_est, frame2_var, frame1_var, frame2_var, args.savepatch)
		else:
			output = forward(frame1_var, frame2_var, model, warm_up=False, patch_size = args.patch_size)
			color2_est = output[0]
			aff = output[1]
			new_c = output[2]
			coords = output[3]
			Fcolor2_crop = output[-1]

			b,x,x = aff.size()
			color1_est = None
			count = 3
		
			constraint_loss = torch.sum(closs(aff.view(b,1,x,x))) * args.lc
			c_losses.update(constraint_loss.item(), frame1_var.size(0))

			if args.color_switch_flag:
				count += 1
				color1_est = output[count]
			# intra-video transformation
			loss_color = L1_loss(color2_est, Fcolor2_crop, 10, 10, thr=thr, pred1=color1_est, frame1_var = frame1_var)
			reconstruction_loss = L1_loss(output[-4], frame1_var, 10, 10, thr=thr)
			loss_ = loss_color + reconstruction_loss + constraint_loss

			if args.coord_switch_flag:
				count += 1
				grids = output[count]
				C11 = output[count+1]
				loss_coord = args.coord_switch * coord_switch_loss(C11, grids)
				loss = loss_ + loss_coord
				sc_losses.update(loss_coord.item(), frame1_var.size(0))				
			else:
				loss = loss_

			if args.contrastive_flag:
				sparse_loss = output[-2]
				color2_est_batch = output[-3]
				# intra- and inter-video transformation consistency
				loss_color2_contrastive = L1_loss(color2_est_batch, color2_est, 10, 10, thr=thr)
				# sparsity constraint for the batch-level affinity
				sparse_loss = sparse_loss.mean() * args.contrastive
				# total loss
				loss = loss + sparse_loss + loss_color2_contrastive    
				cont_losses.update(loss_color2_contrastive.item(), frame1_var.size(0))
				s_losses.update(sparse_loss.item(), frame1_var.size(0))

			if(i % args.log_interval == 0):
				save_vis(color2_est, Fcolor2_crop, frame1_var, frame2_var, args.savepatch, new_c)

		losses.update(loss.item(), frame1_var.size(0))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		batch_time.update(time.time() - end)
		end = time.time()			

		if epoch >= args.wepoch and args.contrastive_flag:
			logger.info('Epoch: [{0}][{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Total Loss {loss.val:.4f} ({loss.avg:.4f})\t '
				'Contrastive Loss {ct_loss.val:.4f} ({ct_loss.avg:.4f})\t '
				'Sparse Loss {s_loss.val:.4f} ({s_loss.avg:.4f})\t '
				'Constraint Loss {c_loss.val:.4f} ({c_loss.avg:.4f})\t '.format(
				epoch, i+1, len(loader), batch_time=batch_time, loss=losses, ct_loss=cont_losses, s_loss=s_losses, c_loss=c_losses))
		else:
			logger.info('Epoch: [{0}][{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Total Loss {loss.val:.4f} ({loss.avg:.4f})\t '
				'Constraint Loss {c_loss.val:.4f} ({c_loss.avg:.4f})\t '.format(
				epoch, i+1, len(loader), batch_time=batch_time, loss=losses, c_loss=c_losses))

		if((i + 1) % args.save_interval == 0):
			is_best = losses.avg < best_loss
			best_loss = min(losses.avg, best_loss)
			checkpoint_path = os.path.join(args.savedir, 'checkpoint_latest.pth.tar')
			save_checkpoint({
					'epoch': epoch + 1,
					'state_dict': model.state_dict(),
					'best_loss': best_loss,
				}, is_best, filename=checkpoint_path, savedir = args.savedir)
			log_current(epoch, losses.avg, best_loss, filename = "log_current.txt", savedir=args.savedir)

	return best_loss


if __name__ == '__main__':
	args = parse_args()
	train(args)
	writer.close()

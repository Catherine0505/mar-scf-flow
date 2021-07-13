from __future__ import print_function
import argparse
import sys
import os
from tqdm import tqdm
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils


import torch.optim as optim
import torch.optim.lr_scheduler as sched

from flow_modules.common_modules import Actnormlayer, InvertibleConv1x1, SqueezeLayer, Split2dMsC, TupleFlip, GaussianDiag
from flow_modules.affine_coupling import AffineCoupling
from flow_modules.mixlogcdf_coupling import MixLogCDFCoupling
from mar_prior.corr_prior import ChannelPriorMultiScale

from utils import get_dataset


class FlowStep(nn.Module):
	def __init__(self,in_channels, out_channels, hidden_channels,actnorm_scale,coupling_type):
		super(FlowStep, self).__init__()
		self.coupling_type = coupling_type
		if coupling_type == 'mixlogcdf':
			self.coupling = MixLogCDFCoupling(in_channels, hidden_channels, num_blocks=10, num_components=32, drop_prob=0.2)
			self.tuple_flip = TupleFlip()
		else:
			self.coupling = AffineCoupling(in_channels, out_channels, hidden_channels)

		self.actnormlayer = Actnormlayer(in_channels, actnorm_scale)
		self.invert_1x1_layer = InvertibleConv1x1(in_channels)

	def forward_inference(self,x,logdet=0.,reverse=False):
		x, logdet = self.actnormlayer(x,logdet,reverse)
		x, logdet = self.invert_1x1_layer(x, logdet,reverse)
		x, logdet = self.coupling(x, logdet,reverse)
		if self.coupling_type == 'mixlogcdf':
			x, logdet = self.tuple_flip(x, logdet,reverse)
		return x, logdet

	def reverse_sampling(self,x,logdet=0.,reverse=True):
		if self.coupling_type == 'mixlogcdf':
			x, logdet = self.tuple_flip(x, logdet,reverse)
		x, logdet = self.coupling(x, logdet,reverse)
		x, logdet = self.invert_1x1_layer(x, logdet,reverse)
		x, logdet = self.actnormlayer(x, logdet,reverse)
		return x, logdet

	def forward(self,input,logdet=0., reverse=False):
		if not reverse:
			z, logdet = self.forward_inference(input,logdet,reverse)
		else:
			z, logdet = self.reverse_sampling(input,logdet,reverse)
		return z, logdet


class FlowNet(nn.Module):
	def __init__(self, batch_size, image_shape, hidden_channels, K, L, coupling_type, actnorm_scale=1.0):
		super(FlowNet, self).__init__()
		self.layers = nn.ModuleList()
		self.output_shapes = []
		self.all_z = []
		self.K = K
		self.L = L
		H, W, C = image_shape
		assert C == 1 or C == 3, ("image_shape should be HWC, like (64, 64, 3)"
								  "C == 1 or C == 3")
		for i in range(L):
			# 1. Squeeze
			C, H, W = C * 4, H // 2, W // 2
			self.layers.append(SqueezeLayer(factor=2))#( 'row' if i%2==0 else 'col' ), _type = 'row'
			self.output_shapes.append([-1, C, H, W])
			# 2. K FlowStep
			for _ in range(K):
				self.layers.append(
					FlowStep(in_channels=C,out_channels=C, hidden_channels=hidden_channels,
						actnorm_scale=actnorm_scale, coupling_type=coupling_type))
				self.output_shapes.append(
					[-1, C, H, W])
			# 3. Split2d
			if i < L - 1:
				self.layers.append(Split2dMsC(C,i+1))
				self.output_shapes.append([-1, C//2 , H, W])
				C = C//2

		self.c_prior = ChannelPriorMultiScale(batch_size,3,32,32,L,mog=False,dp_rate=0,num_layers=3,hidden_size=32)


	def forward(self, input, logdet=0., reverse=False, eps_std=None):
		if not reverse:
			return self.encode(input, logdet)
		else:
			return self.decode(input, eps_std)

	def encode(self, z, logdet=0.0):
		for layer, shape in zip(self.layers, self.output_shapes):
			z, logdet = layer(z, logdet, reverse=False)
			if isinstance(layer, Split2dMsC):
				z1, z2 = z
				logdet = logdet + self.c_prior((z1, z2),layer.level,reverse=False)
				z = z1
		logdet = logdet + self.c_prior(z,self.L,reverse=False)
		return z, logdet

	def decode(self, z, eps_std=None):
		z = self.c_prior(z,self.L,reverse=True)
		for layer in reversed(self.layers):
			if isinstance(layer, Split2dMsC):
				z1 = z
				z2 = self.c_prior(z1,layer.level,reverse=True)
				z = (z1, z2)
			z, logdet = layer(z, logdet=0, reverse=True)
		return z




class MarScfFlow(nn.Module):
	def __init__(self,batch_size,image_shape,coupling_type, L, K, C):
		super().__init__()
		#L = 3
		self.flow = FlowNet( batch_size, image_shape = image_shape, hidden_channels=C, K=K, L=L, coupling_type=coupling_type )
		self.batch_size = batch_size

	def forward(self, x=None, z=None, eps_std=None, reverse=False):
		if not reverse:
			return self.normal_flow(x)
		else:
			return self.reverse_flow(z, eps_std)

	def normal_flow(self, x):
		x_shape = list(x.size())
		#uniform dequantization
		if x.is_cuda:
			z = x + torch.rand( x.size() ).cuda()* (1. / 256.)
			logdet = torch.zeros(x.size(0),).cuda()
		else:
			z = x + torch.rand( x.size() )* (1. / 256.)
			logdet = torch.zeros(x.size(0),)


		logdet = logdet+float(-np.log(256.)*x_shape[1]*x_shape[2]*x_shape[3])
		z, objective = self.flow(z, logdet=logdet, reverse=False)

		nll = (-objective) / float(np.log(2.)*x_shape[1]*x_shape[2]*x_shape[3])
		return z, nll, None

	def reverse_flow(self, z, eps_std):
		with torch.no_grad():
			x = self.flow(z, eps_std=eps_std, reverse=True)
		return x

	def load_my_state_dict(self, state_dict):
		own_state = self.state_dict()
		for name, param in state_dict.items():
			if name not in own_state:
				 continue
			if isinstance(param, nn.Parameter):
				param = param.data
			own_state[name].copy_(param)


class Quantizer(nn.Module):
	def __init__(self, embed_space):
		super().__init__()
		# self.num_embed = embed_space.weight.shape[0]
		# self.embed_dim = embed_space.weight.shape[1]
		self.embed_space = embed_space
		self.embed_space_weights = self.embed_space.weight
		# if torch.cuda.is_available():
		# 	self.embed_space = self.embed_space.cuda()

	def forward(self, input_data):
		N, C, H, W = input_data.shape
		data = input_data.permute((0, 2, 3, 1))
		data = data.reshape(-1, C)
		dist = (
            data.pow(2).sum(1).reshape(data.shape[0], 1).repeat(1, num_embed) - \
            2 * data @ self.embed_space_weights.transpose(0, 1) + \
            self.embed_space_weights.pow(2).sum(1).reshape(1, num_embed).repeat(data.shape[0], 1)
        )
		embed_ind = torch.argmin(dist, dim = 1)
		quantize = self.embed_space(embed_ind)
		quantize_detach = data + (quantize - data).detach()
		return quantize, quantize_detach, data


def save_samples( model, filename, samples ):
	if not os.path.exists('./samples/'):
		os.makedirs('./samples/')
	rev = model(None,None,reverse=True, eps_std=1.0)
	rev[torch.isnan(rev)] = -0.5
	rev = torch.clamp(rev, -0.5, 0.5)
	vutils.save_image(rev[0:samples].clone().detach().cpu(), filename, normalize=True)


def test_model( model, test_loader, num_gpu, quantizer, beta ):
	with torch.no_grad():
		all_nlls = []
		all_quantize_diff = []
		for i, data in enumerate(test_loader, 0):
			data_im = data[0]
			if num_gpu > 0:
				data_im = data_im.cuda()
			z, nll, _ = model( data_im, reverse=False)
			all_nlls.append( nll.detach().cpu().numpy() )
			quantize, quantize_detach, data_flatten = quantizer(z)
			all_quantize_diff.append(((quantize - data_flatten.detach()).pow(2).sum(1).mean().detach() + \
			beta * (quantize.detach() - data_flatten).pow(2).sum(1).mean().detach()).item())
		all_nlls = np.concatenate(all_nlls, axis=0)
	return np.mean(all_nlls), np.mean(all_quantize_diff)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_name', default='cifar10', type=str, help='Name of dataset in [cifar10,mnist,imagenet_32,imagenet_64].')
	parser.add_argument('--data_root', default=None, type=str, help='Name of dataset in [cifar10,mnist,imagenet_32,imagenet_64].')
	parser.add_argument('--coupling', default='affine', type=str, help='Type of coupling in [affine,mixlogcdf].')
	parser.add_argument('--batch_size', default=64, type=int, help='Batch Size.')
	parser.add_argument('--warm_up', default=10000, type=int, help='# of warmup steps.')
	parser.add_argument('--L', default=3, type=int, help='# of levels.')
	parser.add_argument('--K', default=32, type=int,  help='# of layers per level.')
	parser.add_argument('--C', default=512, type=int,  help='# of channels per layer.')
	parser.add_argument('--from_checkpoint', action='store_true', help='Evaluate Checkpoint.')
	parser.add_argument('--num_embed', default=512, type=int,  help='# of embedding vectors.')
	args = parser.parse_args()
	dataset_name = args.dataset_name
	data_root = args.data_root
	coupling_type = args.coupling
	batch_size = args.batch_size
	warm_up = args.warm_up
	L = args.L
	K = args.K
	C = args.C
	from_checkpoint = args.from_checkpoint
	num_embed = args.num_embed
	beta = 0.05
	setting_id = 'marscf_' + str(dataset_name) + '_' + str(coupling_type) + '_' + str(K) + '_' + str(C)

	if torch.cuda.is_available():
		num_gpu = torch.cuda.device_count()
	else:
		num_gpu = 0

	print('Num of GPUs found: ', num_gpu)

	train_loader, test_loader, image_shape = get_dataset(dataset_name, batch_size, data_root)
	mar_scf = MarScfFlow(batch_size//(num_gpu if num_gpu > 0 else 1), image_shape, coupling_type, L, K, C)#.to(device)
	embed_space = nn.Embedding(num_embed, 48, scale_grad_by_freq = True)
	if torch.cuda.is_available():
		embed_space = embed_space.cuda()
	quantizer = Quantizer(embed_space)
	embed_buffer = embed_space.weight.clone().detach().cpu().numpy()

	if not os.path.exists('./checkpoints/'):
		os.makedirs('./checkpoints/')

	if not from_checkpoint:
		if num_gpu > 0:
			mar_scf = nn.DataParallel(mar_scf, output_device=2).cuda()
		optimizerG = optim.Adamax(list(mar_scf.parameters()) + [embed_space.weight],lr=5*1e-6)#.cuda(2)
		scheduler = sched.LambdaLR(optimizerG, lambda s: min(1., s / warm_up))
		global_step = 0

		epochs = 100000
		test_epoch_interval = 1
		best_test_nll = 9999999.
		best_test_diff = 9999999.
		best_epoch_nll = -1
		best_epoch_diff = -1

		for epoch in range(epochs):
			train_bar = tqdm(train_loader)
			for data in train_bar:

				optimizerG.zero_grad()
				data_im = data[0]
				if num_gpu > 0:
					data_im = data_im.cuda()
				z, nll, _ = mar_scf(data_im, reverse = False)
				loss = torch.mean(nll)
				quantize, quantize_detach, data_flatten = quantizer(z)
				# quantize_reshape = quantize_detach.reshape(z.shape)
				# data_reconstruct = mar_scf(quantize_reshape, reverse = True)
				loss = loss + \
					(quantize - data_flatten.detach()).pow(2).sum(1).mean()
				loss.backward()
				# print(z.shape)

				optimizerG.step()
				scheduler.step(global_step)
				global_step += batch_size

				train_bar.set_description('Train NLL (bits/dim) %.2f | Epoch %d -- Iteration ' % (loss.item(),epoch))
				# print(type(embed_buffer))
				if torch.cuda.is_available():
					print(((torch.tensor(embed_buffer) == embed_space.weight.cpu()) == False).sum())
				else:
					print(((torch.tensor(embed_buffer) == embed_space.weight.cpu()) == False).sum())

			if epoch % test_epoch_interval == 0:

				tqdm.write('Evaluating model .... ')

				curr_test_nll, curr_test_diff = test_model( mar_scf, test_loader, num_gpu, quantizer, beta )

				if not math.isnan(curr_test_nll):
					if curr_test_nll < best_test_nll:
						torch.save(mar_scf.module.state_dict(), os.path.join('./checkpoints/', setting_id + '.pt'))
						best_test_nll = curr_test_nll
						best_epoch_nll = epoch
					if curr_test_diff < best_test_diff:
						best_test_diff = curr_test_diff
						best_epoch_diff = epoch
				tqdm.write('Best Test NLL (bits/dim) at Epoch %d -- %.3f \n' % (best_epoch_nll,best_test_nll))
				tqdm.write('Best Quantize Difference at Epoch %d -- %.3f \n' % (best_epoch_diff,best_test_diff))


	else:
		try:
			state_dict = torch.load( os.path.join('./checkpoints/', setting_id +'.pt'))
			mar_scf.load_my_state_dict(state_dict)
			print('Checkpoint loaded!')
		except Exception:
			print('Error loading checkpoint!')
			sys.exit(0)

		if num_gpu > 0:
			mar_scf = nn.DataParallel(mar_scf).cuda()

		print('Evaluating model on checkpoint .... ')
		curr_test_nll = test_model( mar_scf, test_loader, num_gpu )
		print('Test NLL (bits/dim):  %.3f' % curr_test_nll)
		save_samples( mar_scf, os.path.join('./samples/', setting_id + '.png'), samples=batch_size)

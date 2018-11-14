from __future__ import print_function

import torch
from torch import nn, optim
from torch .nn import functional as F

from torchvision .utils import save_image

import numpy as np


# hyperparameters
input_image_size = (480, 640)
input_image_channels = 3

image_dimensions = input_image_channels * input_image_size [0] * input_image_size [1]
feature_dimensions = 1000
encoding_dimensions = 40

learning_rate = 1e-3

# test hyperparameters
test_reconstruction_n = 8
test_sample_n = 8

def thing ():
	class thing (dict):
		def __init__(self):
			pass
		def __getattr__(self, attr):
			return self [attr]
		def __setattr__(self, attr, val):
			self [attr] = val
	return thing ()

def params ():
	import argparse
	import os
	import sys
	parser = argparse .ArgumentParser (description = 'vae x ducks')
	
	parser .add_argument ('--train', type = str, required = True, metavar = 'path', help = 'path to a folder containing training images for the vae')
	parser .add_argument ('--test', type = str, default = None, metavar = 'path', help = 'path to a folder containing test images for the vae (default: training dataset)')
	parser .add_argument ('--init', type = str, default = None, metavar = 'path', help = 'path to a trained model file for initializing training')

	parser .add_argument ('--learning-rate', type = float, default = learning_rate, metavar = 'n', help = 'learning rate for adam (default: ' + str (learning_rate) + ')')
	parser .add_argument ('--feature-dim', type = int, default = feature_dimensions, metavar = 'd', help = 'number of feature dimonsions (default: ' + str (feature_dimensions) + ')')
	parser .add_argument ('--encoding-dim', type = int, default = encoding_dimensions, metavar = 'd', help = 'number of encoding dimensions (default: ' + str (encoding_dimensions) + ')')
	
	parser .add_argument ('--batch-size', type = int, default = 10, metavar = 'n', help = 'batch size for training (default: 10)')
	parser .add_argument ('--epochs', type = int, default = 10, metavar = 'n', help = 'number of epochs to train (default: 10)')

	parser .add_argument ('--activation', type = str, default = 'relu', choices = ['relu', 'leaky_relu', 'selu'], metavar = 'a', help = 'activation function in the hidden layers (default: relu)')

	parser .add_argument ('--log-interval', type = int, default = 10, metavar = 's', help = 'how many batches to wait before logging training status (default: 10)')
	parser .add_argument ('--seed', type = int, default = 1, metavar = 's', help = 'random seed (default: 1)')
	parser .add_argument ('--no-cuda', action = 'store_true', default = False, help = 'disables CUDA training')

	parser .add_argument ('--out', type = str, default = None, metavar = 'path', help = 'path to a folder to store output')
	parser .add_argument ('--out-model', action = 'store_true', default = False, help = 'output model_n.pt')

	args = parser .parse_args ()

	trainer_args = thing ()
	trainer_args .train = args .train
	trainer_args .test = args .test or args .train
	trainer_args .learning_rate = args .learning_rate
	trainer_args .batch_size = args .batch_size
	trainer_args .epochs = args .epochs
	trainer_args .log_interval = args .log_interval
	trainer_args .seed = args .seed
	trainer_args .cuda = not args .no_cuda and torch .cuda .is_available ()
	trainer_args .init = args .init
	trainer_args .out = args .out
	trainer_args .out_model = args .out_model

	model_args = thing ()
	model_args .feature_dimensions = args .feature_dim
	model_args .encoding_dimensions = args .encoding_dim
	model_args .activation = args .activation

	os .makedirs (trainer_args .out, exist_ok = True)
	if os .listdir (trainer_args .out):
		print ('Warning: ' + trainer_args .out + ' is not empty!', file = sys .stderr)

	return trainer_args, model_args

def load_samples (path, cuda = True):
	import os
	import tempfile
	from torch .utils .data import DataLoader
	from torchvision import datasets, transforms

	image_folder_path = tempfile .TemporaryDirectory () .name
	os .makedirs (image_folder_path)
	os .symlink (os .path .realpath (path), os .path .join (image_folder_path, 'data'))

	cuda_args = {'num_workers': 1, 'pin_memory': True} if trainer_args .cuda else {}
	return DataLoader (
			dataset = datasets .ImageFolder (image_folder_path, transform = transforms .ToTensor ()),
			batch_size = trainer_args .batch_size,
			shuffle = True,
			**cuda_args)

def out_file (filename):
	import os
	return os .path .join (trainer_args .out, filename)

def load_state ():
	return torch .load (trainer_args .init) if trainer_args .init else {}
	
def save_state ():
	torch .save ((
			{ 'epoch': epoch
			, 'rng': torch .get_rng_state ()
			, 'model': model .state_dict ()
			, 'optimizer': optimizer .state_dict () })
			, out_file ('state_' + str (epoch) + '.pt'))
	if trainer_args .out_model:
		torch .save ({ 'model': model .state_dict () }
				, out_file ('model_' + str (epoch) + '.pt'))

class VAE (nn .Module):
    def __init__ (self, image_dimensions, feature_dimensions, encoding_dimensions, activation, **kwargs):
        super (VAE, self) .__init__ ()
        self .activation = activation
        self.img_dim = image_dimensions
        self.feat_dim = feature_dimensions
        self.encode_dim = encoding_dimensions

        self .fc1 = nn .Linear (image_dimensions, feature_dimensions)
        self .fc21 = nn .Linear (feature_dimensions, encoding_dimensions)
        self .fc22 = nn .Linear (feature_dimensions, encoding_dimensions)
        self .fc3 = nn .Linear (encoding_dimensions, feature_dimensions)
        self .fc4 = nn .Linear (feature_dimensions, image_dimensions)
        self.device = torch .device ('cuda' if torch.cuda.is_available() else 'cpu')

    def encode (self, x):
        if(type(x) is np.ndarray):
            x = x.reshape(-1, self.img_dim)
            x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)
        else:
            x = x.view(-1, self.img_dim)
        if self .activation == 'relu':
            h1 = F .relu (self .fc1 (x))
        elif self .activation == 'leaky_relu':
            h1 = F .leaky_relu (self .fc1 (x))
        elif self .activation == 'selu':
            h1 = F .selu (self .fc1 (x))
        else:
            raise Exception ('unknown activation', self .activation)
        return self .fc21 (h1), self .fc22 (h1)

    def reparameterize (self, mu, logvar):
        std = torch .exp (0.5 * logvar)
        eps = torch .randn_like (std)
        return eps .mul (std) .add_ (mu)

    def decode (self, z):
        if self .activation == 'relu':
            h3 = F .relu (self .fc3 (z))
        elif self .activation == 'leaky_relu':
            h3 = F .leaky_relu (self .fc3 (z))
        elif self .activation == 'selu':
            h3 = F .selu (self .fc3 (z))
        else:
            raise Exception ('unknown activation', self .activation)
        return torch .sigmoid (self .fc4 (h3))

    def forward (self, x):
        if(type(x) is np.ndarray):
            x = x.reshape(-1, self.img_dim)
            x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)
        else:
            x = x.view(-1, self.img_dim)
        mu, logvar = self .encode (x)
        z = self .reparameterize (mu, logvar)
        return self .decode (z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def objective (recon_x, x, mu, logvar):
	BCE = F .binary_cross_entropy (recon_x, x .view (-1, image_dimensions), reduction = 'sum')

	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# 0.5 * sum (1 + log (sigma^2) - mu^2 - sigma^2)
	KLD = -0.5 * torch .sum (1 + logvar - mu .pow (2) - logvar .exp ())

	return BCE + KLD


def train (epoch):
	model .train ()
	total_train_loss = 0
	for i, (batch_sample, _) in enumerate (train_sampler):
		batch_sample = batch_sample .to (device)
		optimizer .zero_grad ()
		recon_batch, mu, logvar = model (batch_sample)
		loss = objective (recon_batch, batch_sample, mu, logvar)
		loss .backward ()
		total_train_loss += loss .item ()
		optimizer .step ()
		if i % trainer_args .log_interval == 0:
			print ('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}' .format 
					( epoch
					, i * len (batch_sample)
					, len (train_sampler .dataset)
					, 100. * i / len (train_sampler)
					, loss .item () / len (batch_sample)))

	train_loss = total_train_loss / len (train_sampler .dataset)
	print ('====> Epoch: {} Average loss: {:.4f}' .format (epoch, train_loss))


def test (epoch):
	model .eval ()
	total_test_loss = 0
	with torch .no_grad ():
		for i, (batch_sample, _) in enumerate (test_sampler):
			batch_sample = batch_sample .to (device)
			recon_batch, mu, logvar = model (batch_sample)
			total_test_loss += objective (recon_batch, batch_sample, mu, logvar) .item ()
			if trainer_args .out and i == 0:
				test_batch_size = min (batch_sample .size (0), trainer_args .batch_size)
				n = min (test_batch_size, test_reconstruction_n)
				comparison = torch .cat (
						[ batch_sample [:n]
						, recon_batch .view (test_batch_size, input_image_channels, input_image_size [0], input_image_size [1]) [:n] ])
				save_image (comparison .cpu (), out_file ('reconstruction_' + str (epoch) + '.png'), nrow = n)

	test_loss = total_test_loss / len (test_sampler .dataset)
	print ('====> Test set loss: {:.4f}' .format (test_loss))
	if trainer_args .out:
		encoding_sample = torch .randn (test_sample_n ** 2, model_args .encoding_dimensions) .to (device)
		image_sample = model .decode (encoding_sample) .cpu ()
		save_image (image_sample .view (test_sample_n ** 2, input_image_channels, input_image_size [0], input_image_size [1])
				, out_file ('sample_' + str (epoch) + '.png'))



"""
trainer_args, model_args = params ()

torch .manual_seed (trainer_args .seed)

train_sampler = load_samples (trainer_args .train, trainer_args .cuda)
test_sampler = load_samples (trainer_args .test, trainer_args .cuda)

device = torch .device ('cuda' if trainer_args .cuda else 'cpu')

model = VAE (**model_args) .to (device)
optimizer = optim .Adam (model .parameters (), lr = trainer_args .learning_rate)

epoch_offset = 1

state = load_state ()
if 'rng' in state:
	torch .set_rng_state (state ['rng'])
if 'model' in state:
	model .load_state_dict (state ['model'])
if 'optimizer' in state:
	optimizer .load_state_dict (state ['optimizer'])
if 'epoch' in state:
	epoch_offset += state ['epoch']



for epoch in range (epoch_offset, epoch_offset + trainer_args .epochs):
	train (epoch)
	test (epoch)
	
	if trainer_args .out:
		save_state ()
"""

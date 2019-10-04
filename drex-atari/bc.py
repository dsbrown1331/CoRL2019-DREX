#!/usr/bin/env python
from cnn import Network
import numpy as np
import utils
import copy
import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable

class Imitator:
	def __init__(self, min_action_set,
				learning_rate,
				alpha,
				checkpoint_dir,
				hist_len,
				l2_penalty):
		self.minimal_action_set = min_action_set
		print("hist len", hist_len)
		self.network = Network(len(self.minimal_action_set), hist_len)
		if torch.cuda.is_available():
			print("Initializing Cuda Nets...")
			self.network.cuda()
		self.optimizer = optim.Adam(self.network.parameters(),
		lr=learning_rate, weight_decay=l2_penalty)
		self.checkpoint_directory = checkpoint_dir


	def predict(self, state):
		# predict action probabilities
		outputs = self.network(Variable(utils.float_tensor(state)))
		vals = outputs[len(outputs) - 1].data.cpu().numpy()
		return vals

	def get_action(self, state):
		vals = self.predict(state)
		return np.argmax(vals)

	# potentially optimizable
	def compute_labels(self, sample, minibatch_size):
		#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		labels = Variable(utils.long_tensor(minibatch_size))
		actions_taken = [x.action for x in sample]
		#print(actions_taken[0])
		for i in range(len(actions_taken)):
			#print(actions_taken[i])
			labels[i] = np.int(actions_taken[i])
		# The list of ALE actions taken for the minibatch
		#labels = torch.from_numpy(np.array([x.action for x in sample])).long().to(device)
		#for index in range(len(actions_taken)):
		#	labels[index] = torch.from_numpy(actions_taken[index])
		#print(labels[0])
		return labels

	def get_loss(self, outputs, labels):
		return nn.CrossEntropyLoss()(outputs, labels)

	def validate(self, dataset, minibatch_size):
		'''run dataset through loss to get validation error'''
		validation_data = dataset.get_dataset()
		v_loss = 0.0
		for i in range(0,len(validation_data) - minibatch_size,minibatch_size):
			sample = validation_data[i:i+minibatch_size]
			with torch.no_grad():
				state = Variable(utils.float_tensor(np.stack([np.squeeze(x.state) for x in sample])))
				#print(state.size())
				# compute the target values for the minibatch
				labels = self.compute_labels(sample, minibatch_size)
				#print(labels.size())
				#print("labels", labels)
				self.optimizer.zero_grad()
				'''
				Forward pass the minibatch through the
				prediction network.
				'''
				activations = self.network(state)
				'''
				Extract the Q-value vectors of the minibatch
				from the final layer's activations. See return values
				of the forward() functions in cnn.py
				'''
				output = activations[len(activations) - 1]
				loss = self.get_loss(output, labels)
				v_loss += loss
		return v_loss



	def train(self, dataset, minibatch_size):
		# sample a minibatch of transitions
		sample = dataset.sample_minibatch(minibatch_size)
		state = Variable(utils.float_tensor(np.stack([np.squeeze(x.state) for x in sample])))

		# compute the target values for the minibatch
		labels = self.compute_labels(sample, minibatch_size)
		#print("labels", labels)
		self.optimizer.zero_grad()
		'''
		Forward pass the minibatch through the
		prediction network.
		'''
		activations = self.network(state)
		'''
		Extract the Q-value vectors of the minibatch
		from the final layer's activations. See return values
		of the forward() functions in cnn.py
		'''
		output = activations[len(activations) - 1]
		loss = self.get_loss(output, labels)
		#self.losses.append(loss)
		loss.backward()
		self.optimizer.step()
		return loss

	'''
	Args:
	This function checkpoints the network.
	'''
	def checkpoint_network(self, env_name, extra_info):
		print("Checkpointing Weights")
		utils.save_checkpoint({
			'state_dict': self.network.state_dict()
			}, self.checkpoint_directory, env_name, extra_info)
		print("Checkpointed.")


class Clone:
	def __init__(self, min_action_set, hist_len, checkpoint_policy):
		self.minimal_action_set = min_action_set
		print("hist len", hist_len)
		self.network = Network(len(self.minimal_action_set), hist_len)
		self.network.load_state_dict(torch.load(checkpoint_policy)['state_dict'])
		if torch.cuda.is_available():
			print("Initializing Cuda Nets...")
			self.network.cuda()

	def predict(self, state):
		# predict action probabilities
		outputs = self.network(Variable(utils.float_tensor(state)))
		vals = outputs[len(outputs) - 1].data.cpu().numpy()
		return vals

	def get_action(self, state):
		vals = self.predict(state)
		return np.argmax(vals)

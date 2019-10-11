import os.path
import torch
import numpy as np
'''
checkpointing source:
https://blog.floydhub.com/checkpointing-tutorial-for-tensorflow-keras-and-pytorch/
'''
def save_checkpoint(state, checkpoint_dir, env_name, extra_info):
	filename = checkpoint_dir + '/' + env_name +'_' + extra_info + '_network.pth.tar'
	print("Saving checkpoint at " + filename + " ...")
	torch.save(state, filename)  # save checkpoint
	print("Saved checkpoint.")

def get_checkpoint(checkpoint_dir):
	resume_weights = checkpoint_dir + '/network.pth.tar'
	if torch.cuda.is_available():
		print("Attempting to load Cuda weights...")
		checkpoint = torch.load(resume_weights)
		print("Loaded weights.")
	else:
		print("Attempting to load weights for CPU...")
		# Load GPU model on CPU
		checkpoint = torch.load(resume_weights,
								map_location=lambda storage,
								loc: storage)
		print("Loaded weights.")
	return checkpoint

def long_tensor(input):
	if torch.cuda.is_available():
		return torch.cuda.LongTensor(input)
	else:
		return torch.LongTensor(input)

def float_tensor(input):
	if torch.cuda.is_available():
		return torch.cuda.FloatTensor(input)
	else:
		return torch.FloatTensor(input)

def perform_no_ops(ale, no_op_max, preprocessor, state):
	# perform nullops
	num_no_ops = np.random.randint(1, no_op_max + 1)
	for _ in range(num_no_ops):
		ale.act(0)
		preprocessor.add(ale.getScreenRGB())
	if len(preprocessor.preprocess_stack) < 2:
		ale.act(0)
		preprocessor.add(ale.getScreenRGB())
	state.add_frame(preprocessor.preprocess())

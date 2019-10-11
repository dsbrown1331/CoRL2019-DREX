from collections import namedtuple
import numpy as np

class State:
	def __init__(self, hist_len):
		# Initialize a 1 x hist_len x 84 x 84  state
		self.hist_len = hist_len
		self.state = np.zeros((1, hist_len, 84, 84), dtype=np.float32)
		'''
		index of next image, indicates that the first 84 x 84 
		image should be at (0, 0)
		'''
		self.insertLoc = 0

	def add_frame(self, img):
		self.state[0, self.insertLoc, ...] = img.astype(np.float32)/255.0
		# The index to insert at cycles from betweem 0, 1, 2, 3
		self.insertLoc = (self.insertLoc + 1) % self.hist_len

	def get_state(self):
		'''
		return the stacked four frames in the correct order
		Example: Suppose the state contains the following frames:
		[f4 f1 f2 f3]. The return value should be [f1 f2 f3 f4].
		Since the most recent frame inserted in this example was 
		f4 at index 0,  self.insertLoc equals 1. Thus, we 
		"roll" the image a single space to the left, resulting in 
		[f1 f2 f3 f4]
		'''
		return np.roll(self.state, 0 - self.insertLoc, axis=1)
import numpy as np
import cv2
from collections import deque

class Preprocessor:

	def __init__(self):
		self.preprocess_stack = deque([], 2)

	def add(self, aleRGB):
		self.preprocess_stack.append(aleRGB)

	'''
	Implement the preprocessing step phi, from
	the Nature paper. It takes the maximum pixel
	values of two consecutive frames. It then
	grayscales the image, and resizes it to 84x84.
	'''
	def preprocess(self):
		assert len(self.preprocess_stack) == 2
		return self.resize(self.grayscale(np.maximum(self.preprocess_stack[0], 
			self.preprocess_stack[1])))

	'''
	Takes in an RGB image and returns a grayscaled
	image.
	'''
	def grayscale(self, img):
		return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	'''
	Resizes the input to an 84x84 image.
	'''
	def resize(self, image):
		return cv2.resize(image, (84, 84))
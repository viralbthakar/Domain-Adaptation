import os
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from src.helper.utils import read_json, list_of_dict_conv, get_preprocess_fn

class Syn2Real_SourceOnly_Closed_Set_Classification(object):
	def __init__(self, data_json, preprocess_type, batch_size, input_shape, use_data_aug, split_index):
		self.data_json = data_json
		self.preprocess_type = preprocess_type
		self.batch_size = batch_size
		self.input_shape = input_shape
		self.use_data_aug = use_data_aug
		self.split_index = split_index

		self.all_data = read_json(data_json)
		self.source_data = self.all_data["SourceDomain"]
		self.target_data = self.all_data["TargetDomain"]
		
		random.shuffle(self.source_data)
		self.train_data = self.source_data[:int(len(self.source_data)*self.split_index[0])]
		self.val_data = self.source_data[int(len(self.source_data)*self.split_index[0]):]
		self.test_data = self.target_data

		self.total_train_images = len(self.train_data)
		self.total_val_images = len(self.val_data)
		self.total_test_images = len(self.test_data)

		self.class_list = sorted([f["Class_Name"] for f in self.all_data["Classes"]])
		self.num_class = len(self.class_list)

		self.preprocess_fn = get_preprocess_fn(self.preprocess_type)

		print("Creating Source Only data generators ... ")
		print("Found Total {} Classes : {}".format(len(self.class_list), self.class_list))
		print("Found Total {} Images in {} data".format(len(self.train_data), "train"))
		print("Found Total {} Images in {} data".format(len(self.val_data), "val"))
		print("Found Total {} Images in {} data".format(len(self.test_data), "test"))

		self.train_data = list_of_dict_conv(self.train_data)
		self.val_data = list_of_dict_conv(self.val_data)
		self.test_data = list_of_dict_conv(self.test_data)

	def augment(self, image, label):
		image = tf.image.random_flip_up_down(image, seed=None)
		image = tf.image.random_flip_left_right(image, seed=None)
		return image, label

	def preprocess(self, image, label):
		image = self.preprocess_fn(image)
		return image, label

	def get_img_file(self, img_path, img_name):
		img = tf.io.read_file(img_path + os.path.sep + img_name)
		img = tf.image.decode_png(img, channels=3)
		img = tf.image.resize(img, [self.input_shape[0], self.input_shape[1]], antialias=True)
		return img

	def parse_function(self, ip_dict):
		img = self.get_img_file(img_path=ip_dict["path"], img_name=ip_dict["image"])
		label = tf.one_hot(ip_dict["class_id"], self.num_class)
		return img, label
	
	def get_train_dataset(self):
		with tf.device('/cpu:0'):
			dataset = tf.data.Dataset.from_tensor_slices(self.train_data)
			dataset = dataset.shuffle(self.total_train_images)
			dataset = dataset.map(self.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			dataset = dataset.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			if self.use_data_aug:
				dataset = dataset.map(self.augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			dataset = dataset.batch(self.batch_size)
			dataset = dataset.prefetch(buffer_size=1)
		return dataset

	def get_val_dataset(self):
		with tf.device('/cpu:0'):
			dataset = tf.data.Dataset.from_tensor_slices(self.val_data)
			dataset = dataset.shuffle(self.total_val_images)
			dataset = dataset.map(self.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			dataset = dataset.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			if self.use_data_aug:
				dataset = dataset.map(self.augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			dataset = dataset.batch(self.batch_size)
			dataset = dataset.prefetch(buffer_size=1)
		return dataset
		
	def get_test_dataset(self):
		with tf.device('/cpu:0'):
			dataset = tf.data.Dataset.from_tensor_slices(self.test_data)
			dataset = dataset.shuffle(self.total_test_images)
			dataset = dataset.map(self.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			dataset = dataset.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			dataset = dataset.batch(self.batch_size)
			dataset = dataset.prefetch(buffer_size=1)
		return dataset

class Syn2Real_InDomain_Closed_Set_Classification(object):
	def __init__(self, data_json, domain, preprocess_type, batch_size, input_shape, use_data_aug, split_index=[0.7, 0.15, 0.15]):
		self.data_json = data_json
		self.domain = domain
		self.preprocess_type = preprocess_type
		self.batch_size = batch_size
		self.input_shape = input_shape
		self.use_data_aug = use_data_aug
		self.split_index = split_index

		self.all_data = read_json(data_json)
		if self.domain == "source":
			self.data = self.all_data["SourceDomain"]
		elif self.domain == "target":
			self.data = self.all_data["TargetDomain"]
		else:
			raise Exception('Enter Valid Domain : "source" or "target"')

		random.shuffle(self.data)
		self.train_data = self.data[:int(len(self.data)*self.split_index[0])]
		self.val_data = self.data[int(len(self.data)*self.split_index[0]):int(len(self.data)*self.split_index[0])+int(len(self.data)*self.split_index[1])]
		self.test_data = self.data[int(len(self.data)*self.split_index[0])+int(len(self.data)*self.split_index[1]):]
		self.total_train_images = len(self.train_data)
		self.total_val_images = len(self.val_data)
		self.total_test_images = len(self.test_data)

		self.class_list = sorted([f["Class_Name"] for f in self.all_data["Classes"]])
		self.num_class = len(self.class_list)

		self.preprocess_fn = get_preprocess_fn(self.preprocess_type)

		print("Creating {} data generators ... ".format(self.domain))
		print("Found Total {} Classes : {}".format(len(self.class_list), self.class_list))
		print("Found Total {} Images in {} data".format(len(self.data), self.domain))
		print("Found Total {} Images in {} data".format(len(self.train_data), "train"))
		print("Found Total {} Images in {} data".format(len(self.val_data), "val"))
		print("Found Total {} Images in {} data".format(len(self.test_data), "test"))

		self.train_data = list_of_dict_conv(self.train_data)
		self.val_data = list_of_dict_conv(self.val_data)
		self.test_data = list_of_dict_conv(self.test_data)

	def augment(self, image, label):
		image = tf.image.random_flip_up_down(image, seed=None)
		image = tf.image.random_flip_left_right(image, seed=None)
		return image, label
	
	def preprocess(self, image, label):
		image = self.preprocess_fn(image)
		return image, label

	def get_img_file(self, img_path, img_name):
		img = tf.io.read_file(img_path + os.path.sep + img_name)
		img = tf.image.decode_png(img, channels=3)
		img = tf.image.resize(img, [self.input_shape[0], self.input_shape[1]], antialias=True)
		return img

	def parse_function(self, ip_dict):
		img = self.get_img_file(img_path=ip_dict["path"], img_name=ip_dict["image"])
		label = tf.one_hot(ip_dict["class_id"], self.num_class)
		return img, label
	
	def get_train_dataset(self):
		with tf.device('/cpu:0'):
			dataset = tf.data.Dataset.from_tensor_slices(self.train_data)
			dataset = dataset.shuffle(self.total_train_images)
			dataset = dataset.map(self.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			dataset = dataset.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			if self.use_data_aug:
				dataset = dataset.map(self.augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			dataset = dataset.batch(self.batch_size)
			dataset = dataset.prefetch(buffer_size=1)
		return dataset

	def get_val_dataset(self):
		with tf.device('/cpu:0'):
			dataset = tf.data.Dataset.from_tensor_slices(self.val_data)
			dataset = dataset.shuffle(self.total_val_images)
			dataset = dataset.map(self.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			dataset = dataset.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			if self.use_data_aug:
				dataset = dataset.map(self.augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			dataset = dataset.batch(self.batch_size)
			dataset = dataset.prefetch(buffer_size=1)
		return dataset
	
	def get_test_dataset(self):
		with tf.device('/cpu:0'):
			dataset = tf.data.Dataset.from_tensor_slices(self.test_data)
			dataset = dataset.shuffle(self.total_test_images)
			dataset = dataset.map(self.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			dataset = dataset.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			dataset = dataset.batch(self.batch_size)
			dataset = dataset.prefetch(buffer_size=1)
		return dataset
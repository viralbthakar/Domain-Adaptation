import json
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def read_json(filename):
	with open(filename) as f:
		json_data = json.load(f)
	return json_data

def write_json(filepath, json_data):
	with open(filepath, 'w') as f:
		json.dump(json_data, f, indent=4, sort_keys=True)

def list_of_dict_conv(list_of_dict):
	data_dict = {"path":[], "image":[], "class_id":[]}
	for img in list_of_dict:
		data_dict["path"].append(img["path"])
		data_dict["image"].append(img["img_name"])
		data_dict["class_id"].append(img["class_id"])
	return data_dict

def show_batch(image_batch, label_batch, plt_title, rows=2, cols=2):
	num_images_to_show = rows * cols
	plt.figure(figsize=(8,8))
	for n in range(num_images_to_show):
		ax = plt.subplot(rows, cols, n+1)
		plt.imshow(image_batch[n], cmap='gray')
		plt.title(str(label_batch[n].numpy()))
		plt.axis('off')
	plt.suptitle(plt_title, fontsize=14)
	plt.show()

def check_directory_structure(root_dir, new_dir):
	print('creating directory ... ', os.path.join(root_dir, new_dir))
	os.makedirs(os.path.join(root_dir, new_dir), exist_ok=True)

def get_preprocess_fn(fn_id):
	if fn_id == 'vgg16':
		return tf.keras.applications.vgg16.preprocess_input
	elif fn_id == 'resnet50':
		return tf.keras.applications.resnet.preprocess_input
	elif fn_id == 'resnet152':
		return tf.keras.applications.resnet.preprocess_input
	else:
		def normlize(img):
			img = tf.cast(img, tf.float32)/255.0
			return img
		return normalize

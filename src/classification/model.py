import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dense, Activation, GlobalAveragePooling2D, GlobalMaxPool2D 
from tensorflow.keras.layers import Input, Flatten

class Classification_Models(object):
	def __init__(self, input_shape, num_class, last_layer_activation):
		self.input_shape = input_shape
		self.num_class = num_class 
		self.last_layer_activation = last_layer_activation  
		if self.num_class == 2:
			self.last_layer_nodes = 1
		else:
			self.last_layer_nodes = self.num_class

	def get_VGG16_FE(self, input_shape, pre_trained=None):
		vgg16 = tf.keras.applications.VGG16(include_top=False, weights=pre_trained, input_shape=input_shape)
		return vgg16

	def get_RESNET50_FE(self, input_shape, pre_trained=None):
		resnet50 = tf.keras.applications.ResNet50(include_top=False, weights=pre_trained, input_shape=input_shape)
		return resnet50

	def get_RESNET152_FE(self, input_shape, pre_trained=None):
		resnet152 = tf.keras.applications.ResNet152(include_top=False, weights=pre_trained, input_shape=input_shape)
		return resnet152

	def get_feature_extractor(self, model_identifier, pre_trained):
		if model_identifier == 'vgg16':
			return self.get_VGG16_FE(pre_trained=pre_trained, input_shape=self.input_shape)
		elif model_identifier == 'resnet50':
			return self.get_RESNET50_FE(pre_trained=pre_trained, input_shape=self.input_shape)
		elif model_identifier == 'resnet152':
			return self.get_RESNET152_FE(pre_trained=pre_trained, input_shape=self.input_shape)

	def build_model(self, model_identifier, pre_trained, feature_pooling, classification_nodes):
		feature_ext = self.get_feature_extractor(model_identifier, pre_trained)
		feature = feature_ext.output
		if feature_pooling == None:
			feature = Flatten(name='feature')(feature)
		elif feature_pooling == 'avg':
			feature = GlobalAveragePooling2D(name='feature')(feature)
		elif feature_pooling == 'max':
			feature = GlobalMaxPool2D(name='feature')(feature)
		x = Dense(classification_nodes[0], activation='relu', name='fc0')(feature)
		for i in range(1, len(classification_nodes)):
			x = Dense(classification_nodes[i], activation='relu', name='fc'+str(i))(x)
		logits = Dense(self.last_layer_nodes, name='logits')(x)
		probabilities = Activation(self.last_layer_activation)(logits)
		model_arch = Model(inputs=feature_ext.input, outputs=probabilities)
		return model_arch

	def get_model(self, model_identifier, pre_trained, classification_nodes, feature_pooling='avg'):
		model = self.build_model(model_identifier, pre_trained, feature_pooling, classification_nodes)
		return model
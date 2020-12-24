import progressbar
import tensorflow as tf
from src.helper.utils import show_batch, read_json, write_json, check_directory_structure 
from src.helper.loss import get_classification_loss, get_optimizer
from tensorflow.keras.metrics import Recall, Precision, Accuracy, Metric

class Classification_Trainer(object):
	def __init__(self, model, loss_fn, optimizer, performance_para, train_dataset, val_dataset, test_dataset, log_writer, weight_path):
		self.model = model
		self.loss_fn = loss_fn
		self.optimizer = optimizer
		self.performance_para = performance_para
		self.train_dataset = train_dataset
		self.val_dataset = val_dataset
		self.test_dataset = test_dataset
		self.log_writer = log_writer
		self.weight_path = weight_path
		self.history = {x : [] for x in self.performance_para.keys()}

	def train_step(self, inputs, targets):
		with tf.GradientTape() as tape:
			pred_labels = self.model(inputs, training=True)
			loss_value = self.loss_fn(targets, pred_labels)
		gradients = tape.gradient(loss_value, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
		return pred_labels, loss_value

	def val_step(self, inputs, targets):
		pred_labels = self.model(inputs, training=False)
		loss_value = self.loss_fn(targets, pred_labels)
		return pred_labels, loss_value
	
	def test_step(self, inputs, targets):
		pred_labels = self.model(inputs, training=False)
		loss_value = self.loss_fn(targets, pred_labels)
		return pred_labels, loss_value
		
	def train(self, epochs):
		for epoch in range(epochs):
			print("-"*10, "Epoch {}".format(epoch), "-"*10)
			train_bar = progressbar.ProgressBar(maxval=len(self.train_dataset), 
				widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()], term_width=50)
			val_bar = progressbar.ProgressBar(maxval=len(self.val_dataset), 
				widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()], term_width=50)
			test_bar = progressbar.ProgressBar(maxval=len(self.test_dataset), 
				widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()], term_width=50)
			
			print("-"*10, "Epoch {} Training".format(epoch), "-"*10)
			train_bar.start()
			for i, (image_batch, label_batch) in enumerate(self.train_dataset):
				train_bar.update(i+1)
				pred_labels, loss_value = self.train_step(image_batch, label_batch)
				self.update_performance_para("train", loss_value, pred_labels, label_batch)
			train_bar.finish()

			print("-"*10, "Epoch {} Validating".format(epoch), "-"*10)
			val_bar.start()
			for i, (image_batch, label_batch) in enumerate(self.val_dataset):
				val_bar.update(i+1)
				pred_labels, loss_value = self.val_step(image_batch, label_batch)
				self.update_performance_para("val", loss_value, pred_labels, label_batch)
			val_bar.finish()

			print("-"*10, "Epoch {} Testing".format(epoch), "-"*10)
			test_bar.start()
			for i, (image_batch, label_batch) in enumerate(self.test_dataset):
				test_bar.update(i+1)
				pred_labels, loss_value = self.test_step(image_batch, label_batch)
				self.update_performance_para("test", loss_value, pred_labels, label_batch)
			test_bar.finish()

			self.write_scalar_logs(epoch)
			self.update_history()
			self.print_epoch_summary(epoch)
			self.reset_performance_para()
			self.save_weights(epoch)

	def print_epoch_summary(self, epoch):
		print("-"*10, "Epoch {} Summary".format(epoch), "-"*10)
		for key, value in self.performance_para.items():
			print("Summary : {} is {}".format(key, value.result()))

	def update_performance_para(self, mode, loss_value, pred_labels, label_batch):
		self.performance_para[mode+"_loss"].update_state(loss_value)
		self.performance_para[mode+"_acc"].update_state(label_batch, pred_labels)

	def reset_performance_para(self):
		for key, value in self.performance_para.items():
			value.reset_states()

	def update_history(self):
		for key, value in self.performance_para.items():
			self.history[key].append(value.result().numpy())

	def write_scalar_logs(self, epoch):
		for key, value in self.performance_para.items():
			with self.log_writer[key.split('_')[0]].as_default():
				tf.summary.scalar(key, value.result(), step=epoch)

	def save_weights(self, epoch):
		print(self.history)
		val_loss_list = self.history["val_loss"][:-1]
		curr_loss = self.history["val_loss"][-1]
		if epoch == 0:
			print("-"*10, "Epoch {} - Saving Weights".format(epoch), "-"*10)
			self.model.save(self.weight_path)
		elif curr_loss < min(val_loss_list):
			print("-"*10, "Epoch {} - Saving Weights".format(epoch), "-"*10)
			self.model.save(self.weight_path)
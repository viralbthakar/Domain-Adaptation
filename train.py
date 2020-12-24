import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from src.helper.utils import read_json, write_json, check_directory_structure
from src.helper.loss import get_classification_loss, get_optimizer
from src.helper.performance_para import get_classification_performance_metrics
from src.classification.data_generator import Syn2Real_SourceOnly_Closed_Set_Classification, Syn2Real_InDomain_Closed_Set_Classification
from src.classification.model import Classification_Models
from src.classification.trainer import Classification_Trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', help='Config File', required=True)
args = parser.parse_args()

class Syn2Real_Classifier(object):
	def __init__(self, config):
		self.config = config
		self.init_config()

	def init_config(self):
		check_directory_structure(self.config["weight_dir"], self.config["exp_title"])
		check_directory_structure(self.config["log_dir"], self.config["exp_title"])
		print("-"*10, " Configuration ", "-"*10)
		print(json.dumps(self.config, indent=4))
		write_json(filepath=os.path.join(self.config["log_dir"], str(self.config["exp_title"]), str(self.config["exp_title"])+'_config.json'), json_data=self.config)
		self.init_logger()
		self.init_data_pipeline()
		self.init_model()

	def init_data_pipeline(self):
		print("-"*10, " Dataset Summary ", "-"*10)
		if self.config["exp_type"] == "InDomain":
			self.dataset = Syn2Real_InDomain_Closed_Set_Classification(data_json=os.path.join(self.config["DATA_DIR"], self.config["JSON_FILE"]),
						preprocess_type=self.config["model_identifier"],
						domain=self.config["domain"], 
						batch_size=self.config["batch_size"], 
						input_shape=self.config["input_shape"],
						use_data_aug=self.config["use_data_aug"], 
						split_index=self.config["split_index"])
		elif self.config["exp_type"] == "SourceOnly":
			self.dataset = Syn2Real_SourceOnly_Closed_Set_Classification(data_json=os.path.join(self.config["DATA_DIR"], self.config["JSON_FILE"]), 
						preprocess_type=self.config["model_identifier"],
						batch_size=self.config["batch_size"], 
						input_shape=self.config["input_shape"],
						use_data_aug=self.config["use_data_aug"],
						split_index=self.config["split_index"])
		self.train_dataset = self.dataset.get_train_dataset()
		self.validation_dataset = self.dataset.get_val_dataset()
		self.test_dataset = self.dataset.get_test_dataset()
		
	def init_model(self):
		print("-"*10, " Model Architecture ", "-"*10)
		self.model_arch = Classification_Models(input_shape=self.config["input_shape"], num_class=self.config["num_class"], last_layer_activation=self.config["last_layer_activation"])
		self.base_model = self.model_arch.get_model(model_identifier=self.config["model_identifier"],
										pre_trained=self.config["pre_trained"],
										feature_pooling=self.config["feature_pooling"], 
										classification_nodes=self.config["classification_nodes"]) 
		print(self.base_model.summary())
		self.loss_fn = self.init_loss_fn()
		self.optimizer = self.init_optimizer()
		self.performance_para = self.init_performance_para()

	def init_loss_fn(self):
		print("-"*10, " Loss Function {}".format(self.config["loss"]), "-"*10)
		loss_fn = get_classification_loss(self.config["loss"])
		return loss_fn

	def init_optimizer(self):
		print("-"*10, " Optimizer {} with Learning Rate {}".format(self.config["optimizer"], self.config["learning_rate"]), "-"*10)
		optimizer = get_optimizer(optimizer_id=config["optimizer"], learning_rate=config["learning_rate"])
		return optimizer

	def init_performance_para(self):
		print("-"*10, " Performance Parameters ", "-"*10)
		paras = get_classification_performance_metrics()
		return paras

	def init_logger(self):
		print("-"*10, " Log Writers ", "-"*10)
		self.log_writer = {}
		self.log_writer["train"] = tf.summary.create_file_writer(os.path.join(self.config["log_dir"], self.config["exp_title"], "train"))
		self.log_writer["val"] = tf.summary.create_file_writer(os.path.join(self.config["log_dir"], self.config["exp_title"], "val"))
		self.log_writer["test"] = tf.summary.create_file_writer(os.path.join(self.config["log_dir"], self.config["exp_title"], "test"))

	def init_trainer(self):
		trainer = Classification_Trainer(model=self.base_model, 
			loss_fn=self.loss_fn, 
			optimizer=self.optimizer,
			performance_para=self.performance_para, 
			train_dataset=self.train_dataset, 
			val_dataset=self.validation_dataset,
			test_dataset=self.test_dataset, 
			log_writer=self.log_writer, 
			weight_path=os.path.join(self.config["weight_dir"], self.config["exp_title"]))
		trainer.train(self.config["epoch"])

config = read_json(args.config_file)
Syn2Real_Classifier(config).init_trainer()
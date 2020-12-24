import os
import json
import tensorflow as tf
from src.classification.model import Classification_Models
from src.helper.utils import read_json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', help='Config File', required=True)
args = parser.parse_args()

config = read_json(args.config_file)
print("-"*10, " Configuration ", "-"*10)
print(json.dumps(config, indent=4))

print("-"*10, " Model Architecture ", "-"*10)
model_arch = Classification_Models(input_shape=config["input_shape"], 
	num_class=config["num_class"], 
	last_layer_activation=config["last_layer_activation"])
base_model = model_arch.get_model(model_identifier=config["model_identifier"],
								pre_trained=config["pre_trained"],
								feature_pooling=config["feature_pooling"], 
								classification_nodes=config["classification_nodes"]) 
print(base_model.summary())

if True:
	tf.keras.utils.plot_model(
	base_model, to_file=os.path.join(config["model_arch_dir"], config["model_identifier"]+'.png'), 
	show_shapes=True, show_layer_names=True, 
	rankdir='TB', expand_nested=False, dpi=96
	)
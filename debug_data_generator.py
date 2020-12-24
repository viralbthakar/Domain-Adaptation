import os
import json
import numpy as np
from src.helper.utils import show_batch, read_json
from src.classification.data_generator import Syn2Real_SourceOnly_Closed_Set_Classification, Syn2Real_InDomain_Closed_Set_Classification
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', help='Config File', required=True)
args = parser.parse_args()

config = read_json(args.config_file)
print("-"*10, " Configuration ", "-"*10)
print(json.dumps(config, indent=4))

print("-"*10, " Dataset Summary ", "-"*10)
if config["exp_type"] == "InDomain":
	dataset = Syn2Real_InDomain_Closed_Set_Classification(data_json=os.path.join(config["DATA_DIR"], config["JSON_FILE"]),
				preprocess_type=config["model_identifier"],
				domain=config["domain"], 
				batch_size=config["batch_size"], 
				input_shape=config["input_shape"],
				use_data_aug=config["use_data_aug"], 
				split_index=config["split_index"])
elif config["exp_type"] == "SourceOnly":
	dataset = Syn2Real_SourceOnly_Closed_Set_Classification(data_json=os.path.join(config["DATA_DIR"], config["JSON_FILE"]), 
				preprocess_type=config["model_identifier"],
				batch_size=config["batch_size"], 
				input_shape=config["input_shape"],
				use_data_aug=config["use_data_aug"],
				split_index=config["split_index"])
train_dataset = dataset.get_train_dataset()
val_dataset = dataset.get_val_dataset()
test_dataset = dataset.get_test_dataset()

print("-"*10, " Dataset Sample ", "-"*10)
image_batch, label_batch = next(iter(train_dataset))	
print("Label: ", label_batch.numpy())
print("Image shape: ", image_batch.numpy().shape)
print("Max Value : {}".format(np.amax(image_batch.numpy())))
print("Min Value : {}".format(np.min(image_batch.numpy())))
show_batch(image_batch, label_batch, plt_title=dataset.class_list, rows=4, cols=4)

image_batch, label_batch = next(iter(val_dataset))	
print("Label: ", label_batch.numpy())
print("Image shape: ", image_batch.numpy().shape)
print("Max Value : {}".format(np.amax(image_batch.numpy())))
print("Min Value : {}".format(np.min(image_batch.numpy())))
show_batch(image_batch, label_batch, plt_title=dataset.class_list, rows=4, cols=4)

image_batch, label_batch = next(iter(test_dataset))	
print("Label: ", label_batch.numpy())
print("Image shape: ", image_batch.numpy().shape)
print("Max Value : {}".format(np.amax(image_batch.numpy())))
print("Min Value : {}".format(np.min(image_batch.numpy())))
show_batch(image_batch, label_batch, plt_title=dataset.class_list, rows=4, cols=4)
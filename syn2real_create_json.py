import os
import json
from PIL import Image

DATA_DIR = './data/Syn2Real_Closed_Set_Classification'
SRC_DIR = 'SourceDomain'
TRG_DIR = 'TargetDomain'
OUTPUT_JSON_FILE = 'Syn2Real_Closed_Set_Classification.json'
DATA_DICT = {SRC_DIR:[], TRG_DIR:[], "Classes":[]}
src_img_id = 0
target_img_id = 0

source_dir = os.path.join(DATA_DIR, SRC_DIR)
target_dir = os.path.join(DATA_DIR, TRG_DIR)

class_labels = sorted([f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))])

for i, clss in enumerate(class_labels):
	clss_dict = {}
	clss_dict["Class_ID"] = i
	clss_dict["Class_Name"] = clss
	DATA_DICT["Classes"].append(clss_dict)

for clss in class_labels:
	print("Processing {} class from {}".format(clss, source_dir))
	source_domain_files = [f for f in os.listdir(os.path.join(source_dir, clss)) if os.path.splitext(f)[-1] == '.png']
	print("Found total {} images in {} class".format(len(source_domain_files), clss))

	for img_file in source_domain_files:
		img_dict = {}
		im = Image.open(os.path.join(source_dir, clss, img_file))
		width, height = im.size

		img_dict["id"] = src_img_id
		img_dict["height"] = height
		img_dict["width"] = width
		img_dict["img_name"] = img_file
		img_dict["path"] = os.path.join(source_dir, clss)
		img_dict["class_id"] = class_labels.index(clss)
		src_img_id += 1

		DATA_DICT[SRC_DIR].append(img_dict)

for clss in class_labels:
	print("Processing {} class from {}".format(clss, target_dir))
	[os.rename(os.path.join(target_dir, clss,f), os.path.join(target_dir, clss, os.path.splitext(f)[0] + '.png')) for f in os.listdir(os.path.join(target_dir, clss)) if os.path.splitext(f)[-1] == '.jpg']
	target_domain_files = [f for f in os.listdir(os.path.join(target_dir, clss)) if os.path.splitext(f)[-1] == '.png']
	print("Found total {} images in {} class".format(len(target_domain_files), clss))

	for img_file in target_domain_files:
		img_dict = {}

		im = Image.open(os.path.join(target_dir, clss, img_file))
		width, height = im.size

		img_dict["id"] = target_img_id
		img_dict["height"] = height
		img_dict["width"] = width
		img_dict["img_name"] = img_file
		img_dict["path"] = os.path.join(target_dir, clss)
		img_dict["class_id"] = class_labels.index(clss)
		target_img_id += 1

		DATA_DICT[TRG_DIR].append(img_dict)

with open(os.path.join(DATA_DIR, OUTPUT_JSON_FILE), 'w') as json_file:
	json.dump(DATA_DICT, json_file, indent=4)
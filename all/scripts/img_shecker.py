import copy
import json
import os
import sys
import time
from os import listdir
from os.path import join, isfile

import albumentations as A
import numpy as np
import torch
from PIL import Image
from ax.service.ax_client import AxClient
from torch import nn
from torchvision import models
from torchvision import transforms


data_dir = "/ml/dl_cource/practice/flow/project/val/meme/"

file_list = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and ('jpg' in f.lower() or 'jpeg' in f.lower())]


wrong_files = []

for fl in file_list:
	try:
		image = Image.open(join(data_dir,fl)).convert('RGB')
	except:
		wrong_files.append(fl)
		os.remove(join(data_dir,fl))


print(wrong_files)
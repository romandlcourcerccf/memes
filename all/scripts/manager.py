import argparse
import json
import os
import sys
from os.path import isfile, join
from pathlib import Path
from shutil import copyfile

from os import listdir, path
from datetime import datetime
from onnx_coreml import convert

from torch import nn
from torchvision import models

import coremltools.proto.FeatureTypes_pb2 as ft
import coremltools

import torch
import numpy as np

CONFIG_FILE_NAME = 'config.json'
GEN_FOLDER_NAME = 'gen'

VAL_FOLDER_NAME = 'val'
TRAIN_FOLDER_NAME = 'train'
TEST_FOLDER_NAME = 'test'

MODELS_FOLDER_NAME = 'models'
SAVED_FOLDER_NAME = 'saved'
CONVERTED_MODELS_FOLDER_NAME = 'converted'

try:
   ROOT_DIR = os.environ["ROOT_DIR"]
except KeyError:
   print ("Please set the environment variable ROOT_DIR")
   sys.exit(1)

sys.path.append(ROOT_DIR)


CONFIG_FILE_NAME = 'config.json'
METRICS_FILE_NAME = 'metrics.json'

def read_config():
    with open(join(ROOT_DIR, 'scripts', CONFIG_FILE_NAME), 'r') as json_file:
        return json.load(json_file)


def zero(settings):

    classes = settings['classes'].split(',')
    print(classes)

    make_folder(classes, TRAIN_FOLDER_NAME)
    make_folder(classes, GEN_FOLDER_NAME)
    make_folder(classes, VAL_FOLDER_NAME)
    make_folder(classes, TEST_FOLDER_NAME)

    make_folder(SAVED_FOLDER_NAME)
    make_folder(CONVERTED_MODELS_FOLDER_NAME)

    Path(join(path_to_project, MODELS_FOLDER_NAME)).mkdir(parents=True, exist_ok=True)


def make_folder(*args):
    if len(args) == 1:
        folder_name = args[0]
        Path(join(path_to_project, folder_name)).mkdir(parents=True, exist_ok=True)
    elif len(args) == 2:
        folder_name = args[1]
        classes = args[0]

        Path(join(path_to_project, folder_name)).mkdir(parents=True, exist_ok=True)
        for cl in classes:
            Path(join(path_to_project, folder_name, cl)).mkdir(parents=True, exist_ok=True)


def merge(settings):

    path_to_project = settings['project_path']
    split_rate = [float(a) for a in settings['split_rate'].split(',')]

    if int(np.array(split_rate).sum().round()) != 1:
        print("Wrong split rate.")
        sys.exit(1)

    classes = settings['classes'].split(',')

    for cl in classes:

        gen_files = [f for f in listdir(join(path_to_project, GEN_FOLDER_NAME, cl)) if isfile(path.join(path_to_project, GEN_FOLDER_NAME,cl,f)) and (('jpeg' in f.lower()) or  ('jpg' in f.lower()))]
        train_files = [f for f in listdir(join(path_to_project, TRAIN_FOLDER_NAME, cl)) if isfile(path.join(path_to_project, TRAIN_FOLDER_NAME,cl, f)) and (('jpeg' in f.lower()) or  ('jpg' in f.lower()))]
        val_files = [f for f in listdir(join(path_to_project, VAL_FOLDER_NAME, cl)) if isfile(path.join(path_to_project, VAL_FOLDER_NAME,cl, f)) and (('jpeg' in f.lower()) or  ('jpg' in f.lower()))]
        test_files = [f for f in listdir(join(path_to_project, TEST_FOLDER_NAME, cl)) if isfile(path.join(path_to_project, TEST_FOLDER_NAME,cl, f)) and (('jpeg' in f.lower()) or  ('jpg' in f.lower()))]

        previous_gen = set(train_files).union(set(val_files)).union(set(test_files))

        avaliable_files = list(set(gen_files).difference(set(previous_gen)))

        left = int(len(avaliable_files) * split_rate[0])
        middle =  int(len(avaliable_files) * split_rate[1])
        right =   int(len(avaliable_files) * split_rate[2])

        train_files = avaliable_files[0: left]
        val_files = avaliable_files[left: left + middle]
        test_files = avaliable_files[left + middle:]

        for f in train_files:
            copyfile(join(path_to_project, GEN_FOLDER_NAME, cl, f), join(path_to_project, TRAIN_FOLDER_NAME, cl, f))

        for f in val_files:
            copyfile(join(path_to_project, GEN_FOLDER_NAME, cl, f), join(path_to_project, VAL_FOLDER_NAME, cl, f))

        for f in test_files:
            copyfile(join(path_to_project, GEN_FOLDER_NAME, cl, f), join(path_to_project, TEST_FOLDER_NAME, cl, f))


def save(settings):

    path_to_project = settings['project_path']

    with open(join(path_to_project, METRICS_FILE_NAME), 'r') as json_file:
        metrics = json.load(json_file)

    best_parameters = metrics['best_parameters']

    print(best_parameters)

    models = listdir(join(path_to_project, 'models'))

    for fl in models:

        params = {}
        split_params = fl.split('%')

        for sp in split_params:
            if len(sp.split(':')) == 2:
                params[sp.split(':')[0]] = sp.split(':')[1]

        is_equal = True
        for k, val in enumerate(params):
            print(float(params[val]), '  ', float(best_parameters[val]))
            if float(params[val]) != float(best_parameters[val]):
                is_equal = False

        if is_equal:

            dt_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            Path(join(path_to_project, SAVED_FOLDER_NAME, dt_str)).mkdir(parents=True, exist_ok=True)
            copyfile(join(path_to_project, 'models', fl), join(path_to_project, SAVED_FOLDER_NAME, dt_str, fl))


def build_model():
    model = models.mobilenet_v2(pretrained=True)
    print(model.classifier)
    model.classifier[1] = nn.Sequential(nn.Linear(1280, 2), nn.Softmax())
    return model


def convert_to_ml(settings):
    path_to_project = settings['project_path']
    image_size = int(settings['image_size'])

    dummy_input = torch.randn(1, 3, image_size, image_size)

    paths = sorted(Path(join(path_to_project, SAVED_FOLDER_NAME)).iterdir(), key=os.path.getmtime)
    last = paths[-1]
    madel_path =list(Path(last).iterdir())[0]

    model = build_model()
    model.load_state_dict(torch.load(madel_path))

    path_to_converted = join(path_to_project, CONVERTED_MODELS_FOLDER_NAME, datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))

    Path(path_to_converted).mkdir(parents=True, exist_ok=True)

    onnx_model_path = join(path_to_converted, "Model.onnx")
    ml_model_path = join(path_to_converted, "Model.mlmodel")

    torch.onnx.export(model,
                      dummy_input,
                      onnx_model_path,
                      input_names=["image"],
                      output_names=["output"])

    model_cml = convert(model=onnx_model_path, minimum_ios_deployment_target='13')
    model_cml.save(ml_model_path)

    spec = coremltools.utils.load_spec(ml_model_path)

    input = spec.description.input[0]
    input.type.imageType.colorSpace = ft.ImageFeatureType.RGB
    input.type.imageType.height = image_size
    input.type.imageType.width = image_size

    coremltools.utils.save_spec(spec, ml_model_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-1', '--command', nargs='*', help='This will be option One')

    res = parser.parse_args()

    settings = read_config()
    path_to_project = settings['project_path']
    print('path_to_project : ', path_to_project)

    for opt in res.command:
        if opt == 'zero':
            zero(settings)
        if opt == 'merge':
            merge(settings)
        if opt == 'save':
            save(settings)
        if opt == 'convert':
            convert_to_ml(settings)

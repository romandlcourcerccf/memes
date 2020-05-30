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

CONFIG_FILE_NAME = 'config.json'


def read_config():
    with open(join(ROOT_DIR, 'scripts', CONFIG_FILE_NAME), 'r') as json_file:
        return json.load(json_file)


def set_parameter_requires_grad(model, level):
    cnt = 0
    for name, param in model.named_parameters():
        if cnt < level:
            param.requires_grad = False
        cnt += 1

CONFIG_FILE_NAME = 'config.json'
GEN_FOLDER_NAME = 'gen'
VAL_FOLDER_NAME = 'val'
MODELS_FOLDER_NAME = 'models'
TRAIN_FOLDER_NAME = 'train'

try:
    ROOT_DIR = os.environ["ROOT_DIR"]
except KeyError:
    print("Please set the environment variable ROOT_DIR")
    sys.exit(1)


image_size = 224

root_path = ROOT_DIR

data_transforms = {
        'train': A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomBrightnessContrast(),
            A.ShiftScaleRotate(rotate_limit=15, scale_limit=0.10),
            A.Normalize(),

        ], p=1),
        'val': A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize()

        ], p=1)
    }


class MemesDataSet(torch.utils.data.Dataset):

    def __init__(self, data_dir, is_train, transform=transforms.Compose([transforms.ToTensor()])):

        self.data_dir = data_dir
        self.files = []
        self.labels = []
        self.transform = transform

        self.data_dir = join(data_dir, is_train)
        self.classes = listdir(self.data_dir)
        self.classes_number = len(self.classes)

        self.mapping = {}

        for idx, cl in enumerate(self.classes):
            self.mapping[cl] = idx

        for cl in self.classes:

            file_list = [f for f in listdir(join(self.data_dir, cl)) if isfile(join(self.data_dir, cl, f)) and ('jpg' in f.lower() or 'jpeg' in f.lower())]
            for fl in file_list:
                self.files.append(fl)
                self.labels.append(cl)

        self.data_size = len(self.files)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):

        image = Image.open(join(self.data_dir, self.labels[idx], self.files[idx])).convert('RGB')
        image = np.array(image)
        image = self.transform(image=image)
        image = image['image']
        image = image.transpose(2, 1, 0)
        label_num = self.mapping[self.labels[idx].lower()]
        # TODO Do this a fancy way
        label = np.zeros(self.classes_number, dtype=np.float32)
        label[label_num] = 1

        return image, label


def build_model(level_to_freeze):
    model = models.mobilenet_v2(pretrained=True)
    set_parameter_requires_grad(model, level_to_freeze)
    print(model.classifier)
    model.classifier[1] = nn.Sequential(nn.Linear(1280, 2), nn.Softmax())
    return model

def train_model(paramerers, project_path):
    batch_size = 8
    learning_rate = float(paramerers['lr'])
    num_epochs = int(paramerers['num_epochs'])
    level_to_freeze = int(paramerers['level_to_freeze'])

    model = build_model(level_to_freeze)
    criterion = nn.BCELoss(reduce=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    image_datasets = {x: MemesDataSet(data_dir=project_path, is_train=x, transform=data_transforms[x]) for x in ['train', 'val']}
    dataloaders = { x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1) for x in ['train', 'val']}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    model = model.to(device)

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    outputs_idxs = torch.argmax(outputs, axis=1)
                    targets_idxs = torch.argmax(labels, axis=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(outputs_idxs == targets_idxs)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model_file_name = ""
    for key in paramerers.keys():
        model_file_name += key +":"+str(paramerers[key])+"%"

    model_file_name += '.pth'

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), join(project_path, "models", model_file_name))

    return float(best_acc)


def run_train(settings):

    hyper_parameters = settings['hyper_parameters']

    project_path = settings['project_path']

    print(hyper_parameters)

    ax = AxClient(enforce_sequential_optimization=False)
    ax.create_experiment(name="mobile_net_experiment", objective_name="train_model", parameters=hyper_parameters , minimize=False)

    for _ in range(3):
        next_parameters, trial_index = ax.get_next_trial()
        ax.complete_trial(trial_index=trial_index, raw_data=train_model(next_parameters, project_path = project_path))

    best_parameters, metrics = ax.get_best_parameters()

    data = {'best_parameters': best_parameters, 'metrics': metrics}
    with open(join(project_path, 'metrics.json'), 'w') as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    settings = read_config()
    run_train(settings)

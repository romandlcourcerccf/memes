import json
import os
import sys
from os.path import join

from ax.service.ax_client import AxClient
from keras.applications import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import *
from keras.layers import Dense
from keras.models import Model
from keras.models import load_model
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator

CONFIG_FILE_NAME = 'config.json'

def read_config():
    with open(join(ROOT_DIR, 'scripts', CONFIG_FILE_NAME), 'r') as json_file:
        return json.load(json_file)


CONFIG_FILE_NAME = 'config.json'
GEN_FOLDER_NAME = 'gen'
VAL_FOLDER_NAME = 'val'
TEST_FOLDER_NAME = 'test'
MODELS_FOLDER_NAME = 'models'
TRAIN_FOLDER_NAME = 'train'

try:
    ROOT_DIR = os.environ["ROOT_DIR"]
except KeyError:
    print("Please set the environment variable ROOT_DIR")
    sys.exit(1)


image_size = 224
batch_size = 32

root_path = ROOT_DIR


def train_model(paramerers, project_path, trial_index):

    learning_rate_top = float(paramerers['lr_top'])
    num_epochs_top = int(paramerers['num_epochs_top'])
    patience_top = int(paramerers['patience_top'])
    num_epochs_final = int(paramerers['num_epochs_final'])
    patience_final = int(paramerers['patience_final'])
    learning_rate_final = float(paramerers['lr_final'])
    level_to_freeze = int(paramerers['level_to_freeze'])

    base_model = MobileNet(weights='imagenet',  include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(base_model.input, predictions)

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=5,
                                       shear_range=0.5,
                                       zoom_range=0.5,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       cval=0.5,
                                       horizontal_flip=True,
                                       vertical_flip=True
                                       )

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(join(project_path, TRAIN_FOLDER_NAME),
                                                        target_size=(image_size, image_size),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(join(project_path, VAL_FOLDER_NAME),
                                                                  target_size=(image_size, image_size),
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(join(project_path, TEST_FOLDER_NAME),
                                                            target_size=(image_size, image_size),
                                                            batch_size=batch_size,
                                                            class_mode='categorical')

#+++++++++++++++++++++++top phase+++++++++++++++++++++++++++++++++++++++

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=learning_rate_top), loss='categorical_crossentropy', metrics=['accuracy'])
    top_weights_path = join(join(project_path, MODELS_FOLDER_NAME), "memes_" + "top" + "_"+str(trial_index) + '.h5')

    callbacks_list = [
        EarlyStopping(monitor='val_acc', patience=patience_top, verbose=0),
        ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001,
                          cooldown=0, min_lr=0)
    ]

    model.fit_generator(train_generator,
                        epochs=num_epochs_top,
                        validation_data=validation_generator,
                        validation_steps=len(validation_generator) // batch_size,
                        callbacks=callbacks_list,
                        steps_per_epoch=len(train_generator) // batch_size,
                        workers=1,
                        use_multiprocessing=False
                        )

    # +++++++++++++++++++++++final phase+++++++++++++++++++++++++++++++++++++++

    model.load_weights(top_weights_path)

    for layer in model.layers[:level_to_freeze]:
        layer.trainable = False
    for layer in model.layers[level_to_freeze:]:
        layer.trainable = True

    model.compile(optimizer=Adam(lr=learning_rate_final),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    final_weights_path = join(join(project_path, MODELS_FOLDER_NAME), "memes_" + "final" + "_"+str(trial_index) + '.h5')

    callbacks_list = [
        EarlyStopping(monitor='val_acc', patience=patience_final, verbose=0),
        ModelCheckpoint(final_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001,
                          cooldown=0, min_lr=0)
    ]

    model.fit_generator(train_generator,
                        epochs=num_epochs_final,
                        validation_data=validation_generator,
                        validation_steps=len(validation_generator) // batch_size,
                        callbacks=callbacks_list,
                        steps_per_epoch=len(train_generator) // batch_size,
                        workers=1,
                        use_multiprocessing=False
                        )

    model = load_model(final_weights_path)
    test_acc = model.evaluate_generator(test_generator, steps=50)

    return test_acc

def run_train(settings):

    hyper_parameters = settings['hyper_parameters']
    project_path = settings['project_path']
    rounds = int(settings['rounds'])

    print(hyper_parameters)

    ax = AxClient(enforce_sequential_optimization=False)
    ax.create_experiment(name="mobile_net_experiment", objective_name="train_model", parameters=hyper_parameters , minimize=False)

    for _ in range(rounds):
        next_parameters, trial_index = ax.get_next_trial()
        ax.complete_trial(trial_index=trial_index, raw_data=train_model(next_parameters, project_path = project_path, trial_index = trial_index))

    best_parameters, metrics = ax.get_best_parameters()

    data = {'best_parameters': best_parameters, 'metrics': metrics}
    with open(join(project_path, 'metrics.json'), 'w') as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    settings = read_config()
    run_train(settings)

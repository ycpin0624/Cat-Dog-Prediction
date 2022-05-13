from tensorflow.keras.applications import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Flatten,
    Dropout,
    Input,
    Conv2D,
    MaxPooling2D,
)
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from tensorflow.keras import Model
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop, SGD
from keras.utils import np_utils
from keras.models import load_model
from keras.models import Sequential
import keras.optimizers
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL
import numpy as np
import pathlib
import os
import zipfile

# !wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip -O /content/drive/MyDrive/tmp/cats_and_dogs_filtered.zip
local_zip = "/content/drive/MyDrive/tmp/cats_and_dogs_filtered.zip"
zip_ref = zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("/content/drive/MyDrive/tmp/cats_dogs")
zip_ref.close()

img_height = 224
img_width = 224
image_size = (224, 224)
batch_size = 4

data_dir = pathlib.Path(
    "/content/drive/MyDrive/tmp/cats_dogs/cats_and_dogs_filtered/train/"
)
test_data_dir = pathlib.Path(
    "/content/drive/MyDrive/tmp/cats_dogs/cats_and_dogs_filtered/validation/"
)

# 訓練資料
train_datagen = ImageDataGenerator(
    featurewise_center=True,
    rotation_range=20,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.1,
)

train_ds = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",  # class name 轉為 0/1
    shuffle=True,
    subset="training",
)

val_ds = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",  # class name 轉為 0/1
    shuffle=True,
    subset="validation",
)

print(train_ds.class_indices)

# 驗證資料
datagen = ImageDataGenerator()
test_ds = datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
)

# 檢查資料
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


def define_model():
    conv_base = tf.keras.applications.EfficientNetB7(
        weights="imagenet",
        include_top=False,
        input_shape=(img_height, img_width, 3),
        pooling="avg",
    )
    conv_base.trainable = False
    # conv_base.summary()
    # flat1 = Flatten()(model.layers[-1].output)
    # class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    # output = Dense(2, activation='sigmoid')(class1)
    # model = Model(inputs=model.inputs, outputs=output)
    # opt = SGD(learning_rate=0.001, momentum=0.9)
    # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

    model = Sequential()
    model.add(conv_base)
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.0005 / 10),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )
    return model


model = define_model()
model.summary()

history = model.fit_generator(
    train_ds,
    epochs=20,
    steps_per_epoch=len(train_ds) // batch_size,
    validation_data=val_ds,
    validation_steps=len(val_ds) // batch_size,
)

model.save("/content/drive/MyDrive/final_model.h5")

results = model.evaluate(test_ds, batch_size=batch_size, use_multiprocessing=True)
print(results)

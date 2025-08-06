from os import listdir
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

raw_folder = '/content/drive/MyDrive/datachomeo/'

def save_data(raw_folder=raw_folder):
    dest_size = (128, 128)
    print("Bắt đầu xử lý ảnh.....")

    pixels = []
    labels = []

    for folder in listdir(raw_folder):
        if folder != '.DS_Store':
            print('Folder = ', folder)
            for file in listdir(raw_folder + folder):
                if file != '.DS_Store':
                    print('File = ', file)
                    pixels.append(cv2.resize(cv2.imread(raw_folder + folder + '/' + file), dsize=(128, 128)))
                    labels.append(folder)

    pixels = np.array(pixels)
    labels = np.array(labels)

    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)
    print(labels)

    file = open('pix.data', 'wb')
    pickle.dump((pixels, labels), file)
    file.close()

    return


def load_data():
    file = open('pix.data', 'rb')

    (pixels, labels) = pickle.load(file)

    file.close()

    print(pixels.shape)
    print(labels.shape)

    return pixels, labels


save_data()
X, Y = load_data()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

print(X_train.shape)
print(Y_train.shape)

def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    input = Input(shape=(128, 128, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)

    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model


vggmodel = get_model()

filepath = "/content/drive/MyDrive/data/weights-{epoch:02d}-{val_accuracy:.2f}.keras"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.1,
                         rescale=1. / 255,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         brightness_range=[0.2, 1.5], fill_mode="nearest")

aug_val = ImageDataGenerator(rescale=1. / 255)

vgghist=vggmodel.fit(aug.flow(X_train, Y_train, batch_size=64),
                               epochs=50,
                               validation_data=aug.flow(X_test,Y_test,
                               batch_size=len(X_test)),
                               callbacks=callbacks_list)

vggmodel.save('/content/drive/MyDrive/data/vggmodel.keras')

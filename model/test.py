import os
from tqdm import tqdm
import random
from PIL import Image
import numpy as np
import cv2
import matplotlib.image as mpimg  # mpimg 用于读取图片

shape = (128,128)
def data_resize(path,forg,org):
    forg_files = os.listdir(path + forg)
    org_files = os.listdir(path + org)
    forg_files.sort()
    org_files.sort()
    for x in tqdm(forg_files):
        if x.startswith('T'): continue
        img = cv2.imread(path + forg + x, 0)
        imgs = 255 - img
        cv2.imwrite(path + forg + 'gray_' + x, imgs)
    for x in tqdm(org_files):

        if x.startswith('T'): continue
        img = cv2.imread(path + org + x, 0)
        imgs = 255 - img
        cv2.imwrite(path + org + 'gray_' + x, imgs)


def data_inverse(path,forg,org):
    forg_files = os.listdir(path + forg)
    org_files = os.listdir(path + org)
    forg_files.sort()
    org_files.sort()

    for x in tqdm(forg_files):
        if x.startswith('T'): continue
        img = Image.open(path + forg + x)
        img = img.resize(shape)
        img = img.convert('RGB')
        img.save(path + forg + x)
    for x in tqdm(org_files):
        if x.startswith('T'): continue
        img = Image.open(path + org + x)
        img = img.resize(shape)
        img = img.convert('RGB')
        img.save(path + forg + x)


def train_dataloader(path, batch_size):
    samples = [[], [], [], []]
    labels = []
    n = 45
    for index in (range(1, n)):  # 使用多少组图片，
        for first in range(1, 25):  # 每组anc的数量
            for second in range(1, 25):  # 每组pos的数量
                for x in range(2):
                    if x == 0:
                        temp1 = mpimg.imread(path + 'full_org/' + 'original_' + str(index) + '_' + str(first) + '.png')
                        temp2 = mpimg.imread(path + 'full_org/' + 'original_' + str(index) + '_' + str(second) + '.png')
                        temp3 = mpimg.imread(
                            path + 'full_org/' + 'gray_original_' + str(index) + '_' + str(first) + '.png')
                        temp4 = mpimg.imread(
                            path + 'full_org/' + 'gray_original_' + str(index) + "_" + str(second) + '.png')
                        samples[0].append(temp1)
                        samples[1].append(temp2)
                        samples[2].append(temp3)
                        samples[3].append(temp4)
                        labels.append([0, 1])
                        if len(labels) == batch_size:
                            yield (
                            [np.array(samples[0]), np.array(samples[1]), np.array(samples[2]), np.array(samples[3])],
                            [np.array(labels), np.array(labels), np.array(labels)])
                            yield (
                            [np.array(samples[1]), np.array(samples[0]), np.array(samples[3]), np.array(samples[2])],
                            [np.array(labels), np.array(labels), np.array(labels)])

                            samples = [[], [], [], []]
                            labels = []
                    elif x == 1:
                        if second % 2 == 0:
                            temp2 = mpimg.imread(
                                path + 'full_forg/' + 'forgeries_' + str(index) + '_' + str(second) + '.png')
                            temp4 = mpimg.imread(
                                path + 'full_forg/' + 'gray_forgeries_' + str(index) + "_" + str(second) + '.png')
                        else:
                            index_rand = 0
                            second_rand = 0
                            while True:
                                index_rand = random.randint(1, 45)
                                second_rand = random.randint(1, 24)
                                if index_rand != index:
                                    break
                            temp2 = mpimg.imread(
                                path + 'full_org/' + 'original_' + str(index_rand) + '_' + str(second_rand) + '.png')
                            temp4 = mpimg.imread(path + 'full_org/' + 'gray_original_' + str(index_rand) + "_" + str(
                                second_rand) + '.png')

                        samples[0].append(temp1)
                        samples[1].append(temp2)
                        samples[2].append(temp3)
                        samples[3].append(temp4)
                        labels.append([1, 0])
                        if len(labels) == batch_size:
                            yield (
                            [np.array(samples[0]), np.array(samples[1]), np.array(samples[2]), np.array(samples[3])],
                            [np.array(labels), np.array(labels), np.array(labels)])
                            yield (
                            [np.array(samples[1]), np.array(samples[0]), np.array(samples[3]), np.array(samples[2])],
                            [np.array(labels), np.array(labels), np.array(labels)])

                            samples = [[], [], [], []]
                            labels = []


def test_dataloader(path):
    shape = (128, 128)
    samples = [[], [], [], []]
    labels = []
    n = 55
    for index in tqdm(range(45, n)):  # 使用多少组图片，
        for first in range(1, 25):  # 每组anc的数量
            for second in range(1, 25):  # 每组pos的数量
                for x in range(2):
                    temp1 = mpimg.imread(path + 'full_org/' + 'original_' + str(index) + '_' + str(first) + '.png')
                    temp3 = mpimg.imread(path + 'full_org/' + 'gray_original_' + str(index) + '_' + str(first) + '.png')
                    if x == 0:
                        temp2 = mpimg.imread(path + 'full_org/' + 'original_' + str(index) + '_' + str(second) + '.png')
                        temp4 = mpimg.imread(
                            path + 'full_org/' + 'gray_original_' + str(index) + "_" + str(second) + '.png')
                        samples[0].append(temp1)
                        samples[1].append(temp2)
                        samples[2].append(temp3)
                        samples[3].append(temp4)
                        labels.append([0, 1])


                    elif x == 1:
                        if second % 2 == 0:
                            temp2 = mpimg.imread(
                                path + 'full_forg/' + 'forgeries_' + str(index) + '_' + str(second) + '.png')
                            temp4 = mpimg.imread(
                                path + 'full_forg/' + 'gray_forgeries_' + str(index) + "_" + str(second) + '.png')
                            samples[0].append(temp1)
                            samples[1].append(temp2)
                            samples[2].append(temp3)
                            samples[3].append(temp4)
                            labels.append([1, 0])
                        else:
                            for i in range(5):
                                index_rand = 0
                                second_rand = 0
                                while True:
                                    index_rand = random.randint(1, 55)
                                    second_rand = random.randint(1, 24)
                                    if index_rand != index:
                                        break
                                temp2 = mpimg.imread(path + 'full_org/' + 'original_' + str(index_rand) + '_' + str(
                                    second_rand) + '.png')
                                temp4 = mpimg.imread(
                                    path + 'full_org/' + 'gray_original_' + str(index_rand) + "_" + str(
                                        second_rand) + '.png')

                                samples[0].append(temp1)
                                samples[1].append(temp2)
                                samples[2].append(temp3)
                                samples[3].append(temp4)
                                labels.append([1, 0])
    return ([np.array(samples[0]), np.array(samples[1]), np.array(samples[2]), np.array(samples[3])],
            [np.array(labels), np.array(labels), np.array(labels)])

path = './signatures/'

Image.open(path +'full_org/' + 'original_' + str(10) + '_' + str(11) + '.png')

test_x, test_y = test_dataloader(path)

import tensorflow as tf
# import efficientnet.keras as efn
from tensorflow.keras import regularizers
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.resnet_v2 import ResNet50V2
# from keras.applications.resnext import ResNeXt50
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet121
# from keras.applications.resnext import ResNeXt50
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, AveragePooling2D, Concatenate, Input, \
    BatchNormalization, GlobalAveragePooling2D
from triplet_loss_functions import triplet_loss_func
from tensorflow.keras.optimizers import Adam
import h5py
from sklearn.metrics import accuracy_score

# from tensorflow.keras.applications.imagenet_utils import preprocess_input
import os
import tensorflow.keras.backend as KTF
from tensorflow.keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)


class TripMODEL:
    def __init__(self):
        self.shape = (128, 128, 3)

    def baseMode(self):
        #         initial_model = Xception(weights='imagenet', include_top=False,input_shape=self.shape)
        initial_model = VGG16(weights='imagenet', include_top=False, input_shape=self.shape)  ##1.75
        #         initial_model = ResNet50(weights='imagenet',include_top=False,input_shape=self.shape)##0.75
        #         initial_model = InceptionV3(weights='imagenet', include_top=False,input_shape=self.shape,pooling='max')
        #         initial_model = DenseNet121(weights='imagenet', include_top=False,input_shape=self.shape)
        #         initial_model = MobileNet(weights='imagenet',include_top=False,input_shape=self.shape)
        #         initial_model = efn.EfficientNetB4(include_top=False,input_shape=self.shape)
        for layer in initial_model.layers[:]:
            layer.trainable = False

        last = initial_model.output
        x = Flatten()(last)
        x = Dense(256, activation="relu")(x)

        model = Model(initial_model.input, [x])
        return model

    def fine_tuned_model(self):
        first_input = Input(shape=self.shape, name='first_input')
        first_input_gray = Input(shape=self.shape, name='first_input_gray')
        second_input = Input(shape=self.shape, name='second_input')
        second_input_gray = Input(shape=self.shape, name='second_input_gray')

        shared_model = self.baseMode()  # 共享模型

        first_out = shared_model(first_input)
        second_out = shared_model(second_input)
        first_out_gray = shared_model(first_input_gray)
        second_out_gray = shared_model(second_input_gray)

        output1 = Dense(128, activation='relu')(Concatenate()([first_out, second_out]))

        output2 = Dense(128, activation='relu')(Concatenate()([first_out_gray, second_out_gray]))

        output3 = Dense(256, activation='relu')(Concatenate()([first_out, second_out_gray]))

        output4 = Dense(128, activation='relu')(Concatenate()([first_out_gray, second_out]))

        output5 = Dense(128, activation='relu')(Concatenate()([first_out, second_out, first_out_gray, second_out_gray]))

        output1 = Dense(2, activation="softmax", name="loss_1")(output1)
        output2 = Dense(2, activation="softmax", name="loss_2")(output2)
        output3 = Dense(2, activation="softmax", name="loss_3")(output3)
        output4 = Dense(2, activation="softmax", name="loss_4")(output4)
        output5 = Dense(2, activation="softmax", name="loss_5")(output5)
        model = Model(inputs=[first_input, second_input, first_input_gray, second_input_gray],
                      outputs=[output1, output2, output5])

        adam = Adam(lr=1e-3)

        model.compile(loss={"loss_1": "categorical_crossentropy", "loss_2": "categorical_crossentropy",
                            #                             "loss_3":"categorical_crossentropy","loss_4":"categorical_crossentropy",
                            "loss_5": "categorical_crossentropy"
                            }, loss_weights=[1 / 3, 1 / 3, 1 / 3],
                      optimizer=adam, metrics=['acc'])

        return model


import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import pandas as pd
import random


def get_five_result(preds, trues):
    result_prob = []
    result_sum = []
    print(trues.shape)
    for index in range(trues.shape[0]):
        pos_sum = 0
        score = 0.0
        for pos in range(3):
            score += preds[pos][index][1]
            if preds[pos][index][1] > 0.5:
                pos_sum += 1
        if pos_sum >= 2:
            label_sum = [0, 1]
        else:
            label_sum = [1, 0]
        if score >= 1.5:
            label_prob = [0, 1]
        else:
            label_prob = [1, 0]

        result_sum.append(label_sum)
        result_prob.append(label_prob)
    print(len(result_sum))
    print("acc_sum: ", accuracy_score(result_sum, trues))
    print("acc_prob: ", accuracy_score(result_prob, trues))


batch_size = 64
epochs = 10

train_number = 50000
test_number = 50000
train_gen = train_dataloader(path, batch_size)

tm = TripMODEL()
model = tm.fine_tuned_model()
model.summary()
for step in range(epochs):
    print('Step:{}'.format(step))
    train_gen = train_dataloader(path, batch_size)
    model.fit_generator(train_gen,
                        epochs=1,
                        steps_per_epoch=44 * 625 * 4 // 128,
                        verbose=1
                        )
    #     model.save_weights('save_model/model_{}_weights.h5'.format(step))
    model.save('save_model/model_{}.h5'.format(step))
    #     with open('save_model/model_{}.json'.format(step), 'w') as outfile:
    #         outfile.write(model.to_json())
    predictions = model.predict(test_x, batch_size=256,
                                verbose=1)
    #     auc = roc_auc_score(y_test,predictions)
    acc = get_five_result(predictions, test_y[0])
    print('currnt acc:{} '.format(acc))
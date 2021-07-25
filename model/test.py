
# coding: utf-8

import os
from tqdm import tqdm
import random
from PIL import Image
import cv2
import matplotlib.image as mpimg # mpimg 用于读取图片
def dataloader(path):
    shape = (128,128)
    forg_files = os.listdir(path +'full_forg')
    org_files = os.listdir(path +'full_org')
    forg_files.sort()
    org_files.sort()
   
    # for x in tqdm(forg_files):
    #     if x.startswith('T'):continue
    #     img = cv2.imread(path + '/full_forg/' + x, 0)
    #     imgs = 255 - img
    #     cv2.imwrite(path + '/full_forg/' + 'gray_' +x,imgs)
    # for x in tqdm(org_files):
    #
    #     if x.startswith('T'):continue
    #     img = cv2.imread(path + '/full_org/' + x, 0)
    #     imgs = 255 - img
    #     cv2.imwrite(path + '/full_org/' + 'gray_' +x,imgs)
    # forg_files = os.listdir(path +'full_forg')
    # org_files = os.listdir(path +'full_org')
    # forg_files.sort()
    # org_files.sort()
    # for x in tqdm(forg_files):
    #     if x.startswith('T'):continue
    #     img = Image.open(path + '/full_forg/' + x)
    #     img = img.resize(shape)
    #     img = img.convert('RGB')
    #     img.save(path + '/full_forg/' + x)
    # for x in tqdm(org_files):
    #     if x.startswith('T'):continue
    #     img = Image.open(path + '/full_org/' + x)
    #     img = img.resize(shape)
    #     img = img.convert('RGB')
    #     img.save(path + '/full_org/' + x)
    samples = [[],[],[],[]]
    labels = []
    
    for index in tqdm(range(1,25)):#使用多少组图片，
        
        for first in range(1,25):#每组anc的数量
            
            for second in range(1,25):#每组pos的数量
                temp1 = mpimg.imread(path +'full_org/' + 'original_' + str(index) + '_' + str(first) + '.png') 
                temp2 = mpimg.imread(path +'full_org/' + 'original_' + str(index) + '_' + str(second)+ '.png') 
                temp3 = mpimg.imread(path + 'full_org/' + 'gray_original_' + str(index) + '_' + str(first) + '.png')
                temp4 = mpimg.imread(path + 'full_org/' + 'gray_original_' + str(index) + "_" + str(second) + '.png')
                samples[0].append(temp1)
                samples[1].append(temp2)
                samples[2].append(temp3)
                samples[3].append(temp4)
                labels.append([0,1])
                temp2 = mpimg.imread(path +'full_forg/' + 'forgeries_' + str(index) + '_' + str(second)+ '.png') 
                temp4 = mpimg.imread(path + 'full_forg/' + 'gray_forgeries_' + str(index) + "_" + str(second) + '.png')
                samples[0].append(temp1)
                samples[1].append(temp2)
                samples[2].append(temp3)
                samples[3].append(temp4)
                labels.append([1,0])
            
    return samples,labels  

path = './signatures/'

Image.open(path +'full_org/' + 'original_' + str(10) + '_' + str(11) + '.png')
samples,labels = dataloader(path)


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

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten,AveragePooling2D, Concatenate,Input,BatchNormalization,GlobalAveragePooling2D

from tensorflow.keras.optimizers import Adam



import os
import tensorflow.keras.backend as KTF
from tensorflow.keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   
sess = tf.Session(config=config)
KTF.set_session(sess)
def metrics(y_true,pred):
            
            pred = pred[:,0:384]
#             pred = np.sqrt(np.square(pred)/np.sum(np.square(pred)))
            anchor, positive, negative = pred[:, 0:128], pred[:, 128:256], pred[:, 256:384]
            pos_neg = np.sum(np.square(positive - negative),axis=-1,keepdims=True)
            pos_dist = np.sum(np.square(anchor - positive), axis=-1, keepdims=True)
            neg_dist = np.sum(np.square(anchor - negative), axis=-1, keepdims=True)
            basic_loss = pos_dist - neg_dist 
            r_count = 0
            print(basic_loss.shape) 
            r_count = basic_loss[np.where(basic_loss < 0)].shape[0]
            
            d_count = pos_neg[np.where(pos_neg > 0.5)].shape[0]
            return positive,negative, (float(r_count) / pred.shape[0]),float(d_count) / pred.shape[0]
class TripMODEL:
    def __init__(self):
        self.shape = (128,128,3)
        
    def contrastive_loss(self,y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
            http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        sqaure_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean((1-y_true) * sqaure_pred + (y_true) * margin_square)

    def four_loss(self,y_true,y_pred):
        a = y_pred[:,:128]
        b = y_pred[:,128:256]
        c = y_pred[:,256:384]
        d = y_pred[:,384:]
        
        loss_a = K.mean(K.square(1-a))
        loss_b = K.mean(K.square(b))
        loss_c = K.mean(K.square(1-c))
        loss_d = K.mean(K.square(d))
        return loss_a + loss_b + loss_c + loss_d
    def triplet_loss(self,y_true, y_pred):
        """
        Triplet Loss的损失函数
        """
         #softmax loss
#         pred = y_pred[:,384:441]
        
#         anc, pos, neg = pred[:, 0:19], pred[:, 19:38], pred[:, 38:57]
#         anc_loss = K.categorical_crossentropy(y_true, anc)
#         pos_loss = K.categorical_crossentropy(y_true, pos)
# #         neg_loss = K.categorical_crossentropy(y_true, neg)
#         softmax_loss = K.mean((anc_loss + pos_loss )/2.0)
        # triplet loss
        margin = 1.5
        pred = y_pred[:,0:384]

        a = pred[:, 0:128]
        b = pred[:, 128:256]
        c = pred[:, 256:384]
                
#         a = K.l2_normalize(pred[:, 0:128],axis=1)
#         b = K.l2_normalize(pred[:, 128:256],axis=1)
#         c = K.l2_normalize(pred[:, 256:384],axis=1)
        
        anc, pos, neg = a,b,c
        pos_dist =  K.sqrt(K.sum(K.square(anc - pos), axis=-1, keepdims=True))
        neg_dist = K.sqrt(K.sum(K.square(anc - neg), axis=-1, keepdims=True))
        basic_loss = K.max(pos_dist - neg_dist + margin,0)

        trip_loss = K.mean(basic_loss)
       
        gray_pred = y_pred[:,384:]
        d = pred[:,384:384+128]
        e = pred[:,384 + 128:384 + 256]
        f = pred[:,384+256:]
        gray_pos_dist = K.sqrt(K.sum(K.square(d-e),axis=-1,keepdims=True))
        gray_neg_dist = K.sqrt(K.sum(K.square(d-f),axis=-1,keepdims=True))
        gray_basic_loss = K.max(gray_pos_dist - gray_neg_dist + margin,0)
        gray_trip_loss = K.mean(gray_basic_loss)
        
        
       
        ##final loss
        loss =0.5 * trip_loss  + 0.5 * gray_trip_loss
        return loss
    def baseMode(self):
#         initial_model = Xception(weights='imagenet', include_top=False,input_shape=self.shape)
        initial_model = VGG16( weights='imagenet',include_top=False,input_shape=self.shape)##1.75
#         initial_model = ResNet50(weights='imagenet',include_top=False,input_shape=self.shape)##0.75
#         initial_model = InceptionV3(weights='imagenet', include_top=False,input_shape=self.shape,pooling='max')
#         initial_model = DenseNet121(weights='imagenet', include_top=False,input_shape=self.shape)
#         initial_model = MobileNet(weights='imagenet',include_top=False,input_shape=self.shape)
#         initial_model = efn.EfficientNetB4(include_top=False,input_shape=self.shape)
#         for layer in initial_model.layers[:]:
#             layer.trainable = False
        
        last = initial_model.output

        
        x = Flatten()(last)
        x = Dense(128, activation='relu')(x)
        
        
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
        
        
        output1 = Dense(128, activation='relu')(Concatenate()([first_out,second_out]))
        output2 = Dense(128, activation='relu')(Concatenate()([first_out_gray,second_out_gray]))
        output3 = Dense(128, activation='relu')(Concatenate()([first_out,second_out_gray]))
        output4 = Dense(128, activation='relu')(Concatenate()([second_out,first_out_gray]))
        output5 = Dense(128, activation='relu')(Concatenate()([first_out,second_out,first_out_gray,second_out_gray]))
        output1 = Dense(2,activation="softmax",name="loss_1")(output1)
        output2 = Dense(2,activation="softmax",name="loss_2")(output2)
        output3 = Dense(2,activation="softmax",name="loss_3")(output3)
        output4 = Dense(2,activation="softmax",name="loss_4")(output4)
        output5 = Dense(2,activation="softmax",name="loss_5")(output5)
        model = Model(inputs=[first_input,second_input,first_input_gray,second_input_gray],outputs=[output1,output2,output3,output4,output5])
       
        adam = Adam(lr=3e-4)

        model.compile(loss={"loss_1":"categorical_crossentropy","loss_2":"categorical_crossentropy","loss_3":"categorical_crossentropy"
                           ,"loss_4":"categorical_crossentropy","loss_5":"categorical_crossentropy"}, optimizer=adam,metrics=['acc'])
        
        return model





import numpy as np

from sklearn.metrics import accuracy_score


def get_three_result(preds,trues):
    result = []
    for index in range(len(preds[0])):
        if preds[0][index][0] + preds[1][index][0] + preds[2][index][0] + preds[3][index][0]+ preds[4][index][0]< 2.5:
            label = [0,1]
        else:
            label = [1,0]
        result.append(label)
    print("acc: ",accuracy_score(result,trues))


batch_size = 1
epochs = 10

# anc,pos,neg = dataloader('/home/huangzhen/project/handwritten/signature-recognition/handwritten-data/signatures/')
train_number =  10
test_number = 20000
train_data= ([samples[0][:train_number],samples[1][:train_number],samples[2][:train_number],samples[3][:train_number]])
test_data = ([samples[0][test_number:],samples[1][test_number:],samples[2][test_number:],samples[3][test_number:]])
y_train = [np.array(labels[:train_number]),np.array(labels[:train_number]),np.array(labels[:train_number])
           ,np.array(labels[:train_number]),np.array(labels[:train_number])]
y_test  =  [np.array(labels[test_number:]),np.array(labels[test_number:]),np.array(labels[test_number:])
            ,np.array(labels[test_number:]),np.array(labels[test_number:])]
tm = TripMODEL()
model = tm.fine_tuned_model()
model.summary()
for step in range(epochs):
    print('Step:{}'.format(step))
    model.fit(train_data, y_train,
              batch_size=batch_size,
              epochs=1,
              shuffle=True,
              verbose=1
              )

    # predictions = model.predict(test_data, batch_size=128, verbose=1)

    model.save_weights('save_model/model_{}_weights.h5'.format(step))
    model.save('save_model/model_{}.h5'.format(step))
    with open('save_model/model_{}.json'.format(step), 'w') as outfile:
        outfile.write(model.to_json())

#     auc = roc_auc_score(y_test,predictions)
#     acc = get_three_result(predictions,y_test[0])
#     print('currnt acc:{} '.format(acc))
#

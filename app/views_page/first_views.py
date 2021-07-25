from flask import request
import cv2
from PIL import Image
import matplotlib.image as mpimg # mpimg 用于读取图片
from tensorflow.keras.models import load_model
from app import app

@app.route("/first/request_score",methods=['GET', 'POST'])
def request_score():
    if request.method == 'POST':
        origin_img = request.files.get("origin_name")
        valid_img = request.files.get("valid_name")
        print(origin_img)
        print(valid_img)
        origin_img.save("test_origin.png")
        valid_img.save("test_valid.png")
        first_origin,first_inverse = img_inverse_resize("test_origin",".png")
        second_origin,second_inverse = img_inverse_resize("test_valid",".png")
        inputs = [[first_origin], [first_inverse], [second_origin], [second_inverse]]

        score_prob, score_pos, score_sum = predict(inputs)
        print("score_prob: ", score_prob)
        scores = ""
        for x in score_prob:
            scores += str(x) + "\t"
        scores += str(score_pos) + "\t" + str(score_sum)
        print("resturn scores : ",scores)
        return scores
def img_inverse_resize(path,suffix):
    shape = (128,128)
    img_origin = cv2.imread(path+suffix, 0)
    img_inverse = 255 - img_origin
    cv2.imwrite(path +"_gray" + suffix, img_inverse)

    img_origin = Image.open(path  + suffix)
    img_origin = img_origin.resize(shape)
    img_origin = img_origin.convert('RGB')
    img_origin.save(path+suffix)
    img_inverse = Image.open(path + "_gray"  + suffix)
    img_inverse = img_inverse.resize(shape)
    img_inverse = img_inverse.convert('RGB')
    img_inverse.save(path + "_gray"  + suffix)

    img_origin = mpimg.imread(path + suffix)
    img_inverse = mpimg.imread(path + "_gray" + suffix)
    print(img_origin[0][0])
    print(img_inverse[0][0])
    return img_origin,img_inverse

import tensorflow as tf
from tensorflow.python.keras.backend import set_session
sess = tf.Session()
graphs = tf.get_default_graph()
set_session(sess)
model = load_model('../model/save_model/model_4.h5')

def predict(inputs):
    global sess
    global graphs
    with graphs.as_default():
        set_session(sess)
        res = model.predict(inputs)
        print(res)
        return predict_result(res)
def predict_result(scores):
    score_prob = []
    score_pos = 0
    score_sum = 0.0
    for score in scores:
        value = score[0][1]
        if value > 0.5:
            score_pos += 1
        score_prob.append(value)
        score_sum += value
    return score_prob,score_pos,score_sum


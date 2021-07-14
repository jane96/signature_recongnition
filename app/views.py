from flask import render_template
from app import app
from flask import request
import cv2
from PIL import Image
import matplotlib.image as mpimg # mpimg 用于读取图片

from werkzeug.utils import secure_filename

@app.route('/')
@app.route('/first')
def first():
    user = { 'nickname': 'Miguel' } # fake user

    return render_template("first.html",
        title = 'Home',
        user = user)
@app.route('/second')
def second():
    user = { 'nickname': 'Miguel' } # fake user
    return render_template("second.html",
        title = 'Home',
        user = user)
@app.route('/thrid')
def thrid():
    user = { 'nickname': 'Miguel' } # fake user
    return render_template("thrid.html",
        title = 'Home',
        user = user)
@app.route('/four')
def four():
    user = { 'nickname': 'Miguel' } # fake user
    return render_template("four.html",
        title = 'Home',
        user = user)
@app.route('/five')
def five():
    user = { 'nickname': 'Miguel' } # fake user
    return render_template("five.html",
        title = 'Home',
        user = user)

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
    print(img_origin.shape)
    print(img_inverse.shape)


@app.route("/first/request_score",methods=['GET', 'POST'])
def request_score():
    if request.method == 'POST':
        origin_img = request.files.get("origin_name")
        valid_img = request.files.get("valid_name")
        origin_img.save("test_origin.png")
        valid_img.save("test_valid.png")
        img_inverse_resize("test_origin",".png")
        img_inverse_resize("test_valid",".png")



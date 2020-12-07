#利用flask打包模型
import io
import json

from torchvision import models
import torchvision.transforms as tf
from PIL import Image
from flask import Flask,jsonify,request
from detecter import Detector

app = Flask(__name__)


@app.route("/kuang",methods=['POST'])
def kuang():
    if request.method =='POST':
        file = request.files['file']
        img_bytes =file.read()
        img = io.BytesIO(img_bytes)
        image = Image.open(img)
        detect=Detector()
        boxes = detect.detect(image)
        for box in boxes:
            x1 = str(box[0])
            y1 = str(box[1])
            x2 = str(box[2])
            y2 = str(box[3])
            return jsonify({"x1":x1,"y1":y1,"x2":x2,"y2":y2})

if __name__ == '__main__':
    app.run()
from detecter import Detector
import cv2
import torch
from torchvision import transforms
from ar_de_relizeface import facenet
import os
import time
from PIL import Image

detector = Detector()
data_path = r"D:\data_image\face_r_data"
facenet = facenet.Net()
facenet.load_state_dict(torch.load(r"D:\deeplearing\face_decter\ar_de_relizeface\parms\arc_99.pt"))
id = input("登记者：")
cap = cv2.VideoCapture(0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #获取视频的宽度
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(w,h)
path = os.path.join(data_path, id)
while True:
    x = time.time()
    ret, photo = cap.read()
    if ret:
        b,g,r = cv2.split(photo)
        img = cv2.merge([r,g,b])
    else:
        break
    im = Image.fromarray(img,"RGB")

    boexs = detector.detect(im)#,zoom_factor=0.7,p_conf = 0.5,p_nms=0.3,r_conf = 0.7,r_nms=0.3,o_conf = 0.999,o_nms=0.3

    data = []
    for i ,box in enumerate(boexs):
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        w = x2-x1
        h = y2-y1
        a = max(w,h)

        b = a-w
        c = a-h

        ng = im.crop((x1+1, y1+1, x2+b-1, y2+c-1))
        ng_t = ng.resize((100, 100))
        tf = transforms.ToTensor()
        ng_data = tf(ng_t)
        ng_feat,_ = facenet(ng_data)
        data.append(ng_feat)

        cv2.rectangle(photo,(x1,y1),(x2,y2),(0,0,255),3)


    cv2.imshow("capture",photo)
    if cv2.waitKey(10)=="q":
        break
    print("写入中...")
    txt = open('{}.txt'.format(path), 'w')
    txt.write(str(data))
    txt.close()
    print("写入成功")
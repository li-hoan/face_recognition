import os
from detecter import *
from PIL import Image


image_path = r"D:\data_image\1234"
for i in os.listdir(image_path):
    detector = Detector()
    with Image.open(os.path.join(image_path,i)) as im: # 打开图片
        # boxes = detector.detect(im)
        print("----------------------------")
        boxes = detector.detect(im)
        print("size:",im.size)
        imDraw = ImageDraw.Draw(im)
        for box in boxes: # 多个框，没循环一次框一个人脸
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            print((x1, y1, x2, y2))

            print("conf:",box[4]) # 置信度
            imDraw.rectangle((x1, y1, x2, y2), outline='red',width=2)
            #im.show() # 每循环一次框一个人脸
        im.show()
        # exit()
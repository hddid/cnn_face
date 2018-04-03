# -*- coding=utf-8 -*-
import cv2
import os

out_dir = './my_faces'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)



# 获取分类器
haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 打开摄像头 参数为输入流，可以为摄像头或视频文件
camera = cv2.VideoCapture(0)

n = 1
while True:
    if (n <= 10000):
        print('正在生成第 %s 照片.' % n)
        # 读帧

        success, img = camera.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#将图片转换为灰度图
        faces = haar.detectMultiScale(gray_img, 1.3, 5)#在灰度图中检测出人脸
        for x, y, w, h in faces:
            face = img[y:y+h, x:x+w]#从灰度图中抠出脸部设置大小
            face = cv2.resize(face, (64,64))
            face_gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            cv2.imshow('img', face_gray)
            cv2.imwrite(out_dir+'/'+str(n)+'.jpg', face_gray)
            n += 1
        key = cv2.waitKey(30) & 0xff == ord('q')
        if key == 27:
            break
    else:
        break
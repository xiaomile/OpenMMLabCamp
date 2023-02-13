import os
import random
import shutil

if not os.path.exists('H:/aiCodeCamp/clothes/data'):
    os.mkdir('H:/aiCodeCamp/clothes/data')
# print(os.getcwd())

img_path = 'H:/aiCodeCamp/clothes/jpeg_images/IMAGES/'
ann_path = 'H:/aiCodeCamp/clothes/png_masks/MASKS/'
train_img_path = 'H:/aiCodeCamp/clothes/data/img/train/'
train_ann_path = 'H:/aiCodeCamp/clothes/data/ann/train/'
val_img_path = 'H:/aiCodeCamp/clothes/data/img/val/'
val_ann_path = 'H:/aiCodeCamp/clothes/data/ann/val/'

if not os.path.exists(train_img_path):
    os.makedirs(train_img_path)
if not os.path.exists(train_ann_path):
    os.makedirs(train_ann_path)
if not os.path.exists(val_img_path):
    os.makedirs(val_img_path)
if not os.path.exists(val_ann_path):
    os.makedirs(val_ann_path)



for i in range(1, 1001):
    fileNo = ('0000'+str(i))[-4:]
    if(random.randint(1, 10)>8):
        shutil.copy(img_path + "img_" + fileNo + '.jpeg', val_img_path + "img_" + fileNo + '.jpg')
        shutil.copy(ann_path + "seg_" + fileNo + '.png', val_ann_path + "img_" + fileNo + '.png')
    else:
        shutil.copy(img_path + "img_" + fileNo + '.jpeg', train_img_path + "img_" + fileNo + '.jpg')
        shutil.copy(ann_path + "seg_" + fileNo + '.png', train_ann_path + "img_" + fileNo + '.png')

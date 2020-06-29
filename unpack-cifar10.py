import cv2
import numpy as np
import random
import os
ratio = 0.8
path = os.path.join(os.getcwd(),"cifar-10-batches-py")
d = r"D:\task\HANJIA\NN\TransferLearningClassification\Test01-csdn\data"
train = os.path.join(d,"train")
test = os.path.join(d,"test")
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

for i in range(1,6):
    file = f'data_batch_{i}'
    print(file)
    file = os.path.join(path,file)
    dict1 = unpickle(file)
    for i in range(len(dict1[b"data"])):
        img = dict1[b"data"][i]#得到图片的数据
        img = np.reshape(img, (3, 32,32))  #转为三维图片数组
        img = img.transpose((1,2,0))#通道转换为CV2的要求形式
        img_name = dict1[b"filenames"][i]#拿到图片的名字

        img_label = dict1[b"labels"][i]#拿到图片的标签

        # print(img_name,img_label,type(img_name),type(img_label))
        img_name = bytes.decode(img_name)
        img_label = str(img_label)
        folder = img_label
        f = random.random()
        if f < ratio:
            folder =os.path.join(train,folder)
        else:
            folder = os.path.join(test,folder)
        os.makedirs(folder,exist_ok=True)
        name = os.path.join(folder,img_name)
        cv2.imwrite(name,img)#保存

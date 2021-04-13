import os
import pickle
import torch
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

class LoadImages(object):
    def __init__(self):
        self.data = []

    def get_image_by_list(self, image_list, label_list):
        if len(image_list) != len(label_list):
            print("datas not match the labels")
            return
        for i in range(len(image_list)):
            self.data.append({label_list[i]: Image.open(image_list[i])})

    def get_image_by_dict(self, image_dict):
        for i in image_dict:
            self.data.append({i: Image.open(image_dict[i])})

    def transform(self):
        out = []
        for i in self.data:
            for key in i:
                out.append({key: transforms.ToTensor()(i[key])})
        return out

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    label = []
    data = []

    cat_path = glob.glob(current_path + "/data_set/cat/*.*")  # 匹配所有的符合条件的文件，并将其以list的形式返回
    for i in cat_path:
        label.append("cat")
        test = Image.open(i)
        data.append(transforms.ToTensor()(Image.open(i)))

    others_path = glob.glob(current_path + "/data_set/others/*.*")  # 匹配所有的符合条件的文件，并将其以list的形式返回
    for repeat in range(10):
        for i in others_path:
            label.append("others")
            data.append(transforms.ToTensor()(Image.open(i).convert("RGB")))

    with open("data_set/data_set.pkl", "wb") as data_set:
        pickle.dump(label, data_set)
        pickle.dump(data, data_set)



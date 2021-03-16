import os
from PIL import Image
import glob
from torchvision import transforms

class LoadImages(object,):
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
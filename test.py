import os
from PIL import Image
import glob
from torchvision import transforms


current_path = os.path.dirname(os.path.abspath(__file__))

img_path_list = glob.glob(current_path + "/datas/*.jpg")  # 匹配所有的符合条件的文件，并将其以list的形式返回
img_path = current_path + "/datas/cat1.jpg"  # 匹配所有的符合条件的文件，并将其以list的形式返回


class LoadImages(object,):
    def __init__(self):
        self.data = []

    def get_image_by_list(self, image_list, label_list):
        if len(image_list) != len(label_list):
            print("datas not match the labels")
            return
        for i in range(len(image_list)):
            self.data.append({label_list[i]: Image.open(image_list[i])})
            print(self.data)

    def get_image_by_dict(self, image_dict):
        for i in image_dict:
            self.data.append({i: Image.open(image_dict[i])})
            print(self.data)

    def transform(self):
        out = []
        for i in self.data:
            for key in i:
                out.append({key:transforms.ToTensor()(i[key])})
        print(out)
        return out


test = LoadImages()
label_list = []
for i in img_path_list:
    label_list.append("cat")
test.get_image_by_list(img_path_list, label_list)
out = test.transform()
pass

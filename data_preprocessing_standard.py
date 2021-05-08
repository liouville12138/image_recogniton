import os
import glob
import torch
import joblib
import pickle
import numpy as np
import torchvision.transforms as T
from sklearn.preprocessing import LabelEncoder
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ImageWarp:
    def __init__(self):
        pass

    @staticmethod
    def resize(image_in, obj_size: tuple):
        transform = T.Resize(obj_size, interpolation=2)
        result = transform(image_in)
        return result

    @staticmethod
    def center_crop(image_in, crop_size):
        transform = T.CenterCrop(crop_size)
        result = transform(image_in)
        return result

    @staticmethod
    def five_crop(image_in, crop_size):
        transform = T.FiveCrop(crop_size)
        results = transform(image_in)
        return results

    @staticmethod
    # 彩色图转灰度图
    def gray_scale(image_in, output_channels: int):
        transform = T.Grayscale(num_output_channels=output_channels)
        result = transform(image_in)
        return result

    @staticmethod
    def random_affine(image_in, degrees, fillcolor):
        transform = T.RandomAffine(degrees=degrees,
                                   translate=None,
                                   scale=None,
                                   shear=0.3,
                                   resample=False,
                                   fillcolor=fillcolor)
        result = transform(image_in)
        return result

    @staticmethod
    def random_rotation(image_in, degrees, expand: bool):
        transform = T.RandomRotation(degrees=degrees,
                                     resample=False,
                                     expand=expand,
                                     center=None,
                                     fill=None)
        result = transform(image_in)
        return result

    @staticmethod
    def horizontal_flip(image_in, probability: float):
        transform = T.RandomHorizontalFlip(p=probability)
        result = transform(image_in)
        return result

    @staticmethod
    def vertical_flip(image_in, probability: float):
        transform = T.RandomVerticalFlip(p=probability)
        result = transform(image_in)
        return result


def get_image():
    current_path = os.path.dirname(os.path.abspath(__file__))
    label_out = []
    data_out = []
    cat_path = glob.glob(current_path + "/data_set/cat/*.*")  # 匹配所有的符合条件的文件，并将其以list的形式返回
    for i in cat_path:
        data_expand = data_augmentation(Image.open(i).convert("RGB"), (192, 192), 60)
        for j in data_expand:
            label_out.append("cat")
            data_out.append(np.array(j).transpose(2, 1, 0))

    others_path = glob.glob(current_path + "/data_set/others/*.*")  # 匹配所有的符合条件的文件，并将其以list的形式返回
    for repeat in range(5):
        for i in others_path:
            data_expand = data_augmentation(Image.open(i).convert("RGB"), (192, 192), 60)
            for j in data_expand:
                label_out.append("others")
                data_out.append(np.array(j).transpose(2, 1, 0))
    return data_out, label_out


def data_augmentation(img_in, object_size: tuple, rotate_degree: int):
    resized = ImageWarp.resize(img_in, object_size)
    crops = ImageWarp.five_crop(resized, tuple(int(index * 0.75) for index in object_size))
    object_out = []
    for crop in crops:
        for degree in range(0, 360, rotate_degree):
            image_rotated = ImageWarp.resize(ImageWarp.random_rotation(crop, (degree, degree + rotate_degree), True), object_size)
            object_out.append(image_rotated)
    return object_out


def data_shuffle(data_in, label_in):
    state = np.random.get_state()
    np.random.shuffle(data_in)
    np.random.set_state(state)
    np.random.shuffle(label_in)


def array_split(percentage, array_in):
    split1 = []
    split2 = []
    for i in range(array_in.shape[0]):
        if i < array_in.shape[0] * percentage:
            split1.append(array_in[i])
        else:
            split2.append(array_in[i])
    return np.array(split1), np.array(split2)


def list_split(percentage, list_in):
    list1 = []
    list2 = []
    length = len(list_in)
    for i in range(length):
        if i < length * percentage:
            list1.append(list_in[i])
        else:
            list2.append(list_in[i])
    return list1, list2


if __name__ == '__main__':
    data, label = get_image()
    data_shuffle(data, label)

    label_encoder = LabelEncoder()
    label_encoder.fit(label)
    label_encoded = np.array(label_encoder.transform(label))

    data_train, data_test = array_split(0.7, np.array(data))
    data_train_label, data_test_label = array_split(0.7, label_encoded)
    torch_train_set = torch.utils.data.TensorDataset(torch.from_numpy(data_train), torch.from_numpy(data_train_label))
    torch_test_set = torch.utils.data.TensorDataset(torch.from_numpy(data_test), torch.from_numpy(data_test_label))

    train_loader = torch.utils.data.DataLoader(
        dataset=torch_train_set,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=torch_test_set,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )

    with open("data_set/standard/data_set_shuffle_augmentation_encoder.pkl", "wb") as data_set_shuffle_augmentation_encoder:
        pickle.dump(label_encoder, data_set_shuffle_augmentation_encoder, protocol=4)
    with open("data_set/standard/data_set_shuffle_augmentation_train_data.pkl", "wb") as data_set_shuffle_augmentation_train_data:
        pickle.dump(train_loader, data_set_shuffle_augmentation_train_data, protocol=4)
    with open("data_set/standard/data_set_shuffle_augmentation_test_data.pkl", "wb") as data_set_shuffle_augmentation_test_data:
        pickle.dump(test_loader, data_set_shuffle_augmentation_test_data, protocol=4)

    pass

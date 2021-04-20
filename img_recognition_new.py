import torch
import pickle
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils import data as utils_data
from spp import SPPLayer


class Net(torch.nn.Module):
    def __init__(self, output_num):
        super(Net, self).__init__()
        self.cnn = torch.nn.Sequential(  # (batch_size , 3, input_size)
            torch.nn.Conv2d(in_channels=3,
                            out_channels=12,
                            kernel_size=[5, 5],
                            stride=[1, 1],  # 卷积核移动步长
                            padding=[1, 1],  # 处理边界时填充0的数量, 默认为0(不填充).
                            dilation=[1, 1],  # 采样间隔数量, 默认为1, 无间隔采样.
                            bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(12),

            torch.nn.Conv2d(in_channels=12,
                            out_channels=48,
                            kernel_size=[5, 5],
                            stride=[1, 1],  # 卷积核移动步长
                            padding=[1, 1],  # 处理边界时填充0的数量, 默认为0(不填充).
                            dilation=[1, 1],  # 采样间隔数量, 默认为1, 无间隔采样.
                            bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(48),

            torch.nn.Conv2d(in_channels=48,
                            out_channels=192,
                            kernel_size=[5, 5],
                            stride=[1, 1],  # 卷积核移动步长
                            padding=[1, 1],  # 处理边界时填充0的数量, 默认为0(不填充).
                            dilation=[1, 1],  # 采样间隔数量, 默认为1, 无间隔采样.
                            bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.spp = SPPLayer(level=8)
        spp_layer_output = self.spp.get_spp_len()
        self.full_connect = torch.nn.Sequential(  # (batch_size , 3, input_size)
            torch.nn.Linear(spp_layer_output, spp_layer_output * 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(spp_layer_output * 4, spp_layer_output * 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(spp_layer_output * 4, spp_layer_output),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(spp_layer_output, output_num)
        )

    def forward(self, x):
        x = self.cnn.forward(x)
        x = self.spp.forward(x)
        x = self.fn.forward(x)
        return x


class ImageRecognition(object):
    def __init__(self, epochs: int, learning_rate: float):
        self.net = Net(2)
        self.loss_func = torch.nn.MSELoss()  # 交叉熵损失不支持独热编码多位输入
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

        self.statistics = {}
        for i in range(2):
            self.statistics.update({
                str(i): {
                    "value": i,
                    "test_count": 0,
                    "predict_count": 0,
                    "correct_count": 0,
                    "recall": 0.0,
                    "precision": 0.0
                }
            })

        print(self.net)

    @staticmethod
    def one_hot_decoder(data_in):
        out = []
        array_data = data_in.numpy()
        for i in range(array_data.shape[0]):
            is_appended = False
            for j in range(array_data[i].shape[0]):
                if array_data[i][j] != 0:
                    out.append(j)
                    is_appended = True
                    break
            if not is_appended:
                out.append(0)
        return torch.from_numpy(np.array(out))

    def train(self, loader, start):
        # 训练过程
        for epoch in range(start, self.epochs):
            print("train epoch : {}".format(epoch))

            for step, (batch_data, batch_label) in enumerate(loader):
                out = self.net(batch_data.to(torch.float32))
                prediction = F.softmax(out, dim=0)
                loss = self.loss_func(prediction, batch_label.to(torch.float32))  # 自带独热编码
                print("{}{} ".format("train loss : ", loss))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

            if epoch % 20 == 0:
                checkpoint = {
                    "net": self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    "epoch": epoch
                }
                torch.save(checkpoint, 'checkpoint/checkpoint_epoch_%s.pth' % (str(epoch)))

    def data_statistics(self, predict, label):
        self.statistics[label]["test_count"] += 1
        self.statistics[predict]["predict_count"] += 1
        if predict == label:
            self.statistics[label]["correct_count"] += 1
        if self.statistics[label]["correct_count"] != 0:
            self.statistics[label]["precision"] = self.statistics[label]["correct_count"] / \
                                                  self.statistics[label]["test_count"]
            self.statistics[label]["recall"] = self.statistics[label]["correct_count"] / \
                                               self.statistics[label]["predict_count"]

    def test(self, loader):
        for (test_set, test_label) in loader:
            out = self.net(test_set.to(torch.float32))
            predict = torch.max(F.softmax(out, dim=0), 1)[1].numpy()
            for i in range(predict.shape[0]):
                self.data_statistics(str(predict[i]), str(test_label[i]))
        return self.statistics

    def test_plot(self):
        pass

    def save_net(self, path):
        torch.save(self.net, path)

    def load_net(self, path):
        self.net = torch.load(path)


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


class MyDataset(utils_data.Dataset):
    def __init__(self, data_in, label_in):
        self.data_in = data_in
        self.label_in = label_in

    def __len__(self):
        return len(self.label_in)

    def __getitem__(self, i):
        out = (self.data_in[i], self.label_in[i])
        return out



if __name__ == '__main__':
    with open("data_set/data_set.pkl", "rb") as data_set:
        label = pickle.load(data_set)
        data = pickle.load(data_set)

        label_encoder = LabelEncoder()
        label_encoder.fit(label)
        label_encoded = label_encoder.transform(label)

        data_train, data_test = list_split(0.7, data)
        data_train_label, data_test_label = array_split(0.7, label_encoded)
        torch_train_set = MyDataset(data_train, data_train_label)
        torch_test_set = MyDataset(data_test, data_test_label)

        train_loader = utils_data.DataLoader(
            dataset=torch_train_set,
            batch_size=64,
            shuffle=True,
            num_workers=6,
        )

        test_loader = utils_data.DataLoader(
            dataset=torch_test_set,
            batch_size=64,
            shuffle=True,
            num_workers=6,
        )

        image_recognition = ImageRecognition(500, 0.00001)
        image_recognition.train(train_loader, 0)
        test_result = image_recognition.test(test_loader)
        print(test_result)

import torch
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from spp import SPPLayer


class Net(torch.nn.Module):
    def __init__(self, output_num):
        super(Net, self).__init__()
        self.cnn = torch.nn.Sequential(  # (batch_size , 3, input_size)
            torch.nn.Conv2d(in_channels=3,
                            out_channels=12,
                            kernel_size=[3, 3],
                            stride=[1, 1],  # 卷积核移动步长
                            padding=[1, 1],  # 处理边界时填充0的数量, 默认为0(不填充).
                            dilation=[1, 1],  # 采样间隔数量, 默认为1, 无间隔采样.
                            bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(12),

            torch.nn.Conv2d(in_channels=12,
                            out_channels=48,
                            kernel_size=[3, 3],
                            stride=[1, 1],  # 卷积核移动步长
                            padding=[1, 1],  # 处理边界时填充0的数量, 默认为0(不填充).
                            dilation=[1, 1],  # 采样间隔数量, 默认为1, 无间隔采样.
                            bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(48),
        )
        self.spp = SPPLayer(level=5)
        full_connect_input = self.spp.get_spp_len() * 48
        self.full_connect = torch.nn.Sequential(  # (batch_size , 3, input_size)
            torch.nn.Linear(full_connect_input, full_connect_input),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(full_connect_input, full_connect_input),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(full_connect_input, full_connect_input),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(full_connect_input, output_num),
        )

    def forward(self, x):
        x = self.cnn.forward(x)
        x = self.spp.forward(x)
        x = self.full_connect.forward(x)
        return x


class ImageRecognition(object):
    def __init__(self, epochs: int, learning_rate: float):
        self.net = Net(1)
        self.loss_func = torch.nn.MSELoss()  # 交叉熵损失不支持独热编码多位输入
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

        self.statistics = {}
        for i in range(2):
            self.statistics.update({
                str(i): {
                    "name": i,
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

    def train(self, data_in, label_in, start):
        # 训练过程
        for epoch in range(start, self.epochs):
            print("train epoch : {}".format(epoch))
            avg_loss = 0
            for i in range(len(label_in)):
                out = self.net(data_in[i][None, ...])
                prediction = torch.sigmoid(out)
                loss = self.loss_func(prediction, torch.tensor([[label_in[i]]]).to(torch.float32))  # 自带独热编码
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
                avg_loss += loss
            avg_loss = avg_loss / len(label_in)
            print("{}{} ".format("train avg loss : ", avg_loss))

            if epoch % 20 == 0:
                checkpoint = {
                    "net": self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    "epoch": epoch
                }
                torch.save(checkpoint, 'checkpoint/checkpoint_epoch_%s.pth' % (str(epoch)))

    def test(self, data_in, label_in, encoder):
        for i in range(len(label_in)):
            out = self.net(data_in[i][None, ...])
            test = torch.round(torch.sigmoid(out))
            predict = torch.round(torch.sigmoid(out)).detach().numpy()[0].astype(int)
            predict_encoded = encoder.inverse_transform(predict)
            label_in_encoded = encoder.inverse_transform([label_in[i]])
            print("predict: {}; origin: {};".format(predict_encoded, label_in_encoded))
            self.data_statistics(str(predict[0]), str(label_in[i]), label_in_encoded[0])
        return self.statistics

    def data_statistics(self, predict, label_in, name):
        self.statistics[label_in]["name"] = name
        self.statistics[label_in]["test_count"] += 1
        self.statistics[predict]["predict_count"] += 1
        if predict == label_in:
            self.statistics[label_in]["correct_count"] += 1
        if self.statistics[label_in]["correct_count"] != 0:
            self.statistics[label_in]["precision"] = self.statistics[label_in]["correct_count"] / \
                                                  self.statistics[label_in]["test_count"]
            self.statistics[label_in]["recall"] = self.statistics[label_in]["correct_count"] / \
                                               self.statistics[label_in]["predict_count"]

    def test_plot(self):
        pass

    def save_net(self, path):
        torch.save(self.net, path)

    def load_net(self, path):
        self.net = torch.load(path)

    def start_from_checkpoint(self, path, obj_epochs:int):
        checkpoint_loaded = torch.load(path)  # 加载断点
        self.net.load_state_dict(checkpoint_loaded['net'])  # 加载模型可学习参数
        self.optimizer.load_state_dict(checkpoint_loaded['optimizer'])  # 加载优化器参数
        self.epochs = obj_epochs
        start_epoch = checkpoint_loaded['epoch']  # 设置开始的epoch
        return start_epoch


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


def data_shuffle(data_in, label_in):    # 同序shuffle
    state = np.random.get_state()
    np.random.shuffle(data_in)
    np.random.set_state(state)
    np.random.shuffle(label_in)
    with open("data_set/data_set_shuffle.pkl", "wb") as data_set_shuffle:
        pickle.dump(label_in, data_set_shuffle)
        pickle.dump(data_in, data_set_shuffle)



if __name__ == '__main__':
    with open("data_set/data_set.pkl", "rb") as data_set:
        label = pickle.load(data_set)
        data = pickle.load(data_set)
        label_encoder = LabelEncoder()
        label_encoder.fit(label)
        label_encoded = label_encoder.transform(label)

    data_shuffle(data, label_encoded)
    with open("data_set/data_set_shuffle.pkl", "rb") as data_set:
        label_encoded = pickle.load(data_set)
        data = pickle.load(data_set)

        data_train, data_test = list_split(0.7, data)
        data_train_label, data_test_label = array_split(0.7, label_encoded)
        # torch.utils.data.DataLoader 不支持 SPP NET 做批训练

        image_recognition = ImageRecognition(200, 0.00001)
        # start_epoch = 0
        # start_epoch = image_recognition.start_from_checkpoint("checkpoint/checkpoint_epoch_60.pth", obj_epochs=61)
        # image_recognition.train(data_train, data_train_label, start_epoch)
        # image_recognition.save_net("net/spp_net.pkl")
        image_recognition.load_net("net/spp_net.pkl")
        test_result = image_recognition.test(data_test, data_test_label, label_encoder)

        print(test_result)

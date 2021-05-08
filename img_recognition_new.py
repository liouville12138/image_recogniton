import torch
import joblib
import numpy as np
from torchsummary import summary


class Net(torch.nn.Module):
    def __init__(self, input_size: tuple, output_num):
        super(Net, self).__init__()
        self.cnn = torch.nn.Sequential(  # (batch_size , 3, input_size)
            torch.nn.Conv2d(in_channels=3,
                            out_channels=9,
                            kernel_size=[3, 3],
                            stride=[1, 1],  # 卷积核移动步长
                            padding=[1, 1],  # 处理边界时填充0的数量, 默认为0(不填充).
                            dilation=[1, 1],  # 采样间隔数量, 默认为1, 无间隔采样.
                            bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(9),

            torch.nn.Conv2d(in_channels=9,
                            out_channels=27,
                            kernel_size=[3, 3],
                            stride=[1, 1],  # 卷积核移动步长
                            padding=[1, 1],  # 处理边界时填充0的数量, 默认为0(不填充).
                            dilation=[1, 1],  # 采样间隔数量, 默认为1, 无间隔采样.
                            bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(27),

            torch.nn.Conv2d(in_channels=27,
                            out_channels=81,
                            kernel_size=[3, 3],
                            stride=[1, 1],  # 卷积核移动步长
                            padding=[1, 1],  # 处理边界时填充0的数量, 默认为0(不填充).
                            dilation=[1, 1],  # 采样间隔数量, 默认为1, 无间隔采样.
                            bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(81),

            torch.nn.Conv2d(in_channels=81,
                            out_channels=243,
                            kernel_size=[3, 3],
                            stride=[1, 1],  # 卷积核移动步长
                            padding=[1, 1],  # 处理边界时填充0的数量, 默认为0(不填充).
                            dilation=[1, 1],  # 采样间隔数量, 默认为1, 无间隔采样.
                            bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(243),

        )
        full_connect_input = 12 * 12 * 243
        self.full_connect = torch.nn.Sequential(  # (batch_size , 3, input_size)
            torch.nn.Linear(full_connect_input, int(full_connect_input / 2)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(int(full_connect_input / 2), int(full_connect_input / 4)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(int(full_connect_input / 4), output_num),
        )

    def forward(self, x):
        x = self.cnn.forward(x)
        x = x.view(x.size(0), -1)  # 保留batch
        x = self.full_connect.forward(x)
        x = x.squeeze(-1)
        return x


class ImageRecognition(object):
    def __init__(self, input_size: tuple, learning_rate: float):
        self.net = Net(input_size, 1)
        summary(self.net, input_size=(3, 192, 192))
        self.loss_func = torch.nn.MSELoss()  # 交叉熵损失不支持独热编码多位输入
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

    def train(self, loader, start, epochs: int):
        # 训练过程
        for epoch in range(start, epochs):
            print("train epoch : {}".format(epoch))
            avg_loss = 0
            for step, (batch_data, batch_label) in enumerate(loader):
                out = self.net(batch_data.to(torch.float32))
                prediction = torch.sigmoid(out)
                loss = self.loss_func(prediction, batch_label.to(torch.float32))  # 自带独热编码
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
                avg_loss += loss
            avg_loss = avg_loss / len(loader)
            print("{}{} ".format("train avg loss : ", avg_loss))

            if epoch % 20 == 0:
                checkpoint = {
                    "net": self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    "epoch": epoch
                }
                torch.save(checkpoint, 'checkpoint/standard/checkpoint_epoch_%s.pth' % (str(epoch)))

    def test(self, loader, encoder):
        for step, (batch_data, batch_label) in enumerate(loader):
            out = self.net(batch_data.to(torch.float32))
            predict = torch.round(torch.sigmoid(out)).detach().numpy().astype(int)
            predict = np.reshape(predict, predict.shape[0])
            predict_encoded = encoder.inverse_transform(predict)
            label_in_encoded = encoder.inverse_transform(batch_label)
            for i in range(label_in_encoded.shape[0]):
                print("predict: {}; origin: {};".format(predict_encoded[i], label_in_encoded[i]))
                self.data_statistics(str(predict[i]), str(batch_label.numpy()[i]), label_in_encoded[i])
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

    def start_from_checkpoint(self, path):
        checkpoint_loaded = torch.load(path)  # 加载断点
        self.net.load_state_dict(checkpoint_loaded['net'])  # 加载模型可学习参数
        self.optimizer.load_state_dict(checkpoint_loaded['optimizer'])  # 加载优化器参数
        start = checkpoint_loaded['epoch']  # 设置开始的epoch
        return start

if __name__ == '__main__':
    with open("data_set/standard/data_set_shuffle_augmentation_encoder.pkl", "rb") as data_set_shuffle_augmentation_encoder:
        label_encoder = joblib.load(data_set_shuffle_augmentation_encoder)
    with open("data_set/standard/data_set_shuffle_augmentation_train_data.pkl", "rb") as data_set_shuffle_augmentation_train_data:
        train_loader = joblib.load(data_set_shuffle_augmentation_train_data)
    with open("data_set/standard/data_set_shuffle_augmentation_test_data.pkl", "rb") as data_set_shuffle_augmentation_test_data:
        test_loader = joblib.load(data_set_shuffle_augmentation_test_data)

    image_recognition = ImageRecognition((256, 256), 0.0001)
    start_epoch = 0
    # start_epoch = image_recognition.start_from_checkpoint("checkpoint/standard/checkpoint_epoch_60.pth")
    image_recognition.train(train_loader, start_epoch, 200)
    image_recognition.save_net("net/standard/spp_net.pkl")
    # load_net("net/standard/spp_net.pkl")
    test_result = image_recognition.test(test_loader, label_encoder)

    print(test_result)

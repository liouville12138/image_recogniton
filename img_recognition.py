import os, glob
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.autograd import Variable

from full_connect import FullConnectNet
from cnn import Cnn
from spp import SPPLayer
from data_preprocessing import LoadImages


class ImageRecognition(object):
    def __init__(self, epoches):
        self.criterian = nn.CrossEntropyLoss()  # 构造函数有自己的参数
        self.epoches = epoches
        self.learning_rate = 1e-3
        self.module = None


    @staticmethod
    def CreatNet(self):
        cnn = Cnn(3, 3)
        self.module = nn.Sequential(
          cnn.layer()
        )



    def forward(self,x):
        cnn = Cnn(3, 3)
        out = cnn.forward(x)
        spp = SPPLayer(3, 3)
        out = spp.forward(out)
        fn = FullConnectNet(len(out))
        out = fn.forward(out)
        return out

    def train(self, train_data):

        optimizer = torch.optim.RMSprop(model.parameters(), lr=self.learning_rate)
        # 训练过程
        for i in range(self.epoches):
            running_loss = 0.0
            running_acc = 0.0
            for data_dict in train_data:
                print(data_dict)
                for label in data_dict:
                    # 转换为Variable类型
                    img = Variable(data_dict[label])
                    label = Variable(label)
                    Optimizer.zero_grad()
                    # feedforward
                    output = self.net(img)
                    loss = self.criterian(output, label)
                    # backward
                    loss.backward()
                    Optimizer.step()
                    # 记录当前的lost以及batchSize数据对应的分类准确数量
                    running_loss += loss.data[0]
                    _, predict = torch.max(output, 1)
                    correct_num = (predict == label).sum()
                    running_acc += correct_num.data[0]
            # 计算并打印训练的分类准确率
            running_loss /= len(train_data)
            running_acc /= len(train_data)
            print("[%d/%d] Loss: %.5f, Acc: %.2f" % (i + 1, self.epoches, running_loss, 100 * running_acc))

    def test(self, test_data):
        # 测试过程
        test_loss = 0.0
        test_acc = 0.0
        for (img, label) in test_data:
            # 转换为Variable类型
            img = Variable(img)
            label = Variable(label)
            # feedforward
            output = self.net(img)
            loss = self.criterian(output, label)
            # 记录当前的lost以及累加分类正确的样本数
            test_loss += loss.data[0]
            _, predict = torch.max(output, 1)
            num_correct = (predict == label).sum()
            test_acc += num_correct.data[0]
        # 计算并打印测试集的分类准确率
        test_loss /= len(test_data)
        test_acc /= len(test_data)
        print("Test: Loss: %.5f, Acc: %.2f %%" % (test_loss, 100 * test_acc))

    def run(self, image):
        output = self.net(image)
        return output


if __name__ == '__main__':
    # 数据标签获取
    current_path = os.path.dirname(os.path.abspath(__file__))
    img_path_list = glob.glob(current_path + "/train_set/*.jpg")  # 匹配所有的符合条件的文件，并将其以list的形式返回
    test = LoadImages()
    label_list = []
    for i in img_path_list:
        label_list.append(torch.tensor([1]))
    test.get_image_by_list(img_path_list, label_list)
    data = test.transform()
    image_recognition = ImageRecognition(1000)

    image_recognition.train(data)
    image_recognition.test()

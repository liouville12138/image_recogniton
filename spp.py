# coding=utf-8
import math
import torch
import torch.nn.functional as F

# 构建SPP层(空间金字塔池化层)
class SPPLayer(torch.nn.Module):

    def __init__(self, level=5, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.pool_type = pool_type
        self.level = level

    @staticmethod
    def _calc_window_size(pool_size_x, pool_size_y, width, height):
        window_width = math.ceil(width / pool_size_x)
        padding_width = math.ceil((window_width * pool_size_x - width) / 2)
        window_height = math.ceil(height / pool_size_y)
        padding_height = math.ceil((window_height * pool_size_y - height) / 2)
        return window_width, window_height, padding_width, padding_height

    def _single_level(self, x, pool_size_x, pool_size_y):
        batch, channel, height, width = x.size()  # num:样本数量 c:通道数 h:高 w:宽

        window_width, window_height, padding_width, padding_height = self._calc_window_size(pool_size_x, pool_size_y, width, height)

        pool_size = (window_height, window_width)
        stride = (window_height, window_width)
        padding = (padding_height, padding_width)

        # 选择池化方式
        if self.pool_type == 'max_pool':
            tensor = F.max_pool2d(x, kernel_size=pool_size, stride=stride, padding=padding).view(batch, -1)
        else:
            tensor = F.avg_pool2d(x, kernel_size=pool_size, stride=stride, padding=padding).view(batch, -1)

        # 展开、拼接
        x_flatten = tensor.view(batch, -1)    #  view 相当于 resize
        return x_flatten

    def get_spp_len(self):
        length = 0
        for i in range(self.level):
            length = length + i * i
        return length

    def forward(self, x):
        for i in range(1, self.level):
            if i == 1:
                levels_flatten = self._single_level(x, i, i)
            else:
                levels_flatten = torch.cat((levels_flatten, self._single_level(x, i, i)), 1)
        return levels_flatten

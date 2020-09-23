# 训练
import cv2

from model.unet_model import UNet
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import numpy as np

def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset, batch_size=batch_size, shuffle=True)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    # 定义loss
    criterion = nn.BCEWithLogitsLoss()

    # best_loss统计
    best_loss = float('inf')

    # 训练epochs次
    for epoch in range(epochs):
        net.train()
        for image, label in train_loader:
            optimizer.zero_grad()
            # 拷贝数据到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 输出预测结果
            pred = net(image)

            ii = image[0].to(device=device, dtype=torch.uint8)
            imShow = np.array(ii.data.cpu())
            imShow = imShow.reshape(512, 512, 3)

            mm = label[0].to(device=device, dtype=torch.uint8)
            maskShow = np.array(mm.data.cpu())
            maskShow = maskShow.reshape(512, 512, 1)*255

            pm = pred[0].to(device=device)
            predShow = np.array(pm.data.cpu())
            predShow = predShow.reshape(512, 512, 1)

            cv2.imshow("image", imShow)
            cv2.imshow("mask", maskShow)
            cv2.imshow("pred", predShow)
            cv2.waitKey()

            # 计算loss
            loss = criterion(pred, label)
            print('Loss/train: ', loss.item())
            # 保存loss值最小的参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络
    net = UNet(n_channels=3, n_classes=1)
    # 网络拷贝到device中
    net.to(device=device)
    #指定训练集
    data_path = 'data/train/'
    # 开始训练
    train_net(net, device, data_path)

'''
    训练生成mask的部分
'''
import cv2
import numpy as np
import torch
from torch import optim, nn
from model.unet_model import UNet
from utils.dataset import TrainData_Loader
from torch import optim
import torch.nn as nn
import torch
from torch.utils.data import Dataset

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss

if __name__ == '__main__':



    # 加载训练集
    # 正式用
    # TrainData_dataset = TrainData_Loader('D:/Research/Dataset/3DOH50K/trainset/')
    # print("数据个数： ", len(TrainData_dataset))
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=TrainData_dataset,
    #     batch_size=64,
    #     shuffle=True
    # )

    # ## 测试用
    TrainData_dataset = TrainData_Loader('D:/Research/Dataset/3DOH50K/subset/')
    print("数据个数： ", len(TrainData_dataset))
    train_loader = torch.utils.data.DataLoader(
        dataset=TrainData_dataset,
        batch_size=16,
        shuffle=True
    )
    # TrainData_dataset = TrainData_Loader('D:/Research/Dataset/3DOH50K/notset/')
    # print("数据个数： ", len(TrainData_dataset))
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=TrainData_dataset,
    #     batch_size=3,
    #     shuffle=False
    # )

    print(len(train_loader))

    out_dir="model_Dice_ep40_bs16.pkl"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = UNet(n_channels=3, n_classes=1).to(device=device)

    # 定义下降算法
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    # optimizer = optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=1e-8, momentum=0.9)

    # 定义loss
    # criterion = nn.L1Loss(reduction='mean')
    # criterion = nn.BCEWithLogitsLoss()
    criterion = DiceLoss()
    best_loss = float('inf')

    # 训练
    epochs = 40
    for epoch in range(epochs):
        model.train()
        cnt = 0
        for image, mask in train_loader:
            cnt += 1
            optimizer.zero_grad()

            image = image.to(device=device,dtype=torch.float32)

            mask = mask.to(device=device,dtype=torch.float32)

            pred = model(image)

            pred = torch.sigmoid(pred)

            ii = (image[0] * 255.0).to(device=device, dtype=torch.uint8)
            imShow = np.array(ii.data.cpu())
            imShow = np.transpose(imShow, axes=[1, 2, 0])

            mm = (mask[0] * 255.0).to(device=device, dtype=torch.uint8)
            maskShow = np.array(mm.data.cpu())
            maskShow = maskShow.reshape(256, 256, 1)

            pm = (pred[0] * 255.0).to(device=device, dtype=torch.uint8)
            predShow = np.array(pm.data.cpu())
            predShow = predShow.reshape(256, 256, 1)

            # cv2.imshow("image", imShow)
            # cv2.imshow("mask", maskShow)
            # cv2.imshow("pred", predShow)
            # cv2.waitKey()

            cv2.imwrite("image.jpg", imShow)
            cv2.imwrite("pred.jpg", predShow)

            # print(pred[0].sum(), mask[0].sum())
            loss = criterion(pred, mask)

            if cnt % 40 == 0 :
                print('Epoch: ', epoch, ' Loss/train: ', loss.item())
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), out_dir)
            loss.backward()
            optimizer.step()


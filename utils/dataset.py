# 加载数据集
import json

import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
from utils.imutils import img_reshape, scale
import numpy as np

class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
        self.labels_path = glob.glob(os.path.join(self.data_path, 'label/*.png'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强， 1-水平翻转 0-垂直翻转 -1-水平+垂直旋转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # 生成image_path
        image_path = self.imgs_path[index]
        # 生成label_path
        label_path = self.labels_path[index]
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        # 数据转为单通道
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(3, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强
        # flipCode = random.choice([-1, 0, 1, 2])
        # if flipCode != 2:
        #     image = self.augment(image, flipCode)
        #     label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        return len(self.imgs_path)



def adjustBbox(bbox, w, h):
    # left top
    x1, y1 = bbox[0]

    x1 = max(0, x1)
    y1 = max(0, y1)

    # right bottom
    x2, y2 = bbox[1]
    x2 = min(x2, w)
    y2 = min(y2, h)

    # 如果出现了不符合的bbox，大家都为0
    if x1 > w or x2 < 0 or y1 > h or y2 < 0:
        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0

    return x1, y1, x2, y2

class TrainData_Loader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'images/*.jpg'))
        self.masks_path = glob.glob(os.path.join(self.data_path, 'masks/*.png'))
        jsons_path = os.path.join(self.data_path, 'annots.json')
        self.joints = json.load(open(jsons_path, encoding='utf-8'))

    def __getitem__(self, index):
        # 生成image_path
        image_path = self.imgs_path[index]
        # 生成mask_path
        mask_path = self.masks_path[index]

        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        # 加载bounding box [[x1, y1], [x2, y2]]
        bbox = self.joints[image_path.split('\\')[-1].replace('.jpg', '')]['bbox']

        h, w = image.shape[:2]


        # 调整bounding box
        x1, y1, x2, y2 = adjustBbox(bbox, w, h)

        # 对数据进行裁剪 补充
        image = image[y1:y2, x1:x2, :]
        mask = mask[y1:y2, x1:x2, :]

        h, w = image.shape[:2]
        try:
            if h != 256 or w != 256:
                max_size = max(h, w)
                ratio = 256 / max_size
                if image is None:
                    pass
                else:
                    image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
                image = img_reshape(image)
                if mask is None:
                    pass
                else:
                    mask = cv2.resize(mask, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
                mask = img_reshape(mask)
        except:
            print(image_path)
            print(mask_path)
            print(bbox)

        # print(image.shape)
        # cv2.imshow("image", image)
        # cv2.imshow("mask", mask)
        # cv2.waitKey()


        # 数据转为[3, height, width]
        image = np.transpose(image, axes=[2, 0, 1])


        if mask is None:
            pass
        else:
           mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask = mask.reshape(1, mask.shape[0], mask.shape[1])

        if mask.max() > 1:
            mask = mask / 255

        if image.max() > 1:
            image = image / 255

        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return image, mask
    def __len__(self):
        return len(self.imgs_path)

class TestData_Loader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'images/*.jpg'))


    def __getitem__(self, index):
        # 生成image_path
        image_path = self.imgs_path[index]

        # 读取测试图片
        image = cv2.imread(image_path)

        # 缩小size到256*256
        h, w = image.shape[:2]

        if h != 256 or w != 256:
            max_size = max(h, w)
            ratio = 256 / max_size
            image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
            image = img_reshape(image)

        # cv2.imshow("image", image)
        # cv2.waitKey()

        # 数据转为[3, height, width]
        image = np.transpose(image, axes=[2, 0, 1])
        if image.max() > 1:
            image = image / 255.0
        return image

    def __len__(self):
        return len(self.imgs_path)


if __name__ == '__main__':
    isbi_dataset = ISBI_Loader('D:/Research/Dataset/3DOH50K/trainset/')
    print("数据个数： ", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(
        dataset=isbi_dataset,
        batch_size=2,
        shuffle=True
    )
    for image, label in train_loader:
        print(image.shape)
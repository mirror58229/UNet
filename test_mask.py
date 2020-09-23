'''
    训练生成mask的部分
'''
import glob

import cv2
import numpy as np
import torch
from torch import optim, nn

from utils.dataset import TestData_Loader
from model.unet_model import UNet

if __name__ == '__main__':
    batch_size = 64
    # 加载测试集
    # TestData_dataset = TestData_Loader('D:/Research/Dataset/3DOH50K/testset/')
    # tests_path = 'D:/Research/Dataset/3DOH50K/testset/masks/'

    TestData_dataset = TestData_Loader('D:/Research/Dataset/3DOH50K/notset/')
    tests_path = 'D:/Research/Dataset/3DOH50K/notset/masks_test/'
    print("数据个数： ", len(TestData_dataset))
    test_loader = torch.utils.data.DataLoader(
        dataset=TestData_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    out_dir="model_BCE_bs16.pkl"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = UNet(n_channels=3, n_classes=1).to(device=device)
    model.load_state_dict(torch.load(out_dir))
    model.eval()

    batchcnt = 0

    with torch.no_grad():
        for image in test_loader:
            image = image.to(device=device,dtype=torch.float32)
            pred = model(image)
            # 写入batch中的每个数据
            cnt = 0
            for mask in pred:

                # ii = (image[cnt] * 255.0).to(device=device, dtype=torch.uint8)
                # imShow = np.array(ii.data.cpu())
                # imShow = np.transpose(imShow, axes=[1, 2, 0])

                # print(mask)
                mm = (mask[0]*255.0).to(device=device)
                maskShow = np.array(mm.data.cpu())
                maskShow = maskShow.reshape(256, 256, 1)


                # cv2.imshow("image", imShow)
                # cv2.imshow("testmask", maskShow)
                # cv2.waitKey()

                # 写入
                test_path = tests_path + '%05d.jpg' % (batchcnt * batch_size + cnt)
                cnt = cnt + 1
                cv2.imwrite(test_path , maskShow)

            batchcnt = batchcnt+1
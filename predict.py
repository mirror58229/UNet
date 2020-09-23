import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load('model.pth', map_location=device))
    net.eval()
    tests_path = glob.glob('D:/Research/Dataset/3DOH50K/testset/*.jpg')
    for test_path in tests_path:
        save_res_path = test_path.split('.')[0] + '_res.jpg'
        img = cv2.imread(test_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        pred = net(img_tensor)
        pred = np.array(pred.data.cpu()[0])[0]
        pred[pred>=0.5]=255
        pred[pred<0.5]=0
        cv2.imwrite(save_res_path, pred)
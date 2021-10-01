import pytorch_ssim
import torch
from torch.autograd import Variable
import cv2
import numpy as np

img_1 = cv2.imread('../data/5.jpg')
img_2 = cv2.imread('../data/7.jpg')

img1 = torch.from_numpy(np.rollaxis(img_1, 2)).float().unsqueeze(0)/255.0
img2 = torch.from_numpy(np.rollaxis(img_2, 2)).float().unsqueeze(0)/255.0

print(img1.shape)

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()

img1 = Variable(img1)
img2 = Variable(img2)

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()

print(pytorch_ssim.ssim(img1,img2))

ssim_loss = pytorch_ssim.SSIM(window_size = 11)

print(ssim_loss(img1,img2))
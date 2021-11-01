# Author: Fengx

import os
import time
import random
import numpy as np
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
from torchvision import transforms
from data import TEST_DATASET
from PIL import Image

## Select the model of your task for evaluation.
from model_reflection import Model
#from model_dehazing import Model
#from model_deraining import Model

# ------------------------------tensor2numpy2ssim/psnr------------------
import assess
def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip((image_numpy+1) /2.0, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    return image_numpy

def t2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    return image_numpy

# -------------------------Main Testing------------------------------------
EPS = 1e-12
def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Deep-Masking Generative Network for Background Restoration from Superimposed Images')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--testpath', default='./testing/', help='path to testing images')
    # Select the *.pth model to deal with specific task.
    parser.add_argument('--model', default='./checkpoints/Dereflection_netG.pth', help='path to pre-trained')
    args = parser.parse_args()

    # Create model
    model = Model().cuda()
    with torch.no_grad():
        print('Warning! Loading pre-trained weights. XD')
        model = nn.DataParallel(model)
        #model = model.module
        try:
            model.load_state_dict(torch.load(args.model,map_location='cuda:0'))
        except:
            model = model.module
            model.load_state_dict(torch.load(args.model,map_location='cuda:0'))
        #model = model.module
        print('Model loaded.')
        print('Model weight initilized.')

        batch_size = args.bs
        # Loading test data
        test_dataset = TEST_DATASET(args.testpath,256,256,0)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=1,
                                                shuffle=True,
                                                num_workers=0
                                                )
        
        print('-------------------------------------------------')
        print('||||||   Attention!!! data testing now!   ||||||') 

        # Start testing data pairs...
        with torch.no_grad():
            # Switch to train mode
            model.eval()          
            for c,img in enumerate(test_loader):
                # Testing
                output_r,output_c,output_t,maskB,mask,RDM = model(img.cuda())
                # Save
                Image.fromarray(tensor2im(output_t).astype(np.uint8)).save('results/'+str(c)+'output_t.png')
                Image.fromarray(tensor2im(output_c).astype(np.uint8)).save('results/'+str(c)+'output_c.png')
                Image.fromarray(tensor2im(output_r).astype(np.uint8)).save('results/'+str(c)+'output_r.png')
                Image.fromarray(tensor2im(img).astype(np.uint8)).save('results/'+str(c)+'input_a.png')

if __name__ == '__main__':
    main()

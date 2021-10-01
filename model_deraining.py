import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vgg import Vgg19

class hyper(nn.Module):
    def __init__(self):
        super(hyper, self).__init__()
        self.vgg = Vgg19(requires_grad=False)
        self.conv = nn.Sequential(
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(1475,64,3,1,0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(64,64,3,1,0),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        hypercolumn = self.vgg(x)
        _,C,H,W = x.size()
        hypercolumn = [F.interpolate(feature.detach(),size=(H,W),mode='bilinear',align_corners=False) for feature in hypercolumn]
        inputs = [x]
        inputs.extend(hypercolumn)
        inputs = torch.cat(inputs, dim=1)
        output = self.conv(inputs)
        #output= torch.cat([output,x],dim=1)
        return output

class Transit_mask(nn.Module):
    def __init__(self, in_c=64, out_c=64):
        super(Transit_mask,self).__init__()
        self.conv = nn.Sequential(
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(in_c,out_c,3,1,padding=0),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(in_c,out_c,3,1,padding=0),
                nn.ReLU(inplace=True),
            )
        self.mask = nn.Sequential(
                #nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(in_c,64,1,1,padding=0),
                nn.Tanh(),
                nn.Conv2d(64,1,1,1,padding=0),
                nn.Sigmoid()
            )

    def forward(self,input):
        out = self.conv(input)
        mask = self.mask(out)
        out1 = out*mask
        out2 = out*(1-mask)
        return out1,out2,mask,1-mask

class Mask_learning(nn.Module):
    def __init__(self,channels):
        super(Mask_learning,self).__init__()
        self.mask = nn.Sequential(
                nn.Conv2d(channels,channels,1,1,padding=0),
                nn.Tanh(),
                nn.Conv2d(channels,1,1,1,padding=0),
                nn.Sigmoid()
            )

    def forward(self,feature):
        return self.mask(feature)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.Depthwise_conv = nn.Sequential(
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(channel, channel, 5, 4, 0, groups=channel, bias=False),
            nn.ReLU(inplace=True)
            )
        self.avgp = nn.AdaptiveAvgPool2d(1)
        self.Channelwise_conv1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )
        self.mask = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0, groups=channel, bias=False),
            nn.Sigmoid()
            )
        self.sig = nn.Sigmoid()

    def forward(self, x, state=1):
        b, c, _, _ = x.size()
        feat = self.Depthwise_conv(x)
        mask = self.mask(feat)
        feat = feat*mask
        mean = self.avgp(feat).view(b,c)
        cw1 = self.Channelwise_conv1(mean)
        weight = self.sig(cw1).view(b,c,1,1) * state
        return x * weight.expand_as(x)

class SE_ResBlock(nn.Module):
    def __init__(self, in_c, channels, kernel_size,reduction=16):
        super(SE_ResBlock, self).__init__()
        self.conv1 = Padding_Conv(in_c,channels)
        self.conv2 = Padding_Conv(channels,channels)
        self.se = SELayer(channels, reduction)

    def forward(self, x):
        residual = x
        #scale = 0.1
        scale = 1
        out = self.conv1(x)
        out = self.conv2(out)

        out = self.se(out)
        out = out*scale
        out += residual
        return out

class Seperation_module(nn.Module):
    def __init__(self, in_c, out_c, print_mask=True, downsample=False,dilation=1):  
        super(Seperation_module,self).__init__()
        self.mask_1 = Mask_learning(in_c)

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d((dilation,dilation,dilation,dilation)),
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=0, bias=False,dilation=dilation),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((dilation,dilation,dilation,dilation)),
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=0, bias=False,dilation=dilation),
            nn.ReLU(inplace=True),
            )
        self.conv3 = Padding_Conv(out_c,out_c*2,stride=2)

        self.print_mask = print_mask
        self.down = downsample

    def forward(self,B):
        # 2 convs 3x3
        feat_B = self.conv1(B)
        # learn mask 
        mask1 = self.mask_1(feat_B)
        # apply mask
        out_B = mask1*feat_B + B

        concat_feat = out_B
        if self.down:
            out_B = self.conv3(out_B)
            return out_B,mask1.data,concat_feat
        return out_B,mask1.data

class Padding_Conv(nn.Module):
    def __init__(self,in_c,out_c,stride=1):
        super(Padding_Conv,self).__init__()
        self.conv = nn.Sequential(
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=0, bias=False),
                nn.ReLU(inplace=True)
            )

    def forward(self,feature):
        return self.conv(feature)

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()  
        self.up = nn.Sequential(
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            )      

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.up(torch.cat([up_x, concat_with], dim=1))

class OutConv(nn.Sequential):
    def __init__(self,):
        super(OutConv, self).__init__()  
        self.outconv = nn.Sequential(
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            #nn.Tanh()
            )      
    def forward(self, x):
        return self.outconv(x)

class R_Branch(nn.Sequential):
    def __init__(self,):
        super(R_Branch, self).__init__() 
        self.s_m0 = Seperation_module(64,64,downsample=True)
        self.s_m1 = Seperation_module(128,128,downsample=True)
        self.s_m2 = Seperation_module(256,256)
        self.s_m3 = Seperation_module(256,256)
        self.s_m4 = Seperation_module(256,256)
        self.s_m5 = Seperation_module(256,256)
        self.s_m21 = Seperation_module(256,256)
        self.s_m31 = Seperation_module(256,256)
        self.s_m41 = Seperation_module(256,256)
        self.s_m51 = Seperation_module(256,256)
        self.s_m6 = Seperation_module(128,128)
        self.s_m7 = Seperation_module(64,64)
        self.up1 = UpSample(256+128,128)
        self.up2 = UpSample(128+64,64)

    def forward(self, feat_B):
        outB0,mask0,c0 = self.s_m0(feat_B)
        outB1,mask1,c1 = self.s_m1(outB0)
        outB2,mask2 = self.s_m2(outB1)
        outB3,mask3 = self.s_m3(outB2)
        outB4,mask4 = self.s_m4(outB3)
        outB5,mask5 = self.s_m5(outB4)
        outB6,mask6 = self.s_m21(outB5)
        outB7,mask7 = self.s_m31(outB6)
        outB8,mask8 = self.s_m41(outB7)
        outB5,mask5 = self.s_m51(outB8)

        outB5 = self.up1(outB5,c1)
        outB6,mask6 = self.s_m6(outB5)
        outB6 = self.up2(outB6,c0)
        outB7,mask7 = self.s_m7(outB6)
        return outB7

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.head_feat = hyper()
        self.seperator = Transit_mask()
        self.seperator2 = Transit_mask_B()

        self.s_m0 = Seperation_module(64,64,downsample=True)
        self.s_m1 = Seperation_module(128,128,downsample=True)
        self.s_m2 = Seperation_module(256,256)
        self.s_m3 = Seperation_module(256,256)
        self.s_m4 = Seperation_module(256,256)
        self.s_m5 = Seperation_module(256,256)
        self.s_m21 = Seperation_module(256,256)
        self.s_m31 = Seperation_module(256,256)
        self.s_m41 = Seperation_module(256,256)
        self.s_m51 = Seperation_module(256,256)
        self.s_m6 = Seperation_module(128,128)
        self.s_m7 = Seperation_module(64,64)

        self.up1 = UpSample(256+128,128)
        self.up2 = UpSample(128+64,64)

        # Coarse Output
        self.outconv_B_coarse = OutConv()
        self.outconv_R = OutConv()
        # R_Branch
        self.r_branch = R_Branch()
        # Channel-wise Concat
        #self.cwc = Channel_Wise_Concat()
        self.Channel_weight = SELayer(64)
        # Refine_Branch
        self.refine = Refined()
        

    def forward(self,I):
        input = self.head_feat(I)
        # Coarse
        feat_B,feat_R,mask_B,mask_R = self.seperator(input)
        outB0,mask0,c0 = self.s_m0(feat_B)
        outB1,mask1,c1 = self.s_m1(outB0)
        outB2,mask2 = self.s_m2(outB1)
        outB3,mask3 = self.s_m3(outB2)
        outB4,mask4 = self.s_m4(outB3)
        outB5,mask5 = self.s_m5(outB4)
        outB6,mask6 = self.s_m21(outB5)
        outB7,mask7 = self.s_m31(outB6)
        outB8,mask8 = self.s_m41(outB7)
        outB5,mask5 = self.s_m51(outB8)

        outB5 = self.up1(outB5,c1)
        outB6,mask6 = self.s_m6(outB5)
        outB6 = self.up2(outB6,c0)
        outB7,mask7 = self.s_m7(outB6)
        # Mid-supervise
        B_coarse = self.outconv_B_coarse(outB7)

        out_R    = self.r_branch(feat_R)
        R        = self.outconv_R(out_R)
        # Get feat from B_coarse and R
        feat_fromR  = self.head_feat(R)
        feat_BC     = self.head_feat(B_coarse)
        # feat weight and concat
        positive_feat  = self.Channel_weight(feat_B)
        negative_feat  = self.Channel_weight(feat_fromR,state=-1)
        feat2refine = torch.cat([feat_BC,positive_feat,negative_feat],dim=1)

        # Final B
        B,mask_fine        = self.refine(feat2refine)
        B = B+I
        mask = [mask0,mask1,mask2,mask3,mask4,mask5,mask6,mask7]
        mask.extend(mask_fine)
        return R,B_coarse,B,mask_B,mask,mask

class Transit_mask_B(nn.Module):
    def __init__(self, in_c=128, out_c=64):
        super(Transit_mask_B,self).__init__()
        self.conv = nn.Sequential(
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(in_c,out_c,3,1,padding=0),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(out_c,out_c,3,1,padding=0),
                nn.ReLU(inplace=True),
            )
        self.mask = nn.Sequential(
                #nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(out_c,64,1,1,padding=0,bias=False),
                nn.Tanh(),
                nn.Conv2d(64,1,1,1,padding=0,bias=False),
                nn.Sigmoid()
            )

    def forward(self,input):
        out = self.conv(input)
        mask = self.mask(out)
        return mask

class Seperation_module_B(nn.Module):
    def __init__(self, in_c, out_c, print_mask=True, downsample=False,dilation=1):  
        super(Seperation_module_B,self).__init__()
        self.mask_1 = Mask_learning(in_c)

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d((dilation,dilation,dilation,dilation)),
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=0, bias=False,dilation=dilation),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((dilation,dilation,dilation,dilation)),
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=0, bias=False,dilation=dilation),
            nn.ReLU(inplace=True),
            )
        self.conv3 = Padding_Conv(out_c,out_c*2,stride=2)

        self.print_mask = print_mask

    def forward(self,B):
        # 2 convs 3x3
        feat_B = self.conv1(B)
        # learn mask 
        mask1 = self.mask_1(feat_B)
        # norm 2 masks
        #mask1 = (mask1+alpha*mask)/(1+alpha)

        # apply mask
        out_B = mask1*feat_B + B
        return out_B,mask1.data

class Refined(nn.Module):
    def __init__(self):
        super(Refined, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(192,64, kernel_size=3, padding=0, dilation=1),
            nn.ReLU(inplace=True)
            )

        self.s_m0 = Seperation_module_B(64,64,dilation=1)
        self.s_m1 = Seperation_module_B(64,64,dilation=2)
        self.s_m2 = Seperation_module_B(64,64,dilation=4)
        self.s_m3 = Seperation_module_B(64,64,dilation=8)
        self.s_m4 = Seperation_module_B(64,64,dilation=16)
        self.s_m5 = Seperation_module_B(64,64,dilation=1)

        self.outconv = nn.Sequential(
            nn.Conv2d(64,64, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, padding=0, dilation=1),
            #nn.Tanh()
            )

    def forward(self, x):
        x = self.conv(x)
        a = 0.4
        outB0,mask0 = self.s_m0(x)
        outB1,mask1 = self.s_m1(outB0)
        outB2,mask2 = self.s_m2(outB1)
        outB3,mask3 = self.s_m3(outB2)
        outB4,mask4 = self.s_m4(outB3)
        outB5,mask5 = self.s_m5(outB4)
        outB        = self.outconv(outB5) 
        mask = [mask0,mask1,mask2,mask3,mask4,mask5]
        return outB,mask

        '''
        # CA module
        self.conv = nn.Sequential(
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(192,64, kernel_size=3, padding=0, dilation=1),
            nn.ReLU(inplace=True),
            SE_ResBlock(64,64,kernel_size=256//4),
            SE_ResBlock(64,64,kernel_size=256//4),
            SE_ResBlock(64,64,kernel_size=256//4),
            SE_ResBlock(64,64,kernel_size=256//4),
            nn.Conv2d(64, 3, kernel_size=1, padding=0, dilation=1),
            nn.Tanh()
            )
        '''

    
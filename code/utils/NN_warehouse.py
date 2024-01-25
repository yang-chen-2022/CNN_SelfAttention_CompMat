import numpy as np
import torch
import torch.nn as nn
from utils.IOfcts import *

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)




# ===========================================================
# neural network for field prediction at each individual step
# ===========================================================

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key   = self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        # attention = self.softmax(torch.bmm(proj_query,proj_key)) # BX (N) X (N) 
        out = torch.bmm(proj_value, self.softmax(torch.bmm(proj_query,proj_key)).permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out



class mesh_encoder_SA(nn.Module):
    def __init__(self, in_channels=1, out_channels=128):
        super(mesh_encoder_SA, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=1, padding=1, padding_mode="circular", bias=True) # 251 -> 251
        self.bn1 = nn.BatchNorm2d(32)
        self.attn1 = Self_Attn(32)
        self.conv2 = nn.Conv2d(32, 48, 5, stride=1, padding=2, padding_mode="circular", bias=True) # 251 -> 251
        self.bn2 = nn.BatchNorm2d(48)
        self.attn2 = Self_Attn(48)
        self.conv3 = nn.Conv2d(48, 64, 5, stride=2, padding=2, padding_mode="circular", bias=True) # 251 -> 125
        self.bn3 = nn.BatchNorm2d(64)
        self.attn3 = Self_Attn(64)
        self.conv4 = nn.Conv2d(64, 96, 3, stride=2, padding=1, padding_mode="circular", bias=True) # 125 -> 63
        self.bn4 = nn.BatchNorm2d(96)
        self.attn4 = Self_Attn(96)
        self.conv5 = nn.Conv2d(96, out_channels, 3, stride=2, padding=1, padding_mode="circular", bias=False) # 63 -> 32
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.attn5 = Self_Attn(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.attn5(self.bn5(self.conv5(x))))

        return x


class field_decoder(nn.Module):
    def __init__(self, in_channels=128, out_channels=1):
        super(field_decoder, self).__init__()

        self.t_conv5 = nn.ConvTranspose2d(in_channels, 96, 3, stride=2, padding=1)  # 32 -> 63
        self.t_conv4 = nn.ConvTranspose2d(96, 64, 3, stride=2, padding=1) # 63 -> 125
        self.t_conv3 = nn.ConvTranspose2d(64, 48, 5, stride=2, padding=1) # 125 -> 251
        self.t_conv2 = nn.ConvTranspose2d(48, 32, 3, stride=1, padding=1) # 251 -> 251
        self.t_conv1 = nn.ConvTranspose2d(32, 6, 3, stride=1, padding=1) # 251 -> 251
        self.t_conv0 = conv1x1(6, out_channels, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(48)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(96)
        self.bn5 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn4(self.t_conv5(x)))
        x = self.relu(self.bn3(self.t_conv4(x)))
        x = self.relu(self.bn2(self.t_conv3(x)))
        x = self.relu(self.bn1(self.t_conv2(x)))
        x = self.relu(self.t_conv1(x))
        x = self.t_conv0(x)

        return x

        

       
class NN_mesh2stress3_SA_2(nn.Module):
    def __init__(self):
        super(NN_mesh2stress3_SA_2, self).__init__()

        self.encoder = mesh_encoder_SA(in_channels=1)
        self.decoder = field_decoder(out_channels=3)

    def forward(self, mesh):

        return self.decoder(self.encoder(mesh))

        


class NN_elas2crk_SA(nn.Module):
    def __init__(self):
        super(NN_elas2crk_SA, self).__init__()

        self.encoder = mesh_encoder_SA(in_channels=1)
        self.decoder = field_decoder(out_channels=1)

    def forward(self, mesh):

        return self.decoder(self.encoder(mesh))

        

   





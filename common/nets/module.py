import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import densenet121, densenet169, densenet201, densenet161
from torchvision.models import DenseNet
from config import cfg
from nets.layer import make_conv_layers, make_deconv_layers, make_conv1d_layers, make_linear_layers
from nets.SemGCN.export import SemGCN

class PoseNet(nn.Module):
    def __init__(self, joint_num):
        super(PoseNet, self).__init__()
        self.joint_num = joint_num
        self.deconv = make_deconv_layers([2048,256,256,256])
        self.conv_x = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        
        if (cfg.stage =='sem_gcn' ):
            self.feat_conv = make_conv_layers((2048,512,128,joint_num), bnrelu_final = True )
            # shape is (N, 17,64)
            self.sem_gcn =  SemGCN(cfg.skeleton, coords_dim = (64,1) )
        else:
            self.conv_z_1 = make_conv1d_layers([2048,256*cfg.output_hm_shape[0]], kernel=1, stride=1, padding=0)
            self.conv_z_2 = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        coord = heatmap1d * torch.arange(heatmap_size).float().cuda()
        coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, img_feat):
        img_feat_xy = self.deconv(img_feat)

        # x axis
        img_feat_x = img_feat_xy.mean((2))
        heatmap_x = self.conv_x(img_feat_x)
        coord_x = self.soft_argmax_1d(heatmap_x)
        
        # y axis
        img_feat_y = img_feat_xy.mean((3))
        heatmap_y = self.conv_y(img_feat_y)
        coord_y = self.soft_argmax_1d(heatmap_y)
        
        if (cfg.stage == 'sem_gcn'):
            #print("img_feat shape is ", img_feat.shape)
            # [N,2048,8,8]
            joint_feat = self.feat_conv(img_feat)
            #print("after conv become", joint_feat.shape)
            joint_feat = joint_feat.flatten(start_dim = 2)
            coord_z = self.sem_gcn(joint_feat)
        else:
            # z axis
            img_feat_z = img_feat.mean((2,3))[:,:,None]
            img_feat_z = self.conv_z_1(img_feat_z)
            img_feat_z = img_feat_z.view(-1,256,cfg.output_hm_shape[0])
            heatmap_z = self.conv_z_2(img_feat_z)
            coord_z = self.soft_argmax_1d(heatmap_z)
    
        joint_coord = torch.cat((coord_x, coord_y, coord_z),2)
        return joint_coord
    

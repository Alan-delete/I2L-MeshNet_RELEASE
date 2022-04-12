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
            # 8 * 8 too small ?
            self.feat_conv = make_conv_layers((2048,512,256,joint_num), bnrelu_final = True )
            # shape is (N, 17,64)
            #heat map 
            self.sem_gcn = SemGCN(cfg.skeleton, coords_dim = (64+2,64), nodes_group = cfg.skeleton if cfg.non_local else None )

        elif (cfg.stage == 'lixel'):
            self.conv_z_1 = make_conv1d_layers([2048,256*cfg.output_hm_shape[0]], kernel=1, stride=1, padding=0)
            self.conv_z_2 = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

        else:
            # after flatten the size is 32*32 , or mean the size is 32
            self.deconv_z = make_deconv_layers([2048,256,self.joint_num])
            #self.feat_conv = make_conv_layers((2048,512,256,joint_num), bnrelu_final = True )
            self.assess = torch.nn.Sequential(torch.nn.Linear(32*32,1) ,torch.nn.Sigmoid())
            self.sem_gcn1 = SemGCN(cfg.skeleton, coords_dim = (2,1), nodes_group = cfg.skeleton if cfg.non_local else None )
            
            self.sem_gcn2 = SemGCN(cfg.skeleton, coords_dim = (32*32,64), nodes_group = cfg.skeleton if cfg.non_local else None )
 
    def sem_gcn_init(self):
        # load pretrained sem_gcn1
        ckpt_path = os.path.join(cfg.model_dir,'sem_gcn_epoch{}'.format(6))
        ckpt = torch.load(ckpt_path)
        self.sem_gcn1.load_state_dict(ckpt['network'])


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
            joint_feat = torch.cat((coord_x, coord_y,joint_feat), dim = 2) 
            #coord_z = self.sem_gcn(joint_feat)
            
            #heatmap form
            heatmap_z = self.sem_gcn(joint_feat)
            coord_z = self.soft_argmax_1d(heatmap_z)
        elif (cfg.stage == 'lixel'):
            # z axis
            img_feat_z = img_feat.mean((2,3))[:,:,None]
            img_feat_z = self.conv_z_1(img_feat_z)
            img_feat_z = img_feat_z.view(-1,256,cfg.output_hm_shape[0])
            heatmap_z = self.conv_z_2(img_feat_z)
            coord_z = self.soft_argmax_1d(heatmap_z)
        # multi-branch
        else:
            with torch.no_grad():
                xy_coord = torch.cat((coord_x, coord_y), 2)
                coord_z1 = self.sem_gcn1(xy_coord)            
#            joint_feat = self.deconv_z(img_feat)
#            joint_feat = joint_feat.flatten(start_dim = 2)
#            #heatmap form
#            heatmap_z = self.sem_gcn2(joint_feat)
#            coord_z2 = self.soft_argmax_1d(heatmap_z)
#            # C is of size N, 17
#            C = self.assess(joint_feat)
#            # make it betweem [0,1]
#            C = 1.0 / ( 1 - torch.exp(-C) )
#            coord_z = ( 1 - C ) * coord_z1 + C * coord_z2 
            coord_z = coord_z1

        joint_coord = torch.cat((coord_x, coord_y, coord_z),2)
        return joint_coord
    

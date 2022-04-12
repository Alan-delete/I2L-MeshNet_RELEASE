import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms 
from nets.resnet import ResNetBackbone
from nets.densenet import DenseNetBackbone
from nets.module import PoseNet
from nets.SemGCN.export import SemGCN
from nets.loss import CoordLoss, BoneVectorLoss, EdgeLengthLoss
from config import cfg
from nets.layer import make_conv_layers, make_deconv_layers, make_conv1d_layers
#from contextlib import nullcontext
import math

class Model_2Dpose(nn.Module):
    def __init__(self, pose_backbone, pose_net):
        super(Model_2Dpose, self).__init__()
        self.pose_backbone = pose_backbone
        self.pose_net = pose_net
        
        self.coord_loss = CoordLoss()
        self.bone_vec_loss = BoneVectorLoss()
        self.trainable_modules = [self.pose_backbone, self.pose_net]
        


    def make_gaussian_heatmap(self, joint_coord_img):
        x = torch.arange(cfg.output_hm_shape[2])
        y = torch.arange(cfg.output_hm_shape[1])
        z = torch.arange(cfg.output_hm_shape[0])
        zz,yy,xx = torch.meshgrid(z,y,x)
        xx = xx[None,None,:,:,:].cuda().float(); yy = yy[None,None,:,:,:].cuda().float(); zz = zz[None,None,:,:,:].cuda().float();
        
        x = joint_coord_img[:,:,0,None,None,None]; y = joint_coord_img[:,:,1,None,None,None]; z = joint_coord_img[:,:,2,None,None,None];
        heatmap = torch.exp(-(((xx-x)/cfg.sigma)**2)/2 -(((yy-y)/cfg.sigma)**2)/2 - (((zz-z)/cfg.sigma)**2)/2)
        return heatmap
    
    def forward(self, inputs, targets, meta_info, mode = 'test'):
        #self.pose_backbone.eval()
        with torch.no_grad():
            shared_img_feat, pose_img_feat = self.pose_backbone(inputs['img'])
        #shared_img_feat, pose_img_feat = self.pose_backbone(inputs['img'])
        # shape of image feature of resnet is [N, 2048, 13, 20]
        #self.pose_net.eval()
        joint_coord_img = self.pose_net(pose_img_feat)
        
        if (mode=='train'):
            loss = {}
            loss['joint_orig'] = self.coord_loss(joint_coord_img, targets['orig_joint_img'], valid = meta_info['orig_joint_trunc'] ,is_3D = meta_info['is_3D'])
            #loss['bone_vector'] = self.bone_vec_loss(joint_coord_img, targets['orig_joint_img'])
#            for k,v in loss.items():
#                if torch.any(torch.isnan(v)):
#                    print("current img feat is:", pose_img_feat)
#                    print("current deconv feat is:", self.pose_net.deconv(pose_img_feat))
#                   
#                    print("prediction is:",joint_coord_img ) 
#                    print("resnet NAN layers are")
#                    for name, p in self.pose_backbone.named_parameters():
#                        if (torch.any(torch.isnan(p))):
#                            print(name)
#                   
#                    print("posenet NAN layers are")
#                    for name, p in self.pose_net.named_parameters():
#                        if (torch.any(torch.isnan(p))):
#                            print(name)
#                     
#                    assert False, 'loss is nan here'
#
            return loss
        else:
            out = {}
            out['joint_coord_img'] = joint_coord_img
            out['bb2img_trans'] = meta_info['bb2img_trans']
            out['img_feat'] = pose_img_feat 
            return out

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        #nn.init.normal_(m.weight,std=0.001)
        nn.init.kaiming_normal_(m.weight,mode = 'fan_out', nonlinearity = 'relu')
    elif type(m) == nn.Conv2d:
        #nn.init.normal_(m.weight,std=0.001)
        nn.init.kaiming_normal_(m.weight,mode = 'fan_out', nonlinearity = 'relu')
        if m.bias != None:
            nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(joint_num, mode):
    pose_backbone = None
    if ('densenet' in cfg.backbone_type):
        pose_backbone = DenseNetBackbone(cfg.densenet_type)
    else:
        pose_backbone = ResNetBackbone(cfg.resnet_type)
    
    pose_net = PoseNet(joint_num)

    if mode == 'train':
        # we load pretained backbone instead of the online one
        backbone_ckpt = torch.load(os.path.join(cfg.model_dir, 'backbone.pth.tar'))
        pose_backbone.load_state_dict(backbone_ckpt['network'])
        #pose_backbone.init_weights()
        pose_net.apply(init_weights)
        # load fixed sub branch
        if (cfg.stage=='hybrid'):
            pose_net.sem_gcn_init()

    model = Model_2Dpose(pose_backbone, pose_net)
    return model


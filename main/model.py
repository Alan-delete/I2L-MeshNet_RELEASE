import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from nets.module import PoseNet, Pose2Feat, MeshNet, ParamRegressor
from nets.loss import CoordLoss, ParamLoss, NormalVectorLoss, EdgeLengthLoss
from utils.smpl import SMPL
from utils.mano import MANO
from config import cfg
from contextlib import nullcontext
import math

class Model_2Dpose(nn.Module):
    def __init__(self, pose_backbone, pose_net):
        super(Model_2Dpose, self).__init__()
        self.pose_backbone = pose_backbone
        self.pose_net = pose_net
        
        #if 'FreiHAND' in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
        #    self.human_model = MANO()
        #    self.human_model_layer = self.human_model.layer.cuda()
        #else:
        #    self.human_model = SMPL()
        #    self.human_model_layer = self.human_model.layer['neutral'].cuda()
        #self.root_joint_idx = self.human_model.root_joint_idx
        #self.mesh_face = self.human_model.face
        #self.joint_regressor = self.human_model.joint_regressor

        self.coord_loss = CoordLoss()
        

   
    def make_gaussian_heatmap(self, joint_coord_img):
        x = torch.arange(cfg.output_hm_shape[2])
        y = torch.arange(cfg.output_hm_shape[1])
        z = torch.arange(cfg.output_hm_shape[0])
        zz,yy,xx = torch.meshgrid(z,y,x)
        xx = xx[None,None,:,:,:].cuda().float(); yy = yy[None,None,:,:,:].cuda().float(); zz = zz[None,None,:,:,:].cuda().float();
        
        x = joint_coord_img[:,:,0,None,None,None]; y = joint_coord_img[:,:,1,None,None,None]; z = joint_coord_img[:,:,2,None,None,None];
        heatmap = torch.exp(-(((xx-x)/cfg.sigma)**2)/2 -(((yy-y)/cfg.sigma)**2)/2 - (((zz-z)/cfg.sigma)**2)/2)
        return heatmap
    
    def forward(self, inputs, targets, meta_info, mode):

        if cfg.stage == 'lixel':
            cm = nullcontext()
        else:
            cm = torch.no_grad()
        
        with cm:
            # posenet forward
            shared_img_feat, pose_img_feat = self.pose_backbone(inputs['img'])
            joint_coord_img = self.pose_net(pose_img_feat)
            
            # make 3D heatmap from posenet output and convert to image feature
            with torch.no_grad():
                joint_heatmap = self.make_gaussian_heatmap(joint_coord_img.detach())


        if mode == 'train':
            # loss functions
            loss = {}
            if cfg.stage == 'lixel':
                loss['joint_fit'] = self.coord_loss(joint_coord_img, targets['fit_joint_img'], meta_info['fit_joint_trunc'] * meta_info['is_valid_fit'][:,None,None])
                loss['joint_orig'] = self.coord_loss(joint_coord_img, targets['orig_joint_img'], meta_info['orig_joint_trunc'], meta_info['is_3D'])
            return loss
        
        else:
            # test output
            out = {}
            out['joint_coord_img'] = joint_coord_img
            out['bb2img_trans'] = meta_info['bb2img_trans']

            return out

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(vertex_num, joint_num, mode):
    # resnet has output feature number 2048
    feat_num = 2048
    # check whether to use densenet instead of resnet

    if ('densenet' in cfg.backbone_type):
        pose_backbone = 

    pose_backbone = ResNetBackbone(cfg.resnet_type)
    pose_net = PoseNet(feat_num, joint_num)

    if mode == 'train':
        pose_backbone.init_weights()
        pose_net.apply(init_weights)

    model = Model_2Dpose(pose_backbone, pose_net)
    return model


import os
import os.path as osp
import sys
import numpy as np

class Config:
    
    ## dataset
    # MuCo, Human36M, MSCOCO, PW3D, FreiHAND
    trainset_3d = ['Human36M']#MuCo, Human36M, FreiHAND
    trainset_2d = ['MSCOCO'] # MSCOCO
    testset = 'Human36M'# Human36M, MSCOCO, PW3D, FreiHAND

    ## model setting
    backbone_type = 'resnet50'
    resnet_type = 50 # 50, 101, 152
    densenet_type = 121 # 169, 201, 161
    
    ## input, output
    input_img_shape = (256, 256) 
    output_hm_shape = (64, 64, 64)
    bbox_3d_size = 2 if 'FreiHAND' not in trainset_3d + trainset_2d + [testset] else 0.3
    sigma = 2.5
    ## set joints information
    joints_name = ('Pelvis', 'R_Hip', 'R_Knee','R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso','Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow','L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
    joint_num = 17
    skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10),(8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15,16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
    root_joint_idx = 0

    ## training config
    lr_dec_epoch = [10,12] if 'FreiHAND' not in trainset_3d + trainset_2d + [testset] else [17,21]
    end_epoch = 13 if 'FreiHAND' not in trainset_3d + trainset_2d + [testset] else 25
    lr = 1e-4
    lr_dec_factor = 10
    train_batch_size = 8
    normal_loss_weight = 0.1

    ## testing config
    test_batch_size = 8
    use_gt_info = False

    ## others
    num_thread = 10
    gpu_ids = '0'
    num_gpus = 1
    stage = '2D' # 2D ,3D
    continue_train = False
    
    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    #model_dir = osp.join(output_dir, 'model_dump')
    model_dir = osp.join(root_dir, 'weights')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    #mano_path = osp.join(root_dir, 'common', 'utils', 'manopth')
    #smpl_path = osp.join(root_dir, 'common', 'utils', 'smplpytorch')
    
    def set_args(self, gpu_ids, stage='2D', continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.stage = stage
        # extend training schedule
        if self.stage == 'param':
            self.lr_dec_epoch = [x+5 for x in self.lr_dec_epoch]
            self.end_epoch = self.end_epoch + 5
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))
        
        if self.testset == 'FreiHAND':
            assert self.trainset_3d[0] == 'FreiHAND'
            assert len(self.trainset_3d) == 1
            assert len(self.trainset_2d) == 0

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
for i in range(len(cfg.trainset_3d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d[i]))
for i in range(len(cfg.trainset_2d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_2d[i]))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)

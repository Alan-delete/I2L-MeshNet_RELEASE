import os
import os.path as osp
import numpy as np
from config import cfg
import copy
import json
import scipy.io as sio
import cv2
import random
import math
import torch
#import transforms3d
from pycocotools.coco import COCO
#from utils.smpl import SMPL
from utils.preprocessing import load_img,process_bbox, augmentation,root_joint_normalize
#from utils.vis import vis_keypoints, vis_mesh, save_obj
from utils.transforms import world2cam, cam2pixel, pixel2cam, transform_joint_to_other_db

class MSCOCO(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = 'train' if data_split == 'train' else 'val'
        self.img_path = osp.join('..','data','MSCOCO', 'images')
        self.annot_path = osp.join('..', 'data', 'MSCOCO', 'annotations')
        self.rootnet_output_path = osp.join('..', 'data', 'MSCOCO', 'rootnet_output', 'bbox_root_coco_output.json')
        self.fitting_thr = 3.0 # pixel in cfg.output_hm_shape space

        # mscoco skeleton
        self.coco_joint_num = 18 # original: 17, manually added pelvis
        self.coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis')
        self.coco_skeleton = ( (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12) )
        self.coco_flip_pairs = ( (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16) )
        self.coco_joint_regressor = np.load(osp.join('..', 'data', 'MSCOCO', 'J_regressor_coco_hip_smpl.npy'))
        if (data_split == 'test'):
            self.joint_num = cfg.smpl_joint_num
            self.joints_name = cfg.smpl_joints_name
            self.skeleton = cfg.smpl_skeleton
        else:
            self.joint_num = cfg.joint_num
            self.joints_name = cfg.joints_name
            self.skeleton = cfg.skeleton
        self.root_joint_idx = self.joints_name.index('Pelvis') 
        self.datalist = self.load_data()

    def add_pelvis(self, joint_coord):
        lhip_idx = self.coco_joints_name.index('L_Hip')
        rhip_idx = self.coco_joints_name.index('R_Hip')
        pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
        pelvis[2] = joint_coord[lhip_idx,2] * joint_coord[rhip_idx,2] # joint_valid
        pelvis = pelvis.reshape(1, 3)
        joint_coord = np.concatenate((joint_coord, pelvis))
        return joint_coord

    def load_data(self):
        db = COCO(osp.join(self.annot_path, 'person_keypoints_' + self.data_split + '2017.json'))
        

        datalist = []
        if self.data_split == 'train':
            for aid in db.anns.keys():
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]
                imgname = osp.join('train2017', img['file_name'])
                img_path = osp.join(self.img_path, imgname)
                width, height = img['width'], img['height']
                
                if ann['iscrowd'] or (ann['num_keypoints'] == 0):
                    continue
                
                # bbox
                bbox = process_bbox(ann['bbox'], width, height) 
                if bbox is None: continue
                
                # joint coordinates
                joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)
                joint_img = self.add_pelvis(joint_img)
                joint_valid = (joint_img[:,2].copy().reshape(-1,1) > 0).astype(np.float32)
                joint_img[:,2] = 0

                datalist.append({
                    'img_path': img_path,
                    'img_shape': (height, width),
                    'bbox': bbox,
                    'joint_img': joint_img,
                    'joint_valid': joint_valid,
                })

        else:
            with open(self.rootnet_output_path) as f:
                rootnet_output = json.load(f)
            print('Load RootNet output from  ' + self.rootnet_output_path)
            for i in range(len(rootnet_output)):
                image_id = rootnet_output[i]['image_id']
                if image_id not in db.imgs:
                    continue
                img = db.loadImgs(image_id)[0]
                imgname = osp.join('val2017', img['file_name'])
                img_path = osp.join(self.img_path, imgname)
                height, width = img['height'], img['width']

                fx, fy, cx, cy = 1500, 1500, img['width']/2, img['height']/2
                focal = np.array([fx, fy], dtype=np.float32); princpt = np.array([cx, cy], dtype=np.float32);
                root_joint_depth = np.array(rootnet_output[i]['root_cam'][2])
                bbox = np.array(rootnet_output[i]['bbox']).reshape(4)
                cam_param = {'focal': focal, 'princpt': princpt}

                datalist.append({
                    'img_path': img_path,
                    'img_shape': (height, width),
                    'bbox': bbox,
                    'root_joint_depth': root_joint_depth,
                    'cam_param': cam_param
                })

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']
       
        # image load and affine transform
        img = load_img(img_path)
        # transforms include 0.25 scale, 30
        # degree rotate, 0.2 color factor, flip in 0.5
        # probability (this could be disabled by
        # exclude_flip= True)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split,exclude_flip = True)
        img = self.transform(img.astype(np.float32))/255.
        
        if self.data_split == 'train':
            # coco gt
            coco_joint_img = data['joint_img']
            coco_joint_valid = data['joint_valid']
            #print('here test coco_joint', coco_joint_img)
            #print('here test cooc_valid',coco_joint_valid)
            if do_flip:
                coco_joint_img[:,0] = img_shape[1] - 1 - coco_joint_img[:,0]
                for pair in self.coco_flip_pairs:
                    coco_joint_img[pair[0],:], coco_joint_img[pair[1],:] = coco_joint_img[pair[1],:].copy(), coco_joint_img[pair[0],:].copy()
                    coco_joint_valid[pair[0],:], coco_joint_valid[pair[1],:] = coco_joint_valid[pair[1],:].copy(), coco_joint_valid[pair[0],:].copy()

            coco_joint_img_xy1 = np.concatenate((coco_joint_img[:,:2], np.ones_like(coco_joint_img[:,:1])),1)
            coco_joint_img[:,:2] = np.dot(img2bb_trans, coco_joint_img_xy1.transpose(1,0)).transpose(1,0)
            coco_joint_img[:,0] = coco_joint_img[:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            coco_joint_img[:,1] = coco_joint_img[:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
            
            # backup for calculating fitting error
            _coco_joint_img = coco_joint_img.copy()
            _coco_joint_valid = coco_joint_valid.copy()
            
            # check truncation
            coco_joint_trunc = coco_joint_valid * ((coco_joint_img[:,0] >= 0) * (coco_joint_img[:,0] < cfg.output_hm_shape[2]) * \
                        (coco_joint_img[:,1] >= 0) * (coco_joint_img[:,1] < cfg.output_hm_shape[1])).reshape(-1,1).astype(np.float32)

            # transform coco joints to target db joints
            coco_joint_img = transform_joint_to_other_db(coco_joint_img, self.coco_joints_name, self.joints_name)
            coco_joint_cam = np.zeros((self.joint_num,3), dtype=np.float32) # dummy
            
            
            # create root-relative and normalized
            # joints coordinates
            coco_joint_root = root_joint_normalize(coco_joint_img,self.joints_name)

            
            coco_joint_valid = transform_joint_to_other_db(coco_joint_valid, self.coco_joints_name, self.joints_name)
            coco_joint_trunc = transform_joint_to_other_db(coco_joint_trunc, self.coco_joints_name, self.joints_name)
            
            # 3D data rotation augmentation
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1]], dtype=np.float32)
            inputs = {'img': img}
            targets ={'orig_joint_img':coco_joint_img,'orig_joint_cam':coco_joint_cam,'normalize_joint_root':coco_joint_root}
            meta_info = {'orig_joint_valid': coco_joint_valid, 'orig_joint_trunc': coco_joint_trunc,  'is_3D': float(False)}
            return inputs, targets, meta_info
        else:
            inputs = {'img': img}
            targets = {}
            meta_info = {'bb2img_trans': bb2img_trans}
            return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):

        annots = self.datalist
        sample_num = len(outs)
        eval_result = {}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            
            # x,y: resize to input image space and perform bbox to image affine transform
            bb2img_trans = out['bb2img_trans']
            mesh_out_img = out['mesh_coord_img']
            mesh_out_img[:,0] = mesh_out_img[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            mesh_out_img[:,1] = mesh_out_img[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            mesh_out_img_xy1 = np.concatenate((mesh_out_img[:,:2], np.ones_like(mesh_out_img[:,:1])),1)
            mesh_out_img[:,:2] = np.dot(bb2img_trans, mesh_out_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]

            # z: devoxelize and translate to absolute depth
            root_joint_depth = annot['root_joint_depth']
            mesh_out_img[:,2] = (mesh_out_img[:,2] / cfg.output_hm_shape[0] * 2. - 1) * (cfg.bbox_3d_size * 1000 / 2) # change cfg.bbox_3d_size from meter to milimeter
            mesh_out_img[:,2] = mesh_out_img[:,2] + root_joint_depth

            # camera back-projection
            cam_param = annot['cam_param']
            focal, princpt = cam_param['focal'], cam_param['princpt']
            mesh_out_cam = pixel2cam(mesh_out_img, focal, princpt)

            if cfg.stage == 'param':
                mesh_out_cam = out['mesh_coord_cam']

            vis = False
            if vis:
                filename = annot['img_path'].split('/')[-1][:-4] + '_' + str(n)

                img = load_img(annot['img_path'])[:,:,::-1]
                cv2.imwrite(filename + '.jpg', img)

                save_obj(mesh_out_cam, self.smpl.face, filename + '.obj')

        return eval_result

    def print_eval_result(self, eval_result):
        pass 

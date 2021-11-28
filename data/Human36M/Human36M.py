import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import math
import copy
import transforms3d
from pycocotools.coco import COCO
from config import cfg
#from utils.smpl import SMPL
from utils.preprocessing import load_img,get_bbox, process_bbox, generate_patch_image,augmentation,root_joint_normalize
from utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db
#from utils.vis import vis_keypoints, vis_mesh, save_obj

class Human36M(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_dir = osp.join('..', 'data', 'Human36M', 'images')
        self.annot_path = osp.join('..', 'data', 'Human36M', 'annotations')
        self.human_bbox_root_dir = osp.join('..', 'data', 'Human36M', 'rootnet_output', 'bbox_root_human36m_output.json')
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
        self.fitting_thr = 25 # milimeter

        # H36M joint set
        self.h36m_joint_num = 17
        self.h36m_joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.h36m_flip_pairs = ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) )
        self.h36m_skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
        self.h36m_root_joint_idx = self.h36m_joints_name.index('Pelvis')
        self.h36m_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.h36m_joint_regressor = np.load(osp.join('..', 'data', 'Human36M', 'J_regressor_h36m_correct.npy'))

        # use the pretrained model with joint num as
        # 29
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

    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5
        elif self.data_split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            subject = [1,5,6,7,8]
        elif self.data_split == 'test':
            subject = [9,11]
        else:
            assert 0, print("Unknown subset")

        return subject
    
    def load_data(self):
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()
        
        # aggregate annotations from each subject
        db = COCO()
        cameras = {}
        joints = {}
        for subject in subject_list:
            # data load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'),'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'),'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'),'r') as f:
                joints[str(subject)] = json.load(f)
#            # smpl parameter load
#            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_smpl_param.json'),'r') as f:
#                smpl_params[str(subject)] = json.load(f)
        db.createIndex()


        if self.data_split == 'test' and not cfg.use_gt_info:
            print("Get bounding box and root from " + self.human_bbox_root_dir)
            bbox_root_result = {}
            with open(self.human_bbox_root_dir) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                bbox_root_result[str(annot[i]['image_id'])] = {'bbox': np.array(annot[i]['bbox']), 'root': np.array(annot[i]['root_cam'])}
        else:
            print("Get bounding box and root from groundtruth")

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            img_shape = (img['height'], img['width'])
           
            # check subject and frame_idx
            frame_idx = img['frame_idx'];
            if frame_idx % sampling_ratio != 0:
                continue

            # check smpl parameter exist
            subject = img['subject']; action_idx = img['action_idx']; subaction_idx = img['subaction_idx']; frame_idx = img['frame_idx'];
#            try:
#                smpl_param = smpl_params[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)]
#            except KeyError:
#                smpl_param = None
#
            # camera parameter
            cam_idx = img['cam_idx']
            cam_param = cameras[str(subject)][str(cam_idx)]
            R,t,f,c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
            cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}
            
            # only use frontal camera following previous works (HMR and SPIN)
            if self.data_split == 'test' and str(cam_idx) != '4':
                continue
                
            # project world coordinate to cam, image coordinate space
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)
            joint_cam = world2cam(joint_world, R, t)
            joint_img = cam2pixel(joint_cam, f, c)
            joint_valid = np.ones((self.h36m_joint_num,1))
        
            if self.data_split == 'test' and not cfg.use_gt_info:
                bbox = bbox_root_result[str(image_id)]['bbox'] # bbox should be aspect ratio preserved-extended. It is done in RootNet.
                root_joint_depth = bbox_root_result[str(image_id)]['root'][2]
            else:
                bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
                if bbox is None: continue
                root_joint_depth = joint_cam[self.h36m_root_joint_idx][2]
    
            datalist.append({
                'img_path': img_path,
                'img_id': image_id,
                'img_shape': img_shape,
                'bbox': bbox,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid,
#                'smpl_param': smpl_param,
                'root_joint_depth': root_joint_depth,
                'cam_param': cam_param})
            
        return datalist
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox,cam_param = data['img_path'], data['img_shape'], data['bbox'], data['cam_param']
         
        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.
        
        if self.data_split == 'train':
            # h36m gt
            h36m_joint_img = data['joint_img']
            h36m_joint_cam = data['joint_cam']
            h36m_joint_cam = h36m_joint_cam - h36m_joint_cam[self.h36m_root_joint_idx,None,:] # root-relative
            h36m_joint_valid = data['joint_valid']
            if do_flip:
                h36m_joint_cam[:,0] = -h36m_joint_cam[:,0]
                h36m_joint_img[:,0] = img_shape[1] - 1 - h36m_joint_img[:,0]
                for pair in self.h36m_flip_pairs:
                    h36m_joint_img[pair[0],:], h36m_joint_img[pair[1],:] = h36m_joint_img[pair[1],:].copy(), h36m_joint_img[pair[0],:].copy()
                    h36m_joint_cam[pair[0],:], h36m_joint_cam[pair[1],:] = h36m_joint_cam[pair[1],:].copy(), h36m_joint_cam[pair[0],:].copy()
                    h36m_joint_valid[pair[0],:], h36m_joint_valid[pair[1],:] = h36m_joint_valid[pair[1],:].copy(), h36m_joint_valid[pair[0],:].copy()

            h36m_joint_img_xy1 = np.concatenate((h36m_joint_img[:,:2], np.ones_like(h36m_joint_img[:,:1])),1)
            h36m_joint_img[:,:2] = np.dot(img2bb_trans, h36m_joint_img_xy1.transpose(1,0)).transpose(1,0)
            h36m_joint_img[:,0] = h36m_joint_img[:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            h36m_joint_img[:,1] = h36m_joint_img[:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
            h36m_joint_img[:,2] = h36m_joint_img[:,2] - h36m_joint_img[self.h36m_root_joint_idx][2] # root-relative
            h36m_joint_img[:,2] = (h36m_joint_img[:,2] / (cfg.bbox_3d_size * 1000  / 2) + 1)/2. * cfg.output_hm_shape[0] # change cfg.bbox_3d_size from meter to milimeter
            
            # check truncation
            h36m_joint_trunc = h36m_joint_valid * ((h36m_joint_img[:,0] >= 0) * (h36m_joint_img[:,0] < cfg.output_hm_shape[2]) * \
                        (h36m_joint_img[:,1] >= 0) * (h36m_joint_img[:,1] < cfg.output_hm_shape[1]) * \
                        (h36m_joint_img[:,2] >= 0) * (h36m_joint_img[:,2] < cfg.output_hm_shape[0])).reshape(-1,1).astype(np.float32)

            # transform h36m joints to target db joints
            # since we use Human3.6 as target db, no
            # need to implement transform anymore
            h36m_joint_img = transform_joint_to_other_db(h36m_joint_img, self.h36m_joints_name, self.joints_name)
            h36m_joint_cam = transform_joint_to_other_db(h36m_joint_cam, self.h36m_joints_name, self.joints_name)
            h36m_joint_valid = transform_joint_to_other_db(h36m_joint_valid, self.h36m_joints_name, self.joints_name)
            h36m_joint_trunc = transform_joint_to_other_db(h36m_joint_trunc, self.h36m_joints_name, self.joints_name)
           
            # 3D data rotation augmentation
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1]], dtype=np.float32)
            # h36m coordinate
            h36m_joint_cam = np.dot(rot_aug_mat, h36m_joint_cam.transpose(1,0)).transpose(1,0) / 1000 # milimeter to meter
            h36m_joint_root = root_joint_normalize(h36m_joint_img,self.h36m_joints_name)
            inputs = {'img': img}
            targets ={'orig_joint_img':h36m_joint_img,'orig_joint_cam': h36m_joint_cam,'normalize_joint_root':h36m_joint_root}
            meta_info = {'orig_joint_valid': h36m_joint_valid, 'orig_joint_trunc': h36m_joint_trunc, 'is_3D': float(True)}
            return inputs, targets, meta_info
        else:
            inputs = {'img': img}
            targets = {}
            meta_info = {'bb2img_trans': bb2img_trans}
            return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):

        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'coor_loss': [], 'z_coor_loss':[]}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            # coordinates loss
            joint_coord = out['joint_coord_img']
            # mesh from lixel
            # x,y: resize to input image space and perform bbox to image affine transform
            joint_coord[:,0] = joint_coord[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            joint_coord[:,1] = joint_coord[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            joint_coord_xy1 = np.concatenate((joint_coord[:,:2], np.ones_like(joint_coord[:,:1])),1)
            joint_coord[:,:2] = np.dot(out['bb2img_trans'], joint_coord_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            # z: devoxelize and translate to absolute depth
            root_joint_depth = annot['root_joint_depth']
            joint_coord[:,2] = (joint_coord[:,2] / cfg.output_hm_shape[0] * 2. - 1) * (cfg.bbox_3d_size * 1000 / 2)
         
            joint_coord[:,2] = joint_coord[:,2] + root_joint_depth
            # camera back-projection
            cam_param = annot['cam_param']
            focal, princpt = cam_param['focal'], cam_param['princpt']
            joint_coord_cam = pixel2cam(joint_coord, focal, princpt)
         

            # h36m joint from gt mesh
            #pose_coord_gt_h36m = annot['joint_cam'] 
            #pose_coord_gt_h36m = annot['joint_img'] 
            #pose_coord_gt_h36m = pose_coord_gt_h36m - pose_coord_gt_h36m[self.h36m_root_joint_idx,None] # root-relative 
            #print("joint_img")
            #for joint in pose_coord_gt_h36m:
                #print('[',joint[0] ,',',joint[1] , ',',joint[2] ,'],')
            #pose_coord_gt_h36m = pose_coord_gt_h36m[self.h36m_eval_joint,:] 
            
            pose_coord_gt_h36m = annot['joint_cam'] 
            pose_coord_gt_h36m = pose_coord_gt_h36m - pose_coord_gt_h36m[self.h36m_root_joint_idx,None] # root-relative 
            #print("joint_cam")
            #for joint in pose_coord_gt_h36m:
               # print('[',joint[0] ,',',joint[1] , ',',joint[2] ,'],')
            pose_coord_gt_h36m = pose_coord_gt_h36m[self.h36m_eval_joint,:] 
            
            # h36m joint from lixel mesh
            #joint_coord= joint_coord-joint_coord[self.h36m_root_joint_idx,None] # root-relative
            #joint_coord = transform_joint_to_other_db(joint_coord, self.joints_name, self.h36m_joints_name)
            #print("output_img")
            #for joint in joint_coord:
                #print('[',joint[0] ,',',joint[1] , ',',joint[2] ,'],')
            
            #joint_coord= joint_coord[self.h36m_eval_joint,:]
             
            joint_coord= joint_coord_cam-joint_coord_cam[self.h36m_root_joint_idx,None] # root-relative
            joint_coord = transform_joint_to_other_db(joint_coord, self.joints_name, self.h36m_joints_name)
            joint_coord= joint_coord[self.h36m_eval_joint,:]
            #print("output_cam")
            #for joint in joint_coord:
                #print('[',joint[0] ,',',joint[1] , ',',joint[2] ,'],')
            #break
            eval_result['coor_loss'].append(np.sqrt(np.sum((joint_coord- pose_coord_gt_h36m)**2,1)).mean())
            
            vis = False
            if vis:
                filename = annot['img_path'].split('/')[-1][:-4]

                img = load_img(annot['img_path'])[:,:,::-1]
                cv2.imwrite(filename + '.jpg', img)

        return eval_result

    def print_eval_result(self, eval_result):
        print('Joint coordinate loss: %.2f mm' %np.mean(eval_result['coor_loss']))
        

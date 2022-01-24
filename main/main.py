import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

from importlib import reload

# no enough disk quota 
# torch.hub.set_dir('../weights')

# load pretrained YOLO5
#YOLO5_model = torch.hub.load('ultralytics/yolov5','yolov5m',force_reload = True, pretrained=True)
#YOLO5_model.cuda()

# need to import and reload utils, because there is also 'utils' package in YOLO being used.
#import utils
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
#reload(utils)

from config import cfg
from model import get_model
from utils.preprocessing import process_bbox, generate_patch_image
from utils.transforms import pixel2cam, cam2pixel


video_dir = "/content"
action_list = ['Jump_Jack', 'Wide_Push_Up']
import json
def fitness_video(video_dir, action_list):
    standard_fitness_action = []
    for action_name in action_list:
        video_path = osp.join(video_dir, '{}.mp4'.format(action_name))
        vr = cv2.VideoCapture(video_path)

        if (vr.isOpened() == False):
            print("Error openeing the file")
            continue
        result  = {'name': action_name, 'joint_coords': []}
        while (vr.isOpened()):
            success, original_img  = vr.read()
            if success:
                # use frame


                original_img_height, original_img_width = original_img.shape[:2]

                # prepare bbox
                # shape of (N = number of detected objects ,6)   xmin	ymin	xmax	ymax  confidence class
                bboxs = YOLO5_model([original_img]).xyxy[0]
                bboxs = bboxs [ bboxs[: , 5] ==0 ]
                # the bbox is already sorted by confidence
                bbox = []
                if len(bboxs >0):
                    xmin = bboxs[0][0]
                    ymin = bboxs[0][1]
                    width = bboxs[0][2] - xmin
                    height = bboxs[0][3] - ymin
                    bbox = [xmin , ymin, width, height]
                else:
                    bbox = [1.0, 1.0, original_img_width, orignal_img_height] 

                #bbox = [139.41, 102.25, 222.39, 241.57] # xmin, ymin, width, height
                bbox = process_bbox(bbox, original_img_width, original_img_height)
                img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False,
cfg.input_img_shape) 

                #img = original_img
                img = transform(img.astype(np.float32))/255
                img = img.cuda()[None,:,:,:]

                # forward
                inputs = {'img': img}
                targets = {}
                meta_info = {'bb2img_trans':None}
                with torch.no_grad():
                    out = model(inputs,targets, meta_info, 'test' )
                    result['joint_coords'].append(out['joint_coord_img'].cpu().numpy().tolist())
            else:
                break
        standard_fitness_action.append(result)
    #print(standard_fitness_action)
    with open('./standard_actions','w') as f:
        json.dump(standard_fitness_action,f)
    vr.release()




    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default = '0',dest='gpu_ids')
    parser.add_argument('--test_epoch', default=12,type=str, dest='test_epoch')
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        assert 0, print("Lack of gpu")
    #torch.cuda.set_device(0)
    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'
    return args

# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids, 'lixel')
cudnn.benchmark = True

# SMPL joint set
joint_num = 29 # original: 24. manually add nose, L/R eye, L/R ear
joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear')
flip_pairs = ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), (22,23) , (25,26), (27,28) )
skeleton = ( (0,1), (1,4), (4,7), (7,10), (0,2), (2,5), (5,8), (8,11), (0,3), (3,6), (6,9), (9,14), (14,17), (17,19), (19, 21), (21,23), (9,13), (13,16), (16,18), (18,20), (20,22), (9,12), (12,24), (24,15), (24,25), (24,26), (25,27), (26,28) )

# snapshot load

#model_path = '../weights/snapshot_%d.pth.tar' % int(args.test_epoch)

model_path = os.path.join(cfg.model_dir, 'snapshot_demo.pth.tar')
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model( joint_num, 'test')

model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
transform = transforms.ToTensor()
img_path = '../demo/input.jpg'
original_img = cv2.imread(img_path)
original_img_height, original_img_width = original_img.shape[:2]

# prepare bbox
# shape of (N = number of detected objects ,6)   xmin	ymin	xmax	ymax  confidence class
#bboxs = YOLO5_model([img_path]).xyxy[0]
#bboxs = bboxs [ bboxs[: , 5] ==0 ]
# the bbox is already sorted by confidence
bbox = []
#if len(bboxs >0):
#    xmin = bboxs[0][0]
#    ymin = bboxs[0][1]
#    width = bboxs[0][2] - xmin
#    height = bboxs[0][3] - ymin
#    bbox = [xmin , ymin, width, height]
#else:
bbox = [1.0, 1.0, original_img_width, original_img_height] 

#bbox = [139.41, 102.25, 222.39, 241.57] # xmin, ymin, width, height
bbox = process_bbox(bbox, original_img_width, original_img_height)
img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape) 

img = original_img
img = transform(img.astype(np.float32))/255
img = img.cuda()[None,:,:,:]

# forward
inputs = {'img': img}
targets = {}
meta_info = {'bb2img_trans':None}
with torch.no_grad():
    out = model(inputs,targets, meta_info, 'test' )
#print(out)


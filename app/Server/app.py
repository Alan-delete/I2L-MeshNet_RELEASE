from flask import Flask, send_from_directory
from flask import jsonify
from flask import request
from flask_ngrok import run_with_ngrok
from flask_cors import CORS
from markupsafe import escape
import json
import os
import sys
import cv2
from numpy.core.numeric import Infinity
from werkzeug.utils import secure_filename
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
from importlib import reload

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}

# load YOLO5
YOLO5_model = torch.hub.load('ultralytics/yolov5','yolov5m', pretrained=True)
YOLO5_model.cuda()

# There is 'utils' in YOLO which will conflict with local 'utils' module, we need to import and override utils 
import utils

# current path is assumed to be root_dir/app/Server/app.py
root_dir =os.path.dirname( os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
main_dir = os.path.join(root_dir,'main')
common_dir = os.path.join(root_dir,'common')
sys.path.append(main_dir)
sys.path.append(common_dir)

reload(utils)



from config import cfg
from model import get_model
from nets.SemGCN.export import SemGCN
from utils.transforms import transform_joint_to_other_db
from utils.preprocessing import process_bbox,generate_patch_image

app = Flask(__name__ ,static_folder = 'public',static_url_path='/public')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)
run_with_ngrok(app)   
cudnn.benchmark = True


class Action_reader():
    def __init__(self, json_name = 'standard_joints.json'):
        json_file = os.path.join(app.static_folder, json_name)  
        assert os.path.exists(json_file), 'Cannot find json file'
        standard_action = ['INIT']
        with open(json_file) as f:
            standard_action = json.load(f)  
            self.standard_action = standard_action

    def __getitem__(self, idx):
        return self.standard_action[idx]

    def __len__(self):
        return len(self.standard_action)

    def get_json(self):
        return self.standard_action
    
    def get_loss(self, user_action, gt_action):
        loss = 0
        for key in user_action.keys():
            loss += np.absolute( np.array(user_action[key]) - \
            np.array( gt_action[key] ) ).mean()
        return loss

    def get_frame_idx(self, user_action):
        #result = {'action_idx':None, 'frame_idx':None}
        loss = Infinity
        first_idx = 0
        second_idx = 0
        for action_idx, action in enumerate (self.standard_action) :
            for frame_idx, action_per_frame in enumerate(action['data']):
                temp_loss = self.get_loss(user_action, action_per_frame)
                if (temp_loss<loss):
                    loss = temp_loss
                    first_idx = action_idx
                    second_idx = frame_idx
        return first_idx,second_idx

    def get_action_list(self):
        action_list = [ action['name'] for action in self.standard_action]
        return action_list

class Videos_reader():
    def __init__(self,action_reader, video_dir = "Fitness_video"):
        self.ar = action_reader
        self.videos = []   
        action_list = action_reader.get_action_list()
        for action_name in action_list:
            video = []
            video_path = os.path.join(app.static_folder,video_dir,'{}.mp4'.format(action_name))
            cap = cv2.VideoCapture(video_path)   
            assert cap.isOpened(), 'Fail in opening video file'
            
            while (cap.isOpened()):
                success, original_img  = cap.read()
                if  success: 
                    video.append(original_img)
                else:
                    break
            self.videos.append(video)
            cap.release() 

    def __getitem__(self,idx):
        return self.videos[idx]

    def __len__(self):
        return len(self.videos)

    # user_action is assumed to be in form as {'human_joint_coords': , ...}
    def get_frame(self, user_action):
        action_idx, frame_idx = self.ar.get_frame_idx(user_action)
        return self.videos[action_idx][frame_idx]



def init_I2L(joint_num = 29,test_epoch = 12,mode = 'test'):

    # snapshot load
    model_path = os.path.join(cfg.model_dir,'snapshot_demo.pth.tar')
    assert os.path.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    I2L_model = get_model( joint_num, mode)
    I2L_model = DataParallel(I2L_model).cuda()
    ckpt = torch.load(model_path)
    I2L_model.load_state_dict(ckpt['network'], strict=False)
    I2L_model.eval()
    return I2L_model

def init_semGCN(test_epoch = 1):
    # snapshot load
    model_path = os.path.join(cfg.model_dir, 'sem_gcn_epoch{}.pth.tar'.format(test_epoch))
    assert os.path.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    SemGCN_model = SemGCN(cfg.skeleton).cuda()
    ckpt = torch.load(model_path)
    SemGCN_model.load_state_dict(ckpt['network'], strict=False)
    SemGCN_model.eval()
    return SemGCN_model

ar = Action_reader()
vr = Videos_reader(ar)
I2L_model = init_I2L()
SemGCN_model = init_semGCN()
#cv2.imwrite('./test.png',vr.get_frame(ar[1]['data'][20]))

dummyCoordinates = [ 39.7642, 22.7078, 31.9892,
     
    40.6116, 26.0905, 34.2193,
     
    36.0548, 25.0538, 32.6827,
     
    41.1295, 18.4470, 30.9866,
     
    43.6070, 38.7190, 29.3077,
     
    27.6858, 37.2311, 29.8864,
     
    43.4302, 16.8571, 27.4887,
     
    41.2941, 52.8330, 35.5459,
     
    19.6125, 47.5865, 37.6749,
     
    44.3492, 16.9762, 25.8378,
     
    42.8460, 55.7746, 32.3185,
     
    17.7097, 50.6393, 34.5038,
     
    47.5938, 12.3236, 20.6534,
     
    48.9126, 14.3306, 23.9226,
     
    43.3275, 13.3207, 22.2242,
     
    48.8497, 12.5166, 18.4363,
     
    52.3104, 14.6594, 24.7424,
     
    40.1984, 12.6481, 20.9878,
     
    53.4274, 18.9680, 31.7843,
     
    31.0148, 15.4037, 24.1125,
     
    53.2553, 26.5973, 26.6768,
     
    32.6980, 23.6892, 20.0303,
     
    53.2801, 28.8286, 24.4359,
     
    33.6501, 26.5843, 17.9701,
     
    50.3077, 12.9904, 14.9635,
     
    51.4026, 11.1491, 15.3668,
     
    48.9452, 10.8628, 15.0181,
     
    51.7383,  9.9970, 18.3521,
     
    46.2133,  9.2510, 16.5609,
     ]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/",methods=['GET', 'POST'])
def home():
    #return render_template('index.html')
    print(app.static_folder)
    return send_from_directory(app.static_folder, 'index.html')


@app.route("/getFitness",methods=['GET', 'POST'])
def get_fitness_action():
    return jsonify( ar.get_json() )



def get_output(img_path):
    with torch.no_grad():
        transform = transforms.ToTensor()
        # prepare input image
        original_img = cv2.imread(img_path)
        original_img_height, original_img_width = original_img.shape[:2]
    
        # prepare bbox
        # shape of (N = number of detected objects ,6)   xmin   ymin    xmax    ymax  confidence class
        bboxs = YOLO5_model([img_path]).xyxy[0]
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
            bbox = [1.0, 1.0, original_img_width, original_img_height]
        
        #bbox = [139.41, 102.25, 222.39, 241.57] # xmin, ymin, width, height
        bbox = process_bbox(bbox, original_img_width, original_img_height)
        img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape) 
        img = transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]
    
        # forward
        inputs = {'img': img}
        targets = {}
        meta_info = {'bb2img_trans': None}
        out = I2L_model(inputs, targets, meta_info, 'test')
        
        # of shape (29,3) (17,3)
        I2L_joints = out['joint_coord_img'][0]
        human36_joints = transform_joint_to_other_db(I2L_joints.cpu().numpy(),cfg.smpl_joints_name , cfg.joints_name)
        Sem_joints = SemGCN_model(torch.from_numpy(human36_joints).cuda()[...,:2])[0]

        return {'smpl_joint_coords':I2L_joints.tolist(),\
                'human36_joint_coords':human36_joints.tolist()}
                #'Sem_joints':Sem_joints.tolist() }

    
@app.route("/imageUpload", methods = ['PUT','POST'])
def file_upload():
    # print("file uploaded, processing")
    data = None
    store_folder = os.path.join(app.static_folder, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(store_folder):
        os.mkdir(store_folder)
    # todo: customize file save name
    # todo alt: directly pass the file to NN api
    if 'image' in request.files:
        print("upload success!")
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(store_folder, filename))
            data = get_output(os.path.join(store_folder, filename))
            x = vr.get_frame(data)
            cv2.imwrite('./test.png' , x)
            
    #return json of coordinates
    return jsonify(data)

app.run()

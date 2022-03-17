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
import argparse
from numpy.core.numeric import Infinity
from werkzeug.utils import secure_filename
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
from importlib import reload
#FOR WINDOWS
import mimetypes
import math
import platform
if(platform.system()=='Windows'):
    mimetypes.init()
    mimetypes.add_type("application/javascript", ".js", True)

UPLOAD_FOLDER = 'uploads'
FITNESS_VIDEO_FOLDER = 'Fitness_video'
ALLOWED_EXTENSIONS = {'png','jpg','jpeg','jfif'}

# load YOLO5
YOLO5_model = torch.hub.load('ultralytics/yolov5','yolov5m', pretrained=True)
YOLO5_model.cuda()

#DEFAULT SCORING DATA:
DEFAULT_JOINTS = [[1,2,1,0],[5,3,6,8],[1,2,2,3],]
DEFAULT_FPS = 24
DEFAULT_ACCURACY = 10
DEFAULT_ERROR = 20

# There is 'utils' in YOLO which will conflict with local 'utils' module, we need to import and override utils 
import utils

# current path is assumed to be root_dir/app/Server/app.py
root_dir =os.path.dirname( os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
main_dir = os.path.join(root_dir,'main')
common_dir = os.path.join(root_dir,'common')
#fix include path issue
sys.path.insert(0,main_dir)
sys.path.insert(0,common_dir)

reload(utils)

from config import cfg
from model import get_model
from nets.SemGCN.export import SemGCN
from utils.transforms import transform_joint_to_other_db
from utils.preprocessing import process_bbox,generate_patch_image,get_action_json

app = Flask(__name__ ,static_folder = 'public',static_url_path='/public')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)
if(platform.system()!='Windows'):
    run_with_ngrok(app)   
cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,default='0', dest='gpu_ids')
    parser.add_argument('--stage', type=str,default ='lixel', dest='stage')
    parser.add_argument('--continue',default = False, dest='continue_train', action='store_true')
    parser.add_argument('--non_local',default = False, dest='non_local', action='store_true')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"
 
    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    if not args.stage:
        assert 0, "Please set training stage among [lixel, param]"
   
    return args
args = parse_args()
# cfg.set_args(args.gpu, args.stage, args.continue_train, args.non_local)

class Action_reader():
    def __init__(self, json_name = 'standard_joints.json'):
        self.json_name = json_name
        json_file = os.path.join(app.static_folder, json_name)  
        assert os.path.exists(json_file), 'Cannot find json file'
        self.standard_action = []
        self.user_action_idx = None 
        with open(json_file) as f:
            standard_action = json.load(f)  
            self.standard_action = standard_action
        self.action_list = [ action['name'] for action in self.standard_action]

    def __getitem__(self, idx):
        return self.standard_action[idx]

    def __len__(self):
        return len(self.standard_action)

    def get_json(self):
        return self.standard_action
    
    def get_loss(self, user_action, gt_action):
        loss = 0
        for key in user_action.keys():
            if not key in gt_action.keys():
                continue
            loss += np.absolute( np.array(user_action[key]) - \
            np.array( gt_action[key] ) ).mean()
        return loss

    def save(self):
        path = os.path.join(app.static_folder, self.json_name)  
        assert os.path.exists(path), 'Cannot find json file'
        with open(path,'w') as f :
            json.dump( self.standard_action ,f)

    def append(self, new_action):
        self.action_list.append(new_action['name'])
        self.standard_action.append(new_action)
    
    # user_action is assumed to be in form as {'human_joint_coords': , ...}
    def get_frame_idx(self, user_action, action_choice):
        #result = {'action_idx':None, 'frame_idx':None}
        loss = Infinity
        threshold = Infinity
        first_idx = -1
        second_idx = -1
        # user choose action by himself
        if (action_choice in self.action_list):
            first_idx = self.action_list.index(action_choice)
            for frame_idx, action_per_frame in enumerate(self.standard_action[first_idx]['data']):
                temp_loss = self.get_loss(user_action, action_per_frame)
                if temp_loss<loss and temp_loss < threshold:
                    loss = temp_loss
                    second_idx = frame_idx
        
        # No selected action, scan all standard action
        else:
            for action_idx, action in enumerate (self.standard_action) :
                for frame_idx, action_per_frame in enumerate(action['data']):
                    temp_loss = self.get_loss(user_action, action_per_frame)
                    if temp_loss<loss and temp_loss < threshold:
                        loss = temp_loss
                        first_idx = action_idx
                        second_idx = frame_idx
        return first_idx,second_idx, loss

    def get_action_list(self):
        return self.action_list

class Videos_reader():
    def __init__(self, action_list, video_dir = "Fitness_video"):
        self.videos = []   
        self.video_dir = video_dir
        self.action_list = action_list
        for action_name in self.action_list:
            video = []
            video_path = os.path.join(app.static_folder,self.video_dir,'{}.mp4'.format(action_name))
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

    # return name and frame data
    def get_frame(self, action_idx,frame_idx):
        return self.action_list[action_idx], self.videos[action_idx][frame_idx]

    # could be optimized later 
    def update(self,action_list):
        self.videos = []   
        self.action_list = action_list
        for action_name in self.action_list:
            video = []
            video_path = os.path.join(app.static_folder,self.video_dir,'{}.mp4'.format(action_name))
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

class Scoring():
    def __init__(self, action_list, json_name = 'comparison_data.json'):
        self._json = json_name
        json_file = os.path.join(app.static_folder, self._json) 
        assert os.path.exists(json_file), 'Cannot find json file'
        with open(json_file) as f:
            self._scoring_data = json.load(f)
        self._action_list = [ item['name'] for item in self._scoring_data]
        if(self._action_list != action_list):
            print("warning: action list from joint data and scoring data unmathces")
            
    def get_action_list(self):
        return self._action_list
    
    def get_scoring_data(self):
        return self._scoring_data
    
    def time_to_frame(self, action_index, timestamp):
        return math.floor(int(self._scoring_data[action_index]['fps']) * float(timestamp))
                
    def append(self,name, version = 17, joints = DEFAULT_JOINTS, fps = DEFAULT_FPS, accuracy = DEFAULT_ACCURACY, error = DEFAULT_ERROR):
        self._action_list.append(name)
        new_scoring_data = {'name':name, 'version':version, 'joints':joints, 'fps': fps, 'accuracy': accuracy, 'error': error}
        self._scoring_data.append(new_scoring_data)

    def save(self):
        path = os.path.join(app.static_folder, self._json)  
        assert os.path.exists(path), 'Cannot find json file'
        with open(path,'w') as f :
            json.dump( self._scoring_data ,f)
    
    #TODO: main calculating function
    def score(self,user_action,action_index,recorded_action,predicted_action):
        scoring_data = self._scoring_data[action_index]
        #print(scoring_data)
        joints = np.array(scoring_data['joints'])
        recorded_error, predicted_error = [],[]
        for joint in joints:
            # print(joint)
            recorded_error.append(self.score_joint(joint,user_action,recorded_action))
            predicted_error.append(self.score_joint(joint,user_action,predicted_action))
        return recorded_error, predicted_error
    
    def score_joint(self,joint,user_action,standard_action):
        user_bones = self.get_bones(joint, user_action)
        standard_bones = self.get_bones(joint, standard_action)
        
        user_dot = np.dot(user_bones[0],user_bones[1])
        standard_dot = np.dot(standard_bones[0],standard_bones[1])
        
        user_angle = math.fabs(np.arccos(np.clip(user_dot, -1.0, 1.0)) / math.pi * 180 )
        standard_angle = math.fabs(np.arccos(np.clip(standard_dot, -1.0, 1.0)) / math.pi * 180 )
        user_error = math.fabs(user_angle - standard_angle)
        return user_error
        
    def get_bones(self, joint, action):
        bone1 = action[joint[0]] - action[joint[1]]
        bone1 = bone1 / np.linalg.norm(bone1)
        bone2 = action[joint[2]] - action[joint[3]]
        bone2 = bone2 / np.linalg.norm(bone2)
        return  [bone1, bone2]
    
    #TODO: reload function

def init_I2L(test_epoch = 12,mode = 'test'):
    # snapshot load
    if args.stage == 'sem_gcn':
        joint_num = 17
        model_path = os.path.join(cfg.model_dir,'second_hybrid_8.pth.tar')
    else:
        joint_num = 29
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
vr = Videos_reader(action_list = ar.get_action_list(), video_dir= FITNESS_VIDEO_FOLDER)
sc = Scoring(action_list = ar.get_action_list())
I2L_model = init_I2L()
SemGCN_model = init_semGCN()

#cv2.imwrite('./test.png',vr.get_frame(ar[1]['data'][20]))


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
    return jsonify(ar.get_action_list() )
    #return jsonify( ar.get_json() )



def get_output(img_path):
    with torch.no_grad():
        if args.stage=='sem_gcn':
            normalize = transforms.Normalize(mean = [0.485,0.456,0.406], std  = [0.229,0.224,0.225])
            transform = transforms.Compose([transforms.ToTensor(),normalize])
        else:
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
            bbox = [xmin.cpu() , ymin.cpu(), width.cpu(), height.cpu()]
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
        if args.stage == 'sem_gcn':
            human36_joints = I2L_joints
            return { 'human36_joint_coords':human36_joints.tolist()}
        else:
            human36_joints = transform_joint_to_other_db(I2L_joints.cpu().numpy(),cfg.smpl_joints_name , cfg.joints_name)
            return {'smpl_joint_coords':I2L_joints.tolist(),\
                'human36_joint_coords':human36_joints.tolist()}

# as tested on my laptop, currently the speed of file upload and neural network process is nearly 1 frame per second. For pure neural network process, 19.84 seconds for 100 image   
# also returns match_action name, the action estimate will be executed on front end, since it's little calculation and every user has their own different data record.
@app.route("/realTimeUpload", methods = ['PUT','POST'])
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
            action_idx, frame_idx, loss = ar.get_frame_idx(data, request.values['action_choice'] )
            if action_idx == -1:
                data['action_name'] = 'Loss exceeds threshold!'
                data['loss'] = loss             
            else:
                #TODO: obtain the predicted and recorded action's joint data
                match_action, match_frame = vr.get_frame(action_idx, frame_idx)
                data['action_name'] = match_action
                data['loss'] = loss
                predicted_action = np.array(ar[action_idx]['data'][frame_idx]['human36_joint_coords'])
                if(not 'timestamp' in request.values):
                    print("timestamp not found")
                    recorded_action = predicted_action
                else:
                    recorded_frame = sc.time_to_frame(action_idx,request.values['timestamp']) % len(ar[action_idx]['data'])
                    recorded_action = np.array(ar[action_idx]['data'][recorded_frame]['human36_joint_coords'])
                data['action_accuracy'] = sc.score(np.array(data['human36_joint_coords']),action_idx,recorded_action,predicted_action)
                cv2.imwrite(os.path.join(app.static_folder, 'match_frame.png') , match_frame)
            
    #return json of coordinates
    return jsonify(data)
@app.route("/staticUpload", methods = ['POST', 'PUT'])
def action_upload():
    data = None
    if 'image' in request.files:
        store_folder = os.path.join(app.static_folder, app.config['UPLOAD_FOLDER'])
        if not os.path.exists(store_folder):
            os.mkdir(store_folder)
        print("upload success!")
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(store_folder, filename))
            data = get_output(os.path.join(store_folder, filename))
            action_idx, frame_idx, loss = ar.get_frame_idx(data, 'defalut' )
            if action_idx == -1:
                data['action_name'] = 'Loss exceeds threshold!'
                data['loss'] = loss             
            else:
                match_action, match_frame = vr.get_frame(action_idx, frame_idx)
                data['action_name'] = match_action
                data['loss'] = loss
                predicted_action = np.array(ar[action_idx]['data'][frame_idx]['human36_joint_coords'])
                if(not 'timestamp' in request.values):
                    print("timestamp not found")
                    recorded_action = predicted_action
                else:
                    recorded_frame = sc.time_to_frame(action_idx,request.values['timestamp'])
                    recorded_action = np.array(ar[action_idx]['data'][recorded_frame]['human36_joint_coords'])
                data['action_accuracy'] = sc.score(np.array(data['human36_joint_coords']),action_idx,recorded_action,predicted_action)
                cv2.imwrite(os.path.join(app.static_folder, 'match_frame.png') , match_frame)
       
    if 'video' in request.files:
        store_folder = os.path.join(app.static_folder, FITNESS_VIDEO_FOLDER)
        if not os.path.exists(store_folder):
            os.mkdir(store_folder)

        print('upload succeed!')
        file = request.files['video']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)   

        filename = secure_filename(file.filename)
        file.save(os.path.join(store_folder, filename))
        data = get_action_json(os.path.join(store_folder, filename),I2L_model)    
        ar.append(data)
        ar.save()
        vr.update(ar.get_action_list())
    
    return jsonify(data)


app.run()

#filestr = request.files['image'].read()
#npimg = np.frombuffer(filestr, numpy.uint8)
#img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
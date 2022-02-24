from config import cfg
from nets.SemGCN.export import SemGCN
import torch 
import torchvision.transforms as transforms
from Human36M import Human36M
import os
import numpy as np
from base import Tester
import torch.backends.cudnn as cudnn
from utils.transforms import transform_joint_to_other_db



epoch = 1
sem_gcn = SemGCN(cfg.skeleton)
#db = Human36M(transforms.ToTensor(), 'test')
#data_loader = torch.utils.data.DataLoader(db, batch_size =8 , shuffle = True) 
model_path = os.path.join(cfg.model_dir, 'sem_gcn_epoch{}.pth.tar'.format(epoch))
ckpt = torch.load(model_path)
sem_gcn.load_state_dict(ckpt['network'], strict = False)
sem_gcn.eval()
sem_gcn.cuda()

loss_fn  = torch.nn.MSELoss()


# joints num is 17 for now, go Human36M.py to change joints num 
tester = Tester(12)
tester._make_batch_generator()
tester._make_model()

losses = []
sem_losses = []
I2L_losses = []
for itr, (inputs, targets, meta_info) in enumerate((tester.batch_generator)):  
    # forward
    orig_joints = targets['orig_joint_img'].cuda()
    with torch.no_grad():
        out = tester.model(inputs, targets, meta_info, 'test' )
        joints= out['joint_coord_img'][0]
        joints = joints.cpu().numpy()
        joints = transform_joint_to_other_db( joints, cfg.smpl_joints_name, cfg.joints_name)
        joints = torch.from_numpy(joints).cuda()
        sem_joints = sem_gcn(joints[...,:2])
        
        loss = loss_fn(sem_joints, joints)
        print("batch {} loss :{}".format( itr, loss.item()))
        losses.append(loss.item())  
        sem_losses.append( loss_fn(sem_joints, orig_joints[0]).item() )
        I2L_losses.append( loss_fn(joints, orig_joints[0]).item() )
        #if (loss_fn(sem_joints, orig_joints).item()>50 ):
            #print("target:",orig_joints)
            #print("sem:",sem_joints)
            #print("I2L:",joints)
            #break
#    print(joints)
#    print(sem_joints)
#    break
 

 
print("Two model loss:",sum(losses)/len(losses))
print("Sem_gcn loss:",sum(sem_losses)/len(sem_losses))
print("I2L loss:",sum(I2L_losses)/len(I2L_losses))


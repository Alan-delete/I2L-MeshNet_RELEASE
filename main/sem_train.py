from config import cfg
from nets.SemGCN.export import SemGCN
import torch 
import torchvision.transforms as transforms
from Human36M import Human36M
import os
#
#class sem_gcn_dataset(Human36M):
    #def __init__(self, transform, mode):
        #super(sem_gcn_dataset, self).__init__(transform, mode)
    #def __getitem__(self, idx):
        #data = copy.deepcopy(self.datalist[idx])
         #
        #img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        #img = self.transform(img.astype(np.float32))/255.
        #
        #if self.data_split == 'train':
            ## h36m gt
            #h36m_joint_img = data['joint_img']
            #h36m_joint_cam = data['joint_cam']
            #h36m_joint_cam = h36m_joint_cam - h36m_joint_cam[self.h36m_root_joint_idx,None,:] # root-relative
            #h36m_joint_valid = data['joint_valid']
            #if do_flip:
                #h36m_joint_cam[:,0] = -h36m_joint_cam[:,0]
                #h36m_joint_img[:,0] = img_shape[1] - 1 - h36m_joint_img[:,0]
                #for pair in self.h36m_flip_pairs:
                    #h36m_joint_img[pair[0],:], h36m_joint_img[pair[1],:] = h36m_joint_img[pair[1],:].copy(),
#h36m_joint_img[pair[0],:].copy()
                    #h36m_joint_cam[pair[0],:], h36m_joint_cam[pair[1],:] = h36m_joint_cam[pair[1],:].copy(),
#h36m_joint_cam[pair[0],:].copy()
                    #h36m_joint_valid[pair[0],:], h36m_joint_valid[pair[1],:] =
#h36m_joint_valid[pair[1],:].copy(), h36m_joint_valid[pair[0],:].copy()
#
            #h36m_joint_img_xy1 = np.concatenate((h36m_joint_img[:,:2], np.ones_like(h36m_joint_img[:,:1])),1)
            #h36m_joint_img[:,:2] = np.dot(img2bb_trans, h36m_joint_img_xy1.transpose(1,0)).transpose(1,0)
            #h36m_joint_img[:,0] = h36m_joint_img[:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            #h36m_joint_img[:,1] = h36m_joint_img[:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
            #h36m_joint_img[:,2] = h36m_joint_img[:,2] - h36m_joint_img[self.h36m_root_joint_idx][2] #
#root-relative
            #h36m_joint_img[:,2] = (h36m_joint_img[:,2] / (cfg.bbox_3d_size * 1000  / 2) + 1)/2. *
#cfg.output_hm_shape[0] # change cfg.bbox_3d_size from meter to milimeter
            #
            ## check truncation
            #h36m_joint_trunc = h36m_joint_valid * ((h36m_joint_img[:,0] >= 0) * (h36m_joint_img[:,0] <
#cfg.output_hm_shape[2]) * \
                        #(h36m_joint_img[:,1] >= 0) * (h36m_joint_img[:,1] < cfg.output_hm_shape[1]) * \
                        #(h36m_joint_img[:,2] >= 0) * (h36m_joint_img[:,2] <
#cfg.output_hm_shape[0])).reshape(-1,1).astype(np.float32)
#
#
#

def save_ckpt(state):
    file_path = os.path.join(cfg.model_dir, 'sem_gcn_epoch{}.pth.tar'.format(state['epoch']))
    torch.save(state,file_path)




sem_gcn = SemGCN(cfg.skeleton)
optimizer = torch.optim.Adam(sem_gcn.parameters(), lr = 0.0005)
loss_fn = torch.nn.MSELoss( reduction = 'mean')
#save_ckpt({'epoch':1, 'network':sem_gcn.state_dict(), 'optimizer':optimizer.state_dict() })
db = Human36M(transforms.ToTensor(), 'train')
data_loader = torch.utils.data.DataLoader(db, batch_size = 4, shuffle = True) 

for i, (inputs, targets, meta_info) in enumerate(data_loader):
    optimizer.zero_grad()
    input = targets['orig_joint_img'][... , :2]
    #print(targets['orig_joint_img'])
    output = sem_gcn(input) 
    loss = loss_fn (output, targets['orig_joint_img'])
    print("batch {} loss :{}".format( i, loss.item()))
    #loss = sum(loss)
    loss.backward()
    optimizer.step()
save_ckpt({'epoch':1, 'network':sem_gcn.state_dict(), 'optimizer':optimizer.state_dict() })


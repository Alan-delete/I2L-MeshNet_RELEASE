import argparse
from config import cfg
from nets.SemGCN.export import SemGCN
import torch 
import torchvision.transforms as transforms
from Human36M import Human36M
import os
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type = str,
default = '0', dest = 'gpu_ids')
    parser.add_argument('--continue', defalut =
True, dest= 'continue_train', action = 'store_true')
    parser.add_argument('--epoch', default = 1,
type = int, dest = 'epoch')
    args = parser.parse_args()
    if not args.gpu_ids:
        assert 0 , "Please set proper gpu ids"
    return args


def save_ckpt(state):
    file_path = os.path.join(cfg.model_dir, 'sem_gcn_feat_epoch{}.pth.tar'.format(state['epoch']))
    torch.save(state,file_path)

def main():
    torch.backends.cudnn.benchmard = True

    sem_gcn = SemGCN(cfg.skeleton)
    optimizer = torch.optim.Adam(sem_gcn.parameters(), lr = 0.0002)
    loss_fn = torch.nn.MSELoss( reduction = 'mean')
    #save_ckpt({'epoch':1, 'network':sem_gcn.state_dict(), 'optimizer':optimizer.state_dict() })
    db = Human36M(transforms.ToTensor(), 'train')
    data_loader = torch.utils.data.DataLoader(db, batch_size = 4, shuffle = True) 
    
    epoch = 1
    model_path = os.path.join(cfg.model_dir, 'sem_gcn_epoch{}.pth.tar'.format(epoch))
    ckpt = torch.load(model_path)
    sem_gcn.load_state_dict(ckpt['network'], strict = False)
    sem_gcn.eval()
    sem_gcn.cuda()
    
    
    for i, (inputs, targets, meta_info) in enumerate(data_loader):
        joints = targets['orig_joint_img'].cuda()
        optimizer.zero_grad()
        input = joints[... , :2]
        #print(targets['orig_joint_img'])
        output = sem_gcn(input) 
        loss = loss_fn (output, joints)
        print("batch {} loss :{}".format( i, loss.item()))
        #loss = sum(loss)
        loss.backward()
        optimizer.step()
    save_ckpt({'epoch':2, 'network':sem_gcn.state_dict(), 'optimizer':optimizer.state_dict() })

if __name__ == "__main__":
    main()    

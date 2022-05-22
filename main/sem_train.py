import argparse
import os
from config import cfg
from nets.SemGCN.export import SemGCN 
import torchvision.transforms as transforms
from tqdm import tqdm
import torch
from base import Trainer
import torch.backends.cudnn as cudnn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,default='0', dest='gpu_ids')
    parser.add_argument('--stage', type=str,default ='lixel', dest='stage')
    parser.add_argument('--continue',
default = False, dest='continue_train', action='store_true')
    parser.add_argument('--non_local',default =False, dest='non_local', action = 'store_true')
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
def save_ckpt(state):
    file_path = os.path.join(cfg.model_dir, 'mirror_sem_gcn_epoch{}.pth.tar'.format(state['epoch']))
    torch.save(state, file_path)
def main():
    
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.stage, args.continue_train, args.non_local)
    #cudnn.benchmark = True
    print('Stage: ' + args.stage)
    
    start_epoch = 7
    trainer = Trainer()
    trainer._make_batch_generator()
    cfg.skeleton = cfg.mirror_skeleton
    sem_gcn = SemGCN( cfg.skeleton, coords_dim = (2,1))
    sem_gcn.cuda()
    #trainer._make_model()
    loss_fn = torch.nn.MSELoss( reduction = 'mean')
    trainer.optimizer = torch.optim.Adam(sem_gcn.parameters(), lr = 0.0002)
    if (args.continue_train):
        ckpt_path = os.path.join( cfg.model_dir, 'mirror_sem_gcn_epoch{}.pth.tar'.format(start_epoch))
        ckpt = torch.load(ckpt_path)
        sem_gcn.load_state_dict(ckpt['network'], strict= False)
        trainer.optimizer.load_state_dict(ckpt['optimizer'])       
        start_epoch += 1
    torch.autograd.set_detect_anomaly(True)
    # train
    for epoch in range(start_epoch, cfg.end_epoch):
        
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()
            joints = targets['orig_joint_img'].cuda()
            # forward
            trainer.optimizer.zero_grad()
            input = joints[..., :2]
            output = sem_gcn(input)
            loss = {'joint_coord_loss': loss_fn(output, joints[...,2, None])}
            loss = {k:loss[k].mean() for k in loss}
            # backward
            sum(loss[k] for k in loss).backward()
            torch.nn.utils.clip_grad_norm_(sem_gcn.parameters(), 1.)
            trainer.optimizer.step()
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
            screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
            trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()
        
        save_ckpt({'epoch':epoch, 'network':sem_gcn.state_dict(), 'optimizer':trainer.optimizer.state_dict() }) 

if __name__ == "__main__":
    main()

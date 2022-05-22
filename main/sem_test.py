import torch
import os
import argparse
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np
import torch.backends.cudnn as cudnn
from config import cfg
from nets.SemGCN.export import SemGCN 
from base import Tester

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', default = '12', type=str, dest='test_epoch')
    parser.add_argument('--gpu', type=str,default='0', dest='gpu_ids')
    parser.add_argument('--stage', type=str,default ='lixel', dest='stage')
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

    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids, args.stage )
    cudnn.benchmark = True
    print('Stage: ' + args.stage)

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    #cfg.skeleton = cfg.mirror_skeleton
    sem_gcn = SemGCN(cfg.skeleton, coords_dim = (2,1))
    epoch = 8
    model_path = os.path.join(cfg.model_dir, 'mirror_sem_gcn_epoch{}.pth.tar'.format(epoch))
    model_path = os.path.join(cfg.model_dir, 'sem_gcn_epoch{}.pth.tar'.format(epoch))
    ckpt = torch.load(model_path)
    sem_gcn.load_state_dict(ckpt['network'], strict = False)
    sem_gcn.cuda()
    #state = {'epoch':10, 'network':tester.model.state_dict()}
    #file_path = os.path.join(cfg.model_dir, 'snapshot_demo.pth.tar')
    #torch.save(state, file_path)    
    #print("successfully saved")
 
    eval_result = {}
    cur_sample_idx = 0
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        
        # forward
        input = targets['orig_joint_img'][...,:2].cuda()
        with torch.no_grad():
            joints = torch.cat((input, sem_gcn(input)),2)
            out = {'joint_coord_img':joints}
            out['bb2img_trans'] = meta_info['bb2img_trans']


        #print(out['joint_coord_img'][0])
        #print(targets['orig_joint_img'][0])
        #break
       
        # save output
        out = {k: v.cpu().numpy() for k,v in out.items()}
        for k,v in out.items(): batch_size = out[k].shape[0]
        out = [{k: v[bid]  for k,v in out.items()}for bid in range(batch_size)]
        # evaluate
        cur_eval_result = tester._evaluate(out, cur_sample_idx)
        #break
        for k,v in cur_eval_result.items():
            if k in eval_result: eval_result[k] += v
            else: eval_result[k] = v
        cur_sample_idx += len(out)
    
    print('Parameters number:', sum(p.numel() for p in
sem_gcn.parameters() if p.requires_grad))    

    tester._print_eval_result(eval_result)

if __name__ == "__main__":
    main()

from config import cfg
from nets.SemGCN.export import SemGCN
import torch 
import torchvision.transforms as transforms
from Human36M import Human36M
x = Human36M(transforms.ToTensor(), 'test')
print(x[1])
#sem_gcn = SemGCN(cfg.skeleton)
#input = torch.ones((1,17,2) )
#output = sem_gcn(input)
#print(output)
print(len(x))


from collections import OrderedDict
import torch


pth_path = '/home/csj/desk2t/Code/mmVSPW/work_dirs/iter_12500.pth'
param = torch.load(pth_path, map_location='cpu')['state_dict']
q1 = param['decode_head.query_embed.weight']
q2 = param['decode_head.query_feat.weight']
q1_ = torch.cat((q1.clone(), q1.clone()), 0)
q2_ = torch.cat((q2.clone(), q2.clone()), 0)
param['decode_head.query_embed.weight'] = q1_
param['decode_head.query_feat.weight'] = q2_
torch.save(param, '/home/csj/desk2t/Code/mmVSPW/work_dirs/iter_12500q200.pth')

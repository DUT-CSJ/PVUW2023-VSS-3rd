import torch
from collections import OrderedDict

weight_path1 = '/home/csj/desk2t/Code/mmVSPW/work_dirs/swin2a1002/iter_12500.pth'
weight_path2 = '/home/csj/desk2t/Code/mmVSPW/work_dirs/swin2a1002/iter_25000.pth'
weight_path = [weight_path1, weight_path2]
state_list = []
for i in range(len(weight_path)):
    state_dict = torch.load(weight_path[i], map_location='cuda')['state_dict']
    state_list.append(state_dict)
print('load checkpoints')

# worker_state_dict = [x.state_dict() for x in models]
weight_keys = list(state_list[0].keys())
fed_state_dict = OrderedDict()
for key in weight_keys:
    key_sum = 0
    for i in range(len(state_list)):
        key_sum = key_sum + state_list[i][key]
    fed_state_dict[key] = key_sum / len(state_list)
#### update fed weights to fl model
torch.save(fed_state_dict, '/home/csj/desk2t/Code/mmVSPW/work_dirs/swin2a1002/ensemble.pth')


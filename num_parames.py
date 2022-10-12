import torch

# path = '/home/yaboliu/.cache/torch/checkpoints/osnet_x0_25_imagenet.pth'
# # path = '/home/yaboliu/work/frameworks/hrnet/models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth'
# # path = '/home/yaboliu/work/frameworks/hrnet/models/pytorch/pose_coco/pose_hrnet_w48_256x192.pth'
# path = '/home/yaboliu/work/frameworks/yolov5/pretrain_models/yolov5m.pt'
# path = '/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/HR_shanghai/01/high/stc/Feb19_0154/checkpoints/32_Feb19_0154_stc_sagc_checkpoint.pth.tar' # high
path = '/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/HR_shanghai/03_05_07/low/stc/Apr05_0459/checkpoints/3_Apr05_0459_stc_sagc_checkpoint.pth.tar'

model = torch.load(path)['gcae']
num = 0
for key in model.keys():
    try:
        print('catch ', key)
        # print(model[key].numel())
        num += model[key].numel()
    except:
        print('miss ', key)
        continue
print(num)


# path = '/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation_zhengshi/ablation/HR_shanghai/01/high/stc/Feb19_0154/checkpoints/32_Feb19_0154_stc_sagc_checkpoint.pth.tar'
# model = torch.load(path)
# num = sum(param.numel() for param in model.parameters())
# print(num)
                

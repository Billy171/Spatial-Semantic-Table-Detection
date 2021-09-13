import pickle
import json
import os
from torchvision.utils import save_image, make_grid
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from data.table_bank import build
from torch.utils.data import DataLoader
from model.loss import SetCriterion
from model.table_detection_model import TableDetectionModel
from args import MainArgs
from training import train_one_epoch, train_one_epoch2
from util.misc import my_collate

args = MainArgs(epochs=1, batch_size=4, only_data_subset=False,
                data_dir='/home/ubuntu/ds-volume/semantic-td/data/',
                output_dir='/home/ubuntu/ds-volume/semantic-td/data/ckpnt/')

device = torch.device(args.device)
model_cfg = args.model_cfg
crit_cfg = args.criterion_cfg
model = TableDetectionModel(model_cfg)
model.to(device)
criterion = SetCriterion(num_classes=crit_cfg.num_classes,
                         matcher=crit_cfg.matcher,
                         weight_dict=crit_cfg.weight_dict,
                         eos_coef=crit_cfg.eos_coef,
                         losses=crit_cfg.losses)
criterion.to(device)

param_dict = [{"params": [p for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad]}]
optimizer = torch.optim.AdamW(param_dict, lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
"""load the data"""
if args.only_data_subset:
    dataset_train = build('subset', args.data_dir)
    dataset_val = build('subset', args.data_dir)
else:
    dataset_train = build('train', args.data_dir)
    dataset_val = build('test', args.data_dir)
print(f"training size = {len(dataset_train)}, validation size = {len(dataset_val)}")




sampler_train = torch.utils.data.RandomSampler(dataset_train)
batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                               collate_fn=my_collate, num_workers=args.num_workers)

"""
i = 0
for samples, label in dataset_train:
    if i%1000 == 0:
      print('completed ',i)
    if samples['token_ids'].shape != torch.Size([512]):
        print('wrong size token_ids')
        print(samples['token_ids'].shape)
        print(i)
    if samples['bboxes'].shape != (512, 4):
        print('wrong size bboxes')
        print(samples['bboxes'].shape)
        print(i)
    if samples['image'].shape != (3, 224, 224):
        print('wrong size image')
        print(samples['image'].shape)
        print(i)
    i += 1
"""
i=0
for samples, label in data_loader_train:
    if i%1000 == 0:
        print('completed ',i)
    if samples['token_ids'].shape != (args.batch_size, 512):
        print('wrong size token_ids')
        print(samples['token_ids'].shape)
        print(i)
    if samples['bboxes'].shape != (args.batch_size, 512, 4):
        print('wrong size bboxes')
        print(samples['bboxes'].shape)
        print(i)
    if samples['image'].shape != (args.batch_size, 3, 224, 224):
        print('wrong size image')
        print(samples['image'].shape)
        print(i)
    i += 1



#DIR = '/mnt/d/thesis/data/train'
#print(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))

#with open("data/TableBank/train/examples1.txt", "rb") as f:
#    data = pickle.load(f)
#    n = len(data['token_ids'])
#    for i in range(n):
#        sample = {'token_ids': data['token_ids'][i],
#                  'bboxes': data['bboxes'][i],
#                  'image': data['image'][i].numpy().tolist(),
#                  'labels': data['labels'][i]}
#        print(type())
#        file_name = f"{path}/sample_{i}.json"
#        with open(file_name, "w") as f1:
#            json.dump(sample, f1)













"""
tensor1 = examples['image'][0]
tensor2 = examples['image'][1]


def save_img_from_tensor(tensor,name):
    img = torch.reshape(tensor, (3, 224, 224))
    img = img.permute(1, 2, 0)
    img = img.numpy()
    img = Image.fromarray(img)
    img.save(name+'fp.jpg')  # , format=format)


save_img_from_tensor(tensor1, 'img1')
save_img_from_tensor(tensor2, 'img2')
"""
"""    
img = torch.reshape(tensor1, (3, 224, 224))
img = img.permute(1, 2, 0)
img = img.numpy()
img = Image.fromarray(img)
img.save('fp.jpg')#, format=format)
"""

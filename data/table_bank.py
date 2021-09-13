import os

import numpy as np
from torch.utils.data import Dataset
import math
import pickle
import time
import json
import torch


class TableBankDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __len__(self):
        return len([name for name in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, name))])

    def __getitem__(self, idx):
        filename = f"{self.data_dir}/sample_{idx}.json"
        with open(filename, 'rb') as f:
            sample = json.load(f)
            #image = torch.tensor(np.array(sample['image']))
            image = torch.tensor(np.array(sample['image'])).reshape(3, 224, 224)
            label = {'labels': torch.tensor(sample['label']['labels']),
                     'boxes': torch.tensor(sample['label']['boxes'])}
            example = {'token_ids': torch.tensor(sample['token_ids']),
                       'bboxes': torch.tensor(sample['bboxes']),
                       'image': image}
        return example, label


def build(folder, dir):
    if folder == 'train':
        data_dir = dir+'train'
        #data_dir = '/mnt/d/thesis/data/train'
    elif folder == 'test':
        #data_dir = '/mnt/d/thesis/data/test'
        data_dir = dir + 'test'
    elif folder == 'subset':
        #data_dir = '/mnt/d/thesis/data/subset'
        data_dir = dir + 'subset'
    dataset = TableBankDataset(data_dir)
    return dataset








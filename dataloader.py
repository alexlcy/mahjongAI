# -*- coding: utf-8 -*-
# @FileName : dataloader.py
# @Project  : MAHJONG AI
# @Author   : WANG Jianxing
# @Time     : 2021/4/17
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from MajhongAI.mahjong.Serialization import offline_serialize


class MyDataSet(Dataset):
    def __init__(self, txt_file):
        with open('./MajhongAI/mahjong/raw_dataset.txt', 'r') as fp:
            self.data = fp.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


batch_size = 1
txt_file = 'raw_dataset.txt'
raw_ds = MyDataSet(txt_file)
raw_dl = DataLoader(raw_ds, batch_size=batch_size)
# for epoch in range(2):
for batch_idx, batch in enumerate(raw_dl):
    print("\nBatch = " + str(batch_idx))
    raw_data = batch
    print(raw_data[0])
    print(offline_serialize(raw_data[0]))
    # print(type(raw_data))
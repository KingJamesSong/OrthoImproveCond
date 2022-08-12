from data.base_dataset import BaseDataset
import os
import torch
from util.util import np_load
import pickle
import numpy as np
import cv2
import io
from PIL import Image
import torchvision as tv
import lmdb
import string


class LSUNDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        data_dir = opt.dataroot
        self.env = lmdb.open(data_dir, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        cache_file = "_cache_" + "".join(c for c in data_dir if c in string.ascii_letters)
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.keys, open(cache_file, "wb"))
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize((128, 128)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        #img = self.data[index]
        # to tensor
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
        #img = torch.from_numpy(img.transpose(2, 0, 1).astype('float32'))
        # normalize
        #img = img.div(127.5).sub(1)
        #img = img.mul(2).sub(1)
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])
        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        #img = Image.fromarray(img)
        img = self.transform(img)

        return  {'img':img, 'path':'',}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.length
import math
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import pandas as pd
import torch.nn.functional as F

#######################################################
#                  Create Dataset
#######################################################
DIC_NAME = "dic_name"
FOLDER_PATH = "folder_path"
worms_params = {
    # "batch_size": 2,  # the number of image per sample check if need to be changed -A: FOR NOW ITS FINE !
    "num_workers": 4,
    "image_max_size": (0, 0),
    "in_channels": 1,  # to make sure because its gray image 1 channel or maybe 3 from tensor size? -A: YES
    "num_classes": None  # to make sure we dont have classes because our label is an image -A: EXACTLY !
}

CSV_DATASET_PATH = r"C:\Users\yarin\PycharmProjects\GuidedProject\samples_database.csv"


class CustomDataset(Dataset):
    def __init__(self, image_paths, transform_dic={}):
        self.image_pad_param = self.get_max_image_size(image_paths.rsplit("\\", 1)[0] + "\\Database")
        self.image_paths = pd.read_csv(image_paths)
        self.transform_dic = transform_dic

    def __len__(self):
        return self.image_paths.shape[0]

    def __getitem__(self, index):
        metadata_sample = self.image_paths.loc[index]
        dic_image = torch.from_numpy(cv2.resize(cv2.imread(metadata_sample["sample_path"], 1), (384, 384))).permute(2,0,1).float()
        # dic_image = self.pad_sample(dic_image)

        # Add a resize operation to the transformation pipeline
        # dic_image = F.interpolate(dic_image, size=(3,384, 384))


        # return dic_image[None,:,:], dic_image[None,:,:], metadata_sample["sample_path"]
        # return dic_image[None,None,:[None,:,:],:],dic_image[None,None,:,:]
        return dic_image[None,:,:],dic_image[None,:,:]

    def pad_sample(self, dic):
        if self.image_pad_param[0] == 0 and self.image_pad_param[1] == 0:
            return dic
        new_image_height, new_image_width = self.image_pad_param
        old_image_height, old_image_width = dic.shape[0], dic.shape[1]
        w_pad = new_image_width - old_image_width
        h_pad = new_image_height - old_image_height

        if w_pad % 2 != 0:
            w1_pad = int(math.ceil(w_pad / 2)) - 1
            w2_pad = int(math.ceil(w_pad / 2))
        else:
            w1_pad = int(math.ceil(w_pad / 2))
            w2_pad = int(math.ceil(w_pad / 2))

        if h_pad % 2 != 0:
            h1_pad = int(math.ceil(h_pad / 2)) - 1
            h2_pad = int(math.ceil(h_pad / 2))
        else:
            h1_pad = int(math.ceil(h_pad / 2))
            h2_pad = int(math.ceil(h_pad / 2))

        pad_image = torch.nn.ConstantPad2d((w1_pad, w2_pad, h1_pad, h2_pad), 255)(dic.permute(2, 0, 1))
        pad_image = pad_image.permute(0, 1,2)

        return pad_image

    def get_max_image_size(self, data_path):
        max_h, max_w = 0, 0

        for dirpath, dirnames, filenames in os.walk(data_path):
            if filenames and dirpath.rsplit("\\", 1)[1] == 'Samples':
                for filename in filenames:
                    if filename.endswith('.png'):
                        filepath = os.path.join(dirpath, filename)
                        img = cv2.imread(filepath)
                        h, w = img.shape[:2]
                        if h > max_h:
                            max_h = h
                        if w > max_w:
                            max_w = w
        return max_h, max_w

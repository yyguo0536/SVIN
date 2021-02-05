# - *- coding: utf- 8 - *-
import re
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
#import scipy.io as sio
import SimpleITK as sitk
#import torchvision.transforms as tr
import random
from functools import reduce
import cv2
import numpy as np
from itertools import combinations
import random





class Paired_Subjects(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image96, image48, image24, classes, job, \
            spacing=None, crop=None, ratio=None, rotate=None, \
            include_slices=None):
        assert job in self.suitableJobs, 'not suitable jobs'
        self.job = job
        if job == 'seg':
            assert classes[0] == 0
        if job == 'cla':
            assert len(classes) > 1
        self.classes = classes
        self.image96 = image96
        self.image48 = image48
        self.image24 = image24
        # slices
        if include_slices is None:
            self.slice_indice = 10
        else:
            assert len(include_slices) > 0
            self.slice_indice = 10



        self.imgdata96=[]
        self.imgdata48=[]
        self.imgdata24=[]
        print(self.image96[0])

        for i in range(5):
            imgdata1 = sitk.ReadImage(self.image96[i])
            imgdata1 = sitk.GetArrayFromImage(imgdata1)
            imgdata1 = np.clip(imgdata1, -400, 600)
            imgdata1 = (imgdata1 - imgdata1.mean()) / (imgdata1.max() - imgdata1.min())
            self.imgdata96.append(imgdata1)

            imgdata1 = sitk.ReadImage(self.image48[i])
            imgdata1 = sitk.GetArrayFromImage(imgdata1)
            imgdata1 = np.clip(imgdata1, -400, 600)
            imgdata1 = (imgdata1 - imgdata1.mean()) / (imgdata1.max() - imgdata1.min())
            self.imgdata48.append(imgdata1)

            imgdata1 = sitk.ReadImage(self.image24[i])
            imgdata1 = sitk.GetArrayFromImage(imgdata1)
            imgdata1 = np.clip(imgdata1, -400, 600)
            imgdata1 = (imgdata1 - imgdata1.mean()) / (imgdata1.max() - imgdata1.min())
            self.imgdata24.append(imgdata1)

        

        self.loss_weights = [2.5, 1.5, 1.2, 1.0, 2.2, 1.5, 1.2, 3.5, 3, 5.0]


        temp_data = range(5)
        index_list = list(combinations(temp_data,2))
        
        self.index_list = index_list

    def __len__(self):
        return self.slice_indice

    def __getitem__(self, index):
        # image
        #if (random.randint(0,1)==0):
        fix_num = self.index_list[index][0]
        move_num = self.index_list[index][1]

        fix_image96 = self.imgdata96[fix_num]
        move_image96 = self.imgdata96[move_num]
        fix_image48 = self.imgdata48[fix_num]
        move_image48 = self.imgdata48[move_num]
        fix_image24 = self.imgdata24[fix_num]
        move_image24 = self.imgdata24[move_num]

        #fix_mask = self.maskdata[fix_num]
        #move_mask = self.maskdata[move_num]
                
        # one channel image
        fix_image96 = fix_image96.reshape((1,) + fix_image96.shape)
        fix_image96 = torch.from_numpy(fix_image96.astype(np.float32))
        move_image96 = move_image96.reshape((1,) + move_image96.shape)
        move_image96 = torch.from_numpy(move_image96.astype(np.float32))

        fix_image48 = fix_image48.reshape((1,) + fix_image48.shape)
        fix_image48 = torch.from_numpy(fix_image48.astype(np.float32))
        move_image48 = move_image48.reshape((1,) + move_image48.shape)
        move_image48 = torch.from_numpy(move_image48.astype(np.float32))

        fix_image24 = fix_image24.reshape((1,) + fix_image24.shape)
        fix_image24 = torch.from_numpy(fix_image24.astype(np.float32))
        move_image24 = move_image24.reshape((1,) + move_image24.shape)
        move_image24 = torch.from_numpy(move_image24.astype(np.float32))

        #fix_mask = fix_mask.reshape((1,) + fix_mask.shape)
        #fix_mask = torch.from_numpy(fix_mask.astype(np.float32))
        #move_mask = move_mask.reshape((1,) + move_mask.shape)
        #move_mask = torch.from_numpy(move_mask.astype(np.float32))

        # label
        
        return (fix_image96, fix_image48, fix_image24, move_image96, move_image48, move_image24, self.loss_weights[index])





class Test_Subjects(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image96, mask96, classes, job, \
            spacing=None, crop=None, ratio=None, rotate=None, \
            include_slices=None):
        assert job in self.suitableJobs, 'not suitable jobs'
        self.job = job
        if job == 'seg':
            assert classes[0] == 0
        if job == 'cla':
            assert len(classes) > 1
        self.classes = classes
        self.image96 = image96
        self.mask96 = mask96
        # slices
        if include_slices is None:
            self.slice_indice = 10
        else:
            assert len(include_slices) > 0
            self.slice_indice = 10
            


        self.imgdata96=[]
        self.mskdata96 = []
        print(self.image96[0])

        for i in range(5):
            imgdata1 = sitk.ReadImage(self.image96[i])
            imgdata1 = sitk.GetArrayFromImage(imgdata1)
            imgdata1 = np.clip(imgdata1, -400, 600)
            imgdata1 = (imgdata1 - imgdata1.mean()) / (imgdata1.max() - imgdata1.min())
            self.imgdata96.append(imgdata1)

            imgdata1 = sitk.ReadImage(self.mask96[i])
            imgdata1 = sitk.GetArrayFromImage(imgdata1)
            self.mskdata96.append(imgdata1)



        

        self.loss_weights = [2.5, 1.5, 1.2, 1.0, 2.2, 1.5, 1.2, 3.5, 3, 5.0]


        temp_data = range(5)
        index_list = list(combinations(temp_data,2))
        
        self.index_list = index_list

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        # image
        #if (random.randint(0,1)==0):
        fix_num = self.index_list[index][0]
        move_num = self.index_list[index][1]

        fix_image96 = self.imgdata96[fix_num]
        move_image96 = self.imgdata96[move_num]

        fix_mask96 = self.mskdata96[fix_num]
        move_mask96 = self.mskdata96[move_num]

        #fix_mask = self.maskdata[fix_num]
        #move_mask = self.maskdata[move_num]
                
        # one channel image
        fix_image96 = fix_image96.reshape((1,) + fix_image96.shape)
        fix_image96 = torch.from_numpy(fix_image96.astype(np.float32))
        move_image96 = move_image96.reshape((1,) + move_image96.shape)
        move_image96 = torch.from_numpy(move_image96.astype(np.float32))

        fix_mask96 = fix_mask96.reshape((1,) + fix_mask96.shape)
        fix_mask96 = torch.from_numpy(fix_mask96.astype(np.float32))
        move_mask96 = move_mask96.reshape((1,) + move_mask96.shape)
        move_mask96 = torch.from_numpy(move_mask96.astype(np.float32))


        #fix_mask = fix_mask.reshape((1,) + fix_mask.shape)
        #fix_mask = torch.from_numpy(fix_mask.astype(np.float32))
        #move_mask = move_mask.reshape((1,) + move_mask.shape)
        #move_mask = torch.from_numpy(move_mask.astype(np.float32))

        # label
        
        return (fix_image96,  move_image96, fix_mask96, move_mask96)



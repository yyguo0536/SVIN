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


class SlicesOfSubject_3d_norm(torch.utils.data.Dataset):
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
            self.slice_indice = 3
        else:
            assert len(include_slices) > 0
            self.slice_indice = 3
        


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
        #index_list = list(combinations(temp_data,2))
        
        self.index_list = [[0.0,1.0,4.0], [0.0,2.0,4.0], [0.0,3.0,4.0]]

    def __len__(self):
        return self.slice_indice

    def __getitem__(self, index):
        # image
        #if (random.randint(0,1)==0):
        ED_num = int(self.index_list[index][0])
        ES_num = int(self.index_list[index][2])
        inter_num = int(self.index_list[index][1])

        ED_image96 = self.imgdata96[ED_num]
        ES_image96 = self.imgdata96[ES_num]
        ED_image48 = self.imgdata48[ED_num]
        ES_image48 = self.imgdata48[ES_num]
        ED_image24 = self.imgdata24[ED_num]
        ES_image24 = self.imgdata24[ES_num]

        inter_image96 = self.imgdata96[inter_num]
        inter_image48 = self.imgdata48[inter_num]
        inter_image24 = self.imgdata24[inter_num]

        #fix_mask = self.maskdata[fix_num]
        #move_mask = self.maskdata[move_num]
                
        # one channel image
        ED_image96 = ED_image96.reshape((1,) + ED_image96.shape)
        ED_image96 = torch.from_numpy(ED_image96.astype(np.float32))
        ES_image96 = ES_image96.reshape((1,) + ES_image96.shape)
        ES_image96 = torch.from_numpy(ES_image96.astype(np.float32))
        inter_image96 = inter_image96.reshape((1,) + inter_image96.shape)
        inter_image96 = torch.from_numpy(inter_image96.astype(np.float32))

        ED_image48 = ED_image48.reshape((1,) + ED_image48.shape)
        ED_image48 = torch.from_numpy(ED_image48.astype(np.float32))
        ES_image48 = ES_image48.reshape((1,) + ES_image48.shape)
        ES_image48 = torch.from_numpy(ES_image48.astype(np.float32))
        inter_image48 = inter_image48.reshape((1,) + inter_image48.shape)
        inter_image48 = torch.from_numpy(inter_image48.astype(np.float32))

        ED_image24 = ED_image24.reshape((1,) + ED_image24.shape)
        ED_image24 = torch.from_numpy(ED_image24.astype(np.float32))
        ES_image24 = ES_image24.reshape((1,) + ES_image24.shape)
        ES_image24 = torch.from_numpy(ES_image24.astype(np.float32))
        inter_image24 = inter_image24.reshape((1,) + inter_image24.shape)
        inter_image24 = torch.from_numpy(inter_image24.astype(np.float32))

        #fix_mask = fix_mask.reshape((1,) + fix_mask.shape)
        #fix_mask = torch.from_numpy(fix_mask.astype(np.float32))
        #move_mask = move_mask.reshape((1,) + move_mask.shape)
        #move_mask = torch.from_numpy(move_mask.astype(np.float32))

        # label
        
        return (ED_image96, ED_image48, ED_image24, ES_image96, \
            ES_image48, ES_image24, inter_image96, inter_image48, \
            inter_image24, self.index_list[index][1])



class AnalysisOfSubject_3d_norm(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image96, classes, job, \
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
        # slices
        if include_slices is None:
            self.slice_indice = 3
        else:
            assert len(include_slices) > 0
            self.slice_indice = 3
        


        self.imgdata96=[]
        print(self.image96[0])
        self.patients = self.image96[0].split('/')[-3]

        for i in range(5):
            imgdata1 = sitk.ReadImage(self.image96[i])
            imgdata1 = sitk.GetArrayFromImage(imgdata1)
            imgdata1 = np.clip(imgdata1, -400, 600)
            imgdata1 = (imgdata1 - imgdata1.mean()) / (imgdata1.max() - imgdata1.min())
            self.imgdata96.append(imgdata1)

        

        self.loss_weights = [2.5, 1.5, 1.2, 1.0, 2.2, 1.5, 1.2, 3.5, 3, 5.0]


        temp_data = range(5)
        #index_list = list(combinations(temp_data,2))
        
        self.index_list = [[0.0,1.0,4.0], [0.0,2.0,4.0], [0.0,3.0,4.0]]

    def __len__(self):
        return self.slice_indice

    def __getitem__(self, index):
        # image
        #if (random.randint(0,1)==0):
        ED_num = int(self.index_list[index][0])
        ES_num = int(self.index_list[index][2])
        inter_num = int(self.index_list[index][1])

        ED_image96 = self.imgdata96[ED_num]
        ES_image96 = self.imgdata96[ES_num]

        inter_image96 = self.imgdata96[inter_num]

        #fix_mask = self.maskdata[fix_num]
        #move_mask = self.maskdata[move_num]
                
        # one channel image
        ED_image96 = ED_image96.reshape((1,) + ED_image96.shape)
        ED_image96 = torch.from_numpy(ED_image96.astype(np.float32))
        ES_image96 = ES_image96.reshape((1,) + ES_image96.shape)
        ES_image96 = torch.from_numpy(ES_image96.astype(np.float32))
        inter_image96 = inter_image96.reshape((1,) + inter_image96.shape)
        inter_image96 = torch.from_numpy(inter_image96.astype(np.float32))


        #fix_mask = fix_mask.reshape((1,) + fix_mask.shape)
        #fix_mask = torch.from_numpy(fix_mask.astype(np.float32))
        #move_mask = move_mask.reshape((1,) + move_mask.shape)
        #move_mask = torch.from_numpy(move_mask.astype(np.float32))

        # label
        
        return (ED_image96,ES_image96, \
            inter_image96, self.index_list[index][1], self.patients)


class AnalysisOfMask_3d(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image96, label96, classes, job, \
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
        self.label96 = label96
        # slices
        if include_slices is None:
            self.slice_indice = 3
        else:
            assert len(include_slices) > 0
            self.slice_indice = 3
        


        self.imgdata96 = []
        self.mskdata96 = []
        print(self.image96[0])
        self.patients = self.image96[0].split('/')[-3]

        for i in range(5):
            imgdata1 = sitk.ReadImage(self.image96[i])
            imgdata1 = sitk.GetArrayFromImage(imgdata1)
            imgdata1 = np.clip(imgdata1, -400, 600)
            imgdata1 = (imgdata1 - imgdata1.mean()) / (imgdata1.max() - imgdata1.min())
            self.imgdata96.append(imgdata1)
            imgdata1 = sitk.ReadImage(self.label96[i])
            imgdata1 = sitk.GetArrayFromImage(imgdata1)
            self.mskdata96.append(imgdata1)

        

        self.loss_weights = [2.5, 1.5, 1.2, 1.0, 2.2, 1.5, 1.2, 3.5, 3, 5.0]


        temp_data = range(5)
        #index_list = list(combinations(temp_data,2))
        
        self.index_list = [[0.0,1.0,4.0], [0.0,2.0,4.0], [0.0,3.0,4.0]]

    def __len__(self):
        return self.slice_indice

    def __getitem__(self, index):
        # image
        #if (random.randint(0,1)==0):
        ED_num = int(self.index_list[index][0])
        ES_num = int(self.index_list[index][2])
        inter_num = int(self.index_list[index][1])

        ED_image96 = self.imgdata96[ED_num]
        ES_image96 = self.imgdata96[ES_num]

        ED_mask96 = self.mskdata96[ED_num]
        ES_mask96 = self.mskdata96[ES_num]

        inter_image96 = self.imgdata96[inter_num]

        inter_mask96 = self.mskdata96[inter_num]

        

        #fix_mask = self.maskdata[fix_num]
        #move_mask = self.maskdata[move_num]
                
        # one channel image
        ED_image96 = ED_image96.reshape((1,) + ED_image96.shape)
        ED_image96 = torch.from_numpy(ED_image96.astype(np.float32))
        ES_image96 = ES_image96.reshape((1,) + ES_image96.shape)
        ES_image96 = torch.from_numpy(ES_image96.astype(np.float32))
        inter_image96 = inter_image96.reshape((1,) + inter_image96.shape)
        inter_image96 = torch.from_numpy(inter_image96.astype(np.float32))

        ED_mask96 = ED_mask96.reshape((1,) + ED_mask96.shape)
        ED_mask96 = torch.from_numpy(ED_mask96.astype(np.float32))
        ES_mask96 = ES_mask96.reshape((1,) + ES_mask96.shape)
        ES_mask96 = torch.from_numpy(ES_mask96.astype(np.float32))
        inter_mask96 = inter_mask96.reshape((1,) + inter_mask96.shape)
        inter_mask96 = torch.from_numpy(inter_mask96.astype(np.float32))


        #fix_mask = fix_mask.reshape((1,) + fix_mask.shape)
        #fix_mask = torch.from_numpy(fix_mask.astype(np.float32))
        #move_mask = move_mask.reshape((1,) + move_mask.shape)
        #move_mask = torch.from_numpy(move_mask.astype(np.float32))

        # label
        
        return (ED_image96,ES_image96, \
            inter_image96, ED_mask96, ES_mask96, inter_mask96, \
            self.index_list[index][1], self.patients)



class Slice3D_test_norm(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image96, classes, job, \
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
        # slices
        if include_slices is None:
            self.slice_indice = 3
        else:
            assert len(include_slices) > 0
            self.slice_indice = 3
        


        self.imgdata96=[]
        self.imgdata48=[]
        self.imgdata24=[]
        print(self.image96[0])


        imgdata1 = sitk.ReadImage(self.image96[0])
        imgdata1 = sitk.GetArrayFromImage(imgdata1)
        imgdata1 = np.clip(imgdata1, -400, 600)
        imgdata1 = (imgdata1 - imgdata1.mean()) / (imgdata1.max() - imgdata1.min())
        self.imgdata96.append(imgdata1)

        imgdata1 = sitk.ReadImage(self.image96[-1])
        imgdata1 = sitk.GetArrayFromImage(imgdata1)
        imgdata1 = np.clip(imgdata1, -400, 600)
        imgdata1 = (imgdata1 - imgdata1.mean()) / (imgdata1.max() - imgdata1.min())
        self.imgdata96.append(imgdata1)


        

        self.loss_weights = [2.5, 1.5, 1.2, 1.0, 2.2, 1.5, 1.2, 3.5, 3, 5.0]


        temp_data = range(5)
        #index_list = list(combinations(temp_data,2))
        

    def __len__(self):
        return self.slice_indice

    def __getitem__(self, index):
        # image
        #if (random.randint(0,1)==0):
        ED_num = 0
        ES_num = 1

        ED_image96 = self.imgdata96[ED_num]
        ES_image96 = self.imgdata96[ES_num]


        # one channel image
        ED_image96 = ED_image96.reshape((1,) + ED_image96.shape)
        ED_image96 = torch.from_numpy(ED_image96.astype(np.float32))
        ES_image96 = ES_image96.reshape((1,) + ES_image96.shape)
        ES_image96 = torch.from_numpy(ES_image96.astype(np.float32))
        
        
        return (ED_image96,  ES_image96)








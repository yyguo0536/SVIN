# A Spatiotemporal volumetric interpolation network for 4D dynamic medical image

This is our PyTorch implementation for 3D spatiotemporal motion(deformation) estimation and motion(deformation) guided volumetric interpolation. 

**Note**: The current framework works with PyTorch 0.4.

REQUIREMENTS
Python-2.7
CUDA
You must have an Nvidia Gpu on your system.
Pytorch-0.4
SimpleITK
Scipy
Pandas
CV2


OS
In our case, it is Ubuntu 16.04.

###### Implementations
The source code can directly use it. The detailed information as following:

Customize
We develop our framework based on GAN, and user can optionaly add a discriminator network to train the both networks. 

#### For Motion Field Estimation

```
cd ./motion-net
## For Training Stage: 
python train_motion.py --phase train --which_model_netG motion --model motion --checkpoints_dir ./save_motion/ --dataroot ./ --dataset_mode aligned --display_id 0 --gpu_ids 1 --batchSize 1
## For Testing Stage: 
python test_motion.py --phase test --which_model_netG motion --model test --checkpoints_dir ./save_motion/ --dataroot ./ --dataset_mode single --display_id 0 --gpu_ids 1 --batchSize 1 --net3d_dir_G ./.pth
```

#### For Senquential Volumetric Interpolation

```
cd ./interpolation
## For Training Stage: 
python train_4d.py --phase train --which_model_netG interpolation --model interpolation --checkpoints_dir ./save_interpolation/ --dataroot ./ --dataset_mode aligned --display_id 0 --gpu_ids 1 --batchSize 1
## For Testing Stage: 
python test_motion.py --phase test --which_model_netG interpolation --model test --checkpoints_dir ./save_interpolation/ --dataroot ./ --dataset_mode single --display_id 0 --gpu_ids 1 --batchSize 1 --net3d_dir_G ./.pth
```



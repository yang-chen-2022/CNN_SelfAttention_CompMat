
import torch

from utils.NN_warehouse import *
from utils.train_structure_warehouse import *
from utils.IOfcts import *

import matplotlib.pyplot as plt
import numpy as np

# import cv2
import imgaug.augmenters as iaa


inference = True


# ===============
# check GPU, cuda
# ===============

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# ==============
# Hyperparamters
# ==============
portion_trains = 0.9
num_epochs = 2000
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-5


# list of directories for stress/damage fields
lst_dir0 = list()
lst_idx_UC = list()

if inference==True: #unseen data
    shuffle = False
    #lst_dir0.append( r"../data/unseen/L0.05_vf0.25/h0.0002" )
    #lst_dir0.append( r"../data/unseen/L0.05_vf0.35/h0.0002" )
    lst_dir0.append( r"..\data\unseen\L0.05_vf0.45\h0.0002" ) #change '\' to '/' if running on Linux
    #lst_dir0.append( r"../data/unseen/L0.05_vf0.55/h0.0002" )
    dir_mesh = r"..\data\unseen\mesh"
    
else:  #train data (request through email: yc2634@bath.ac.uk)
    shuffle = True
    #lst_dir0.append( r"../data/L0.05_vf0.3/h0.0002" )
    #lst_dir0.append( r"../data/L0.05_vf0.4/h0.0002" )
    #lst_dir0.append( r"../data/L0.05_vf0.5/h0.0002" )
    #lst_dir0.append( r"../data/L0.05_vf0.6/h0.0002" )
    lst_dir0.append( r"..\data\L0.05_vf0.25\h0.0002" )
    #lst_dir0.append( r"../data/L0.05_vf0.35/h0.0002" )
    #lst_dir0.append( r"../data/L0.05_vf0.45/h0.0002" )
    lst_dir0.append( r"..\data\L0.05_vf0.55\h0.0002" )
    dir_mesh = r"..\data\mesh"



# data augmentation
transform = iaa.Sequential(
    [
        iaa.TranslateX(percent=(0.,0.99), mode="wrap"),
        iaa.TranslateY(percent=(0.,0.99), mode="wrap"),
    ]
)

# prepare the dataset
dataset = elas2crkDataset(dir_mesh, lst_dir0, LOADprefix='Load0.0', transform=transform, outputFields='elas2crk')

numDAT = len(dataset)


idx = 1
mesh, energy, load, field = dataset[idx]
fig,ax = plt.subplots(1,3);
ax[0].imshow(mesh[0]); ax[0].axis('off'); ax[0].set_title('RVE geometry')
ax[1].imshow(energy[0]); ax[1].axis('off'); ax[1].set_title('Elastic energy')
ax[2].imshow(field[0],cmap='rainbow'); ax[2].axis('off'); ax[2].set_title('Damage field')
plt.show()


# ==================
# split train / test
# ==================
if inference == False:
    num_trains = int(len(dataset) * portion_trains)
    num_tests = len(dataset) - num_trains
    train_set, test_set = torch.utils.data.random_split(dataset, [num_trains, num_tests])
    
    loaders = {
        'train': DataLoader(dataset=train_set, num_workers=4, batch_size=batch_size, shuffle=shuffle),
        'test': DataLoader(dataset=test_set, num_workers=4, batch_size=batch_size, shuffle=shuffle),
    }



# =======================
# set training parameters
# =======================

init_epoch = 0

if inference==True:
    checkpoint_path = "../models/elas2crk/check.pt"
    best_model_path = "../models/elas2crk/best.pt"
    dir_tb = '../runs/elas2crk/log'
else:
    checkpoint_path = "../models/elas2crk/mycheck.pt"
    best_model_path = "../models/elas2crk/mybest.pt"
    dir_tb = '../runs/elas2crk/mylog'


net = NN_elas2crk_SA()

if use_cuda:
    net = net.cuda()

reconLoss = nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)


# ===============
# train the model
# ===============
if inference == False:
    trained_model = train_elas2crk(net, init_epoch, num_epochs, np.Inf, loaders,
                                         optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb)



# =============
# visualisation
# =============

# create a model
model = NN_elas2crk_SA()


# load the saved checkpoint
model, optimizer, start_epoch, valid_loss_min = load_ckp(best_model_path, model, optimizer)


# ===================================
from utils.inference import *
import time
import matplotlib.gridspec as gridspec

isample = 1

t0 = time.time()
mesh, energy, load, field = dataset[isample]
mesh = torch.unsqueeze(mesh, 0)
energy = torch.unsqueeze(energy, 0)
load = torch.unsqueeze(load, 0)
field = torch.unsqueeze(field, 0)
#prediction
pred, loss = inference_elas2crk(model, reconLoss, energy, load, field, use_cuda)
print(' inference time: {:.3e} s'.format(time.time()-t0), ' using ', str(device) )


# visu cracks
plt.figure(figsize = (1,4))
gs1 = gridspec.GridSpec(1,4)
gs1.update(wspace=0.01, hspace=0.01, left=0.01, bottom=0.01, top=0.5, right=0.99)
ax1 = plt.subplot(gs1[0]); im=ax1.imshow(mesh[0,0].cpu().T); plt.axis('off'); ax1.set_title('RVE')
ax1 = plt.subplot(gs1[1]); im=ax1.imshow(energy[0,0].cpu().T, vmax=0.5, cmap='jet'); plt.axis('off'); ax1.set_title('Energy')
ax1 = plt.subplot(gs1[2]); im=ax1.imshow(field[0,0].cpu().T, vmin=0, vmax=1, cmap='rainbow'); plt.axis('off'); ax1.set_title('Damage')
ax1 = plt.subplot(gs1[3]); im=ax1.imshow(pred[0,0].cpu().T, vmin=0, vmax=1, cmap='rainbow'); plt.axis('off'); ax1.set_title('Damage (CNN)')
plt.show()

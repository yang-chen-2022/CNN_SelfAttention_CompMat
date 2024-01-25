
import torch

from utils.NN_warehouse import *
from utils.train_structure_warehouse import *

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
    lst_dir0.append( r"..\data\unseen\L0.05_vf0.45\h0.0002" )
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
#if inference:
#    transform = None

# prepare the dataset
dataset = stepDataset(dir_mesh, lst_dir0, LOADprefix='Load0.0', transform=transform, outputFields='stress', varLOAD=False, step=10)
numDAT = len(dataset)

idx = 2
mesh, load, field = dataset[idx]
fig,ax = plt.subplots(1,2)
ax[0].imshow(mesh[0]); ax[0].axis('off'); ax[0].set_title('RVE geometry')
ax[1].imshow(field[0], cmap='jet'); ax[1].axis('off'); ax[1].set_title('S11')
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
    checkpoint_path = "../models/mesh2stress/check.pt"
    best_model_path = "../models/mesh2stress/best.pt"
    dir_tb = '../runs/mesh2stress/log'
else:
    checkpoint_path = "../models/mesh2stress/mycheck.pt"
    best_model_path = "../models/mesh2stress/mybest.pt"
    dir_tb = '../runs/mesh2stress/mylog'

net = NN_mesh2stress3_SA_2()

if use_cuda:
    net = net.cuda()

reconLoss = nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)


# ===============
# train the model
# ===============

if init_epoch > 0:
    # load the saved checkpoint
    net, optimizer, start_epoch, valid_loss_min = load_ckp(best_model_path, net, optimizer)

if inference == False:
    trained_model = train_mesh2stress_2(net, init_epoch, num_epochs, np.Inf, loaders,
                                         optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb, num_field=3)



# =============
# visualisation
# =============

# create a model
model = NN_mesh2stress3_SA_2() #using RVE as input

# load the saved checkpoint
model, optimizer, start_epoch, valid_loss_min = load_ckp(best_model_path, model, optimizer, device)

# ===================================
from utils.inference import *
import time
import matplotlib.gridspec as gridspec

isample = 2

t0 = time.time()
#isample = 123
mesh, load, field = dataset[isample]
mesh = torch.unsqueeze(mesh, 0)
load = torch.unsqueeze(load, 0)
field = torch.unsqueeze(field, 0)
pred, loss = inference_mesh2stress_2(model, reconLoss, mesh, load, field, use_cuda)
print(' inference time: {:.3e} s'.format(time.time()-t0), 'using '+str(device))

    

# visu
ibatch = 0
v00, v01 = field[ibatch,0].min().numpy(), field[ibatch,0].max().numpy()
v10, v11 = field[ibatch,1].min().numpy(), field[ibatch,1].max().numpy()
v20, v21 = field[ibatch,2].min().numpy(), field[ibatch,2].max().numpy()
v0 = [v00, v10, v20]
v1 = [v01, v11, v21]

plt.figure(figsize = (2, 4))
gs1 = gridspec.GridSpec(2, 4)
gs1.update(wspace=0.01, hspace=0.01, left=0.01, bottom=0.01, top=1, right=0.99)
for i in range(8):
    ax1 = plt.subplot(gs1[i])
    plt.axis('off')
    if i==0:
        im=ax1.imshow(mesh[ibatch,0].cpu().T)
    elif i<4:
        im=ax1.imshow(field[ibatch,i-1].cpu().T, cmap='jet', vmin=v0[i-1], vmax=v1[i-1])
    elif i==4:
        continue
    else:
        im=ax1.imshow(pred[ibatch,i-5].cpu().T, cmap='jet', vmin=v0[i-5], vmax=v1[i-5])
    #plt.colorbar(im,ax=ax1,location='bottom', fraction=0.047)

plt.show()



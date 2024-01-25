import numpy as np
import torch
from utils.train_structure_warehouse import *


def inference_mesh2stress_2(model, reconLoss, mesh, load, field, use_cuda=True):
    #use_cuda = torch.cuda.is_available()
    
    if use_cuda:
        mesh = mesh.cuda().float() / 3.
        load = load.cuda().float()
        field = field.cuda().float()
    else:
        mesh = mesh.cpu().float() / 3.
        load = load.cpu().float()
        field = field.cpu().float()
    
    #
    model.eval()
    with torch.no_grad():
        pred = model(mesh)
        #loss = reconLoss(pred, torch.unsqueeze(field[:,0], 1))
        loss = reconLoss(pred, field)
    
    return pred, loss



def inference_elas2crk(model, reconLoss, energy, load, field, use_cuda=True):
    #use_cuda = torch.cuda.is_available()

    if use_cuda:
        load = load.cuda().float()
        field = field.cuda().float()
        energy = energy.cuda().float()
    else:
        load = load.cpu().float()
        field = field.cpu().float()
        energy = energy.cpu().float()

    #
    model.eval()
    with torch.no_grad():
        pred = model(energy)
        loss = reconLoss(pred, field)

    return pred, loss
	

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torchvision
import os
import shutil
from utils.IOfcts import *
from tqdm import tqdm


###############################################################################
def train_mesh2stress_2(net, init_epoch, num_epochs, valid_loss_min_input, loaders,
                        optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb=None, varLOAD=False, num_field=1):
    
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    # write for tensorboard visualisation (don't run this when degugging)
    if dir_tb is not None:
        if os.path.exists(dir_tb):  shutil.rmtree(dir_tb)
        writer = SummaryWriter(dir_tb)
        step0, step1 = 0, 0

    # #debugger
    # (mesh, load, strain) = next(iter(loaders["train"]))

    for epoch in range(init_epoch, num_epochs):

        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # ############# #
        # train the net #
        # ############# #
        net.train()
        for batch_idx, (mesh, load, stress) in enumerate(tqdm(loaders['train']), 0):

            # move to GPU
            if use_cuda:
                mesh = mesh.cuda().float() / 3.
                load = load.cuda().float()
                stress = stress.cuda().float()

            # forward pass: compute the output
            pred = net(mesh)

            # calculate the batch loss
            if num_field == 1:
                loss = reconLoss(pred, torch.unsqueeze(stress[:,0], 1))
            elif num_field == 3:
                loss = reconLoss(pred, stress)
                
            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with w.r. to model parameters
            loss.backward()

            # perform a single optimisation step (parameter update)
            optimizer.step()

            # record the average training loss (comment this out when degugging)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # tensorboard output (comment this out when debugging)
            if dir_tb is not None:
                writer.add_scalar('Training loss', loss, global_step=step0)
                step0 += 1

            # #debugger - overfit deliberately
            # net.eval()
            # with torch.no_grad():
            #     # pred = net(mParamMAP[:, :2], torch.unsqueeze(load[:, 0], 1))
            #     pred = net(mParamMAP[:, :2])
            #     loss1 = lossPhaseField_Energy(pred, strain, mParamMAP)
            # print('epoch'+str(epoch)+', train loss:'+str(loss.item())+', eval loss:'+str(loss1.item()))

        if dir_tb is not None:
            writer.add_scalar('Train loss vs epoch', train_loss, global_step=epoch)
        
        # ################ #
        # validate the net #
        # ################ #
        net.eval()
        with torch.no_grad():

            for batch_idx, (mesh, load, stress) in enumerate(tqdm(loaders['test']), 0):
                
                # move to GPU
                if use_cuda:
                    mesh = mesh.cuda().float() / 3.
                    load = load.cuda().float()
                    stress = stress.cuda().float()

                # forward pass: compute the output
                pred = net(mesh)

                # calculate the batch loss
                if num_field == 1:
                    loss = reconLoss(pred, torch.unsqueeze(stress[:,0], 1))
                elif num_field == 3:
                    loss = reconLoss(pred, stress)

                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                # tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss', loss, global_step=step1)
                    step1 += 1
            
            if dir_tb is not None:
                writer.add_scalar('Valid loss vs epoch', valid_loss, global_step=epoch)

        # calculate average losses
        # train_loss = train_loss/len(loaders['train'].dataset)
        # valid_loss = valid_loss/len(loaders['test'].dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        print('-' * 60)

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

    # return trained model
    return net






###############################################################################
def train_elas2crk(net, init_epoch, num_epochs, valid_loss_min_input, loaders,
                        optimizer, reconLoss, use_cuda, checkpoint_path, best_model_path, dir_tb=None, varLOAD=False):


    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    # write for tensorboard visualisation (don't run this when degugging)
    if dir_tb is not None:
        if os.path.exists(dir_tb):  shutil.rmtree(dir_tb)
        writer = SummaryWriter(dir_tb)
        step0, step1 = 0, 0

    # #debugger
    # (mesh, load, strain) = next(iter(loaders["train"]))

    for epoch in range(init_epoch, num_epochs):

        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # ############# #
        # train the net #
        # ############# #
        net.train()
        for batch_idx, (_, energy, load, field) in enumerate(tqdm(loaders['train']), 0):

            # move to GPU
            if use_cuda:
                load = load.cuda().float()
                energy = energy.cuda().float()
                field = field.cuda().float()


            # forward pass: compute the output
            pred = net(energy)

            # calculate the batch loss
            loss = reconLoss(pred, field)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with w.r. to model parameters
            loss.backward()

            # perform a single optimisation step (parameter update)
            optimizer.step()

            # record the average training loss (comment this out when degugging)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # tensorboard output (comment this out when debugging)
            if dir_tb is not None:
                writer.add_scalar('Training loss', loss, global_step=step0)
                step0 += 1

            # #debugger - overfit deliberately
            # net.eval()
            # with torch.no_grad():
            #     # pred = net(mParamMAP[:, :2], torch.unsqueeze(load[:, 0], 1))
            #     pred = net(mParamMAP[:, :2])
            #     loss1 = lossPhaseField_Energy(pred, strain, mParamMAP)
            # print('epoch'+str(epoch)+', train loss:'+str(loss.item())+', eval loss:'+str(loss1.item()))
        
        if dir_tb is not None:
            writer.add_scalar('Train loss vs epoch', train_loss, global_step=epoch)
        

        # ################ #
        # validate the net #
        # ################ #
        net.eval()
        with torch.no_grad():

            for batch_idx, (_, energy, load, field) in enumerate(tqdm(loaders['test']), 0):

                # move to GPU
                if use_cuda:
                    energy = energy.cuda().float()
                    load = load.cuda().float()
                    field = field.cuda().float()


                # forward pass: compute the output
                pred = net(energy)

                # calculate the batch loss
                loss = reconLoss(pred, field)

                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                # tensorboard output
                if dir_tb is not None:
                    writer.add_scalar('Validation loss', loss, global_step=step1)
                    step1 += 1
                    
            if dir_tb is not None:
                writer.add_scalar('Valid loss vs epoch', valid_loss, global_step=epoch)

        # calculate average losses
        # train_loss = train_loss/len(loaders['train'].dataset)
        # valid_loss = valid_loss/len(loaders['test'].dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        print('-' * 60)

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

    # return trained model
    return net

    
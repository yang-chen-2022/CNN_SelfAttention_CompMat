import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import torch
import shutil
import re
import os


# =====================
# some helper functions
# =====================
def vtkFieldReader(vtk_name, fieldName='tomo_Volume'):
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(vtk_name)
    reader.Update()
    data = reader.GetOutput()
    dim = data.GetDimensions()
    siz = list(dim)
    siz = [i - 1 for i in siz]
    mesh = vtk_to_numpy(data.GetCellData().GetArray(fieldName))
    return mesh.reshape(siz, order='F')

def read_macroStressStrain(fname):
    with open(fname) as f:
        lines = f.readlines()
    data = list()
    for line in lines[6:]:
        data.append( [float(num) for num in line.split()] )
    return np.array(data)


def registerFileName(lst_stress=None, lst_strain=None, lst_damage=None, fprefix=None, loadstep=None, zeroVTK=False):
    if zeroVTK is False:
        if lst_stress is not None:
            for key in lst_stress:
                lst_stress[key].append(fprefix + '_' + key + '_' + str(loadstep) + '.vtk')

        if lst_strain is not None:
            for key in lst_strain:
                lst_strain[key].append(fprefix + '_' + key + '_' + str(loadstep) + '.vtk')

        if lst_damage is not None:
            for key in lst_damage:
                lst_damage[key].append(fprefix + '_' + key + '_' + str(loadstep) + '.vtk')
    else:

        zeroVTKlocation = 'zerovtk.vtk'

        if lst_stress is not None:
            for key in lst_stress:
                lst_stress[key].append(zeroVTKlocation)

        if lst_strain is not None:
            for key in lst_strain:
                lst_strain[key].append(zeroVTKlocation)

        if lst_damage is not None:
            for key in lst_damage:
                lst_damage[key].append(zeroVTKlocation)


def init_dict_StressStrainDamage():
    dict_stress = {'sig1': list(),
                   'sig2': list(),
                   'sig4': list()}
    dict_strain = {'def1': list(),
                   'def2': list(),
                   'def4': list()}
    dict_damage = {'M1_varInt1': list(),
                   'M2_varInt1': list()}

    return dict_stress, dict_strain, dict_damage

def vtk_field_name(key):
    if key == 'sig1':
        return 'Sig_1'
    elif key == 'sig2':
        return 'Sig_2'
    elif key== 'sig4':
        return 'Sig_4'
    elif key == 'def1':
        return 'Def_1'
    elif key == 'def2':
        return 'Def_2'
    elif key == 'def4':
        return 'Def_4'
    elif key== 'M1_varInt1':
        return 'M1_varInt1'
    elif key == 'M2_varInt1':
        return 'M2_varInt1'
    else:
        assert "key unknown, sorry"



import matplotlib.pyplot as plt
def plot_macro_vtksteps(data_macro, idx, Ieps=7, Isig=1):
    plt.plot(data_macro[:,Ieps], data_macro[:,Isig])
    plt.plot(data_macro[idx[:]-1,Ieps], data_macro[idx[:]-1,Isig], 'o')

    plt.show()


class meshvtkDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.files = datasets.utils.list_files(root_dir, ".vtk")
        self.files = sorted(self.files, key=lambda x: float(re.search('iUC(\d+)', x).group(1)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir + '/' + self.files[idx]
        mesh = vtkFieldReader(img_name, 'tomo_Volume')
        mesh = (mesh - 1) * 255

        # # multi-class
        # if np.unique(mesh).min()<=0:
        #     raise ValueError('all labels in the VTK mesh must be larger than 0')
        # label = np.array(mesh)
        # for idx in np.unique(mesh):
        #     label = np.append(label, (mesh==idx), axis=-1)
        # mesh = label[:, :, 1:]

        if self.transform:
            mesh = self.transform(mesh)

        return mesh, mesh




# =======================================================================================================
# import data - stress/damage at each individual load step, for new data with a smaller dimension (L0.05)
# =======================================================================================================

class stepDataset(Dataset):
    def __init__(self, dir_mesh,
                       lst_dir0 = None,
                       LOADprefix = 'Load0.0',
                       transform = None,
                       outputFields = 'damage',
                       varLOAD = False,
                       step = None):

        self.transform = transform

        # create empty lists for input / output
        in_MESH = list()
        in_LOAD = list()

        if outputFields == 'damage':
            _, _, out_field = init_dict_StressStrainDamage()
        elif outputFields == 'stress':
            out_field, _, _ = init_dict_StressStrainDamage()
        elif outputFields == 'strain':
            _, out_field, _ = init_dict_StressStrainDamage()

        # loop over directories - different vf, different UC, vp
        keyword = 'L0.05_vf'
        for dir0 in lst_dir0:  #different vf
            VF = re.search(f"{keyword}.*?(\d+\.\d+)", dir0).group(1)

            for dir1 in os.listdir(dir0):  #different UC, vp

                # initial geometry - mesh
                iuc = re.findall(r"\d+", dir1)[0]
                #img_name = dir_mesh + '/' + keyword + VF + '/' + 'h0.0002' + '/' + dir1 + '.vtk'
                img_name = os.path.join(dir_mesh, keyword + VF, 'h0.0002', dir1 + '.vtk')

                # indices of vtk-steps
                #tmp = datasets.utils.list_files(dir0 + '/' + dir1, ".vtk")
                tmp = datasets.utils.list_files(os.path.join(dir0, dir1), ".vtk")
                idx = list()
                for tmpi in tmp:
                    idx.append([int(num) for num in re.findall(r"(\d+).vtk", tmpi)])
                idx = np.unique(idx)

                # stress-strain curves -> load
                #file_macro = datasets.utils.list_files(dir0 + '/' + dir1, ".std")[0]
                file_macro = datasets.utils.list_files(os.path.join(dir0, dir1), ".std")[0]
                #data_macro = read_macroStressStrain(dir0 + '/' + dir1 + '/' + file_macro)
                data_macro = read_macroStressStrain(os.path.join(dir0, dir1, file_macro))

                # output: damage / stress
                if varLOAD == True:
                    for loadstep in idx[3:]:
                        if outputFields == 'damage':
                            registerFileName(lst_damage = out_field,
                                         #fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         fprefix = os.path.join(dir0, dir1, LOADprefix),
                                         loadstep = loadstep)
                        elif outputFields == 'stress':
                            registerFileName(lst_stress = out_field,
                                         #fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         fprefix = os.path.join(dir0, dir1, LOADprefix),
                                         loadstep = loadstep)
                        elif outputFields == 'strain':
                            registerFileName(lst_strain = out_field,
                                         #fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         fprefix = os.path.join(dir0, dir1, LOADprefix),
                                         loadstep = loadstep)

                        # mesh
                        in_MESH.append(img_name)

                        # load - 6 components: Exx, Eyy, Ezz, Exy, Exz, Ezz
                        in_LOAD.append(data_macro[loadstep-1, 7:13])
                        
                else:
                
                    if step == None:
                        loadstep = idx[-1]
                    else:
                        loadstep = step
                        
                    if outputFields == 'damage':
                        registerFileName(lst_damage = out_field,
                                     #fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                     fprefix = os.path.join(dir0, dir1, LOADprefix),
                                     loadstep = loadstep)
                    elif outputFields == 'stress':
                        registerFileName(lst_stress = out_field,
                                         #fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         fprefix = os.path.join(dir0, dir1, LOADprefix),
                                         loadstep = loadstep)
                    elif outputFields == 'strain':
                        registerFileName(lst_strain = out_field,
                                         #fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         fprefix = os.path.join(dir0, dir1, LOADprefix),
                                         loadstep = loadstep)

                    # mesh
                    in_MESH.append(img_name)

                    # load - 6 components: Exx, Eyy, Ezz, Exy, Exz, Ezz
                    in_LOAD.append(data_macro[loadstep-1, 7:13])

        self.in_MESH = in_MESH
        self.in_LOAD = torch.from_numpy(np.stack(in_LOAD))
        self.out_field = out_field
        self.outputFields = outputFields

    def __len__(self):
        return len(self.in_MESH)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # input - mesh
        mesh = vtkFieldReader(self.in_MESH[idx], 'tomo_Volume')

        # output - damage / stress
        if self.outputFields=='damage':
            output = np.zeros((251, 251, 1))
            for key in self.out_field:
                output += vtkFieldReader(self.out_field[key][idx], fieldName=vtk_field_name(key))

        elif self.outputFields=='stress':
            output = np.zeros((251, 251, 0))
            for key in self.out_field:
                output = np.concatenate( (output, vtkFieldReader(self.out_field[key][idx],
                                                                 fieldName=vtk_field_name(key))), axis=2 )

        elif self.outputFields=='strain':
            output = np.zeros((251, 251, 0))
            for key in self.out_field:
                output = np.concatenate( (output, vtkFieldReader(self.out_field[key][idx],
                                                                 fieldName=vtk_field_name(key))), axis=2 )

        # transform - data augmentation
        if self.transform:

            transform_det = self.transform.to_deterministic()
            mesh = transform_det(image=mesh)
            output = transform_det(image=output)

        # to Tensor
        mesh = torch.moveaxis(torch.from_numpy(mesh), -1, 0)
        output = torch.moveaxis(torch.from_numpy(output), -1, 0)

        return mesh, self.in_LOAD[idx], output





# =========================================================================
# import data - stress/strain at elastic step, damage at final step (L0.05)
# =========================================================================

class elas2crkDataset(Dataset):
    def __init__(self, dir_mesh,
                       lst_dir0 = None,
                       LOADprefix = 'Load0.0',
                       transform = None,
                       outputFields = 'damage',
                       varLOAD = False,
                       step = None,
                       PINN = False):

        self.transform = transform

        # create empty lists for input / output
        in_MESH = list()
        in_LOAD = list()
        bad_results = list()

        if outputFields == 'damage':
            _, _, out_field = init_dict_StressStrainDamage()
        elif outputFields == 'stress':
            out_field, _, _ = init_dict_StressStrainDamage()
        elif outputFields == 'strain':
            _, out_field, _ = init_dict_StressStrainDamage()
        elif outputFields == 'elas2crk':
            in_stress, in_strain, out_field = init_dict_StressStrainDamage()
            if PINN == True:
                _, out_strain, _ = init_dict_StressStrainDamage()

        # loop over directories - different vf, different UC, vp
        keyword = 'L0.05_vf'
        for dir0 in lst_dir0:  #different vf
            VF = re.search(f"{keyword}.*?(\d+\.\d+)", dir0).group(1)
            if dir0.find('_CMC/') != -1:
                VF = VF + '_CMC'
                
            for dir1 in os.listdir(dir0):  #different UC, vp

                # initial geometry - mesh
                iuc = re.findall(r"\d+", dir1)[0]
                img_name = dir_mesh + '/' + keyword + VF + '/' + 'h0.0002' + '/' + dir1 + '.vtk'

                # indices of vtk-steps
                tmp = datasets.utils.list_files(dir0 + '/' + dir1, ".vtk")
                idx = list()
                for tmpi in tmp:
                    idx.append([int(num) for num in re.findall(r"(\d+).vtk", tmpi)])
                idx = np.unique(idx)

                # stress-strain curves -> load
                file_macro = datasets.utils.list_files(dir0 + '/' + dir1, ".std")[0]
                data_macro = read_macroStressStrain(dir0 + '/' + dir1 + '/' + file_macro)


                #sanity check
                if idx.max() > data_macro.shape[0]:
                    bad_results.append( dir0 + '/' + dir1 )
                    continue
                    

                # output: damage / stress
                if varLOAD == True:
                    for loadstep in idx[3:]:
                        if outputFields == 'damage':
                            registerFileName(lst_damage = out_field,
                                         fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         loadstep = loadstep)
                        elif outputFields == 'stress':
                            registerFileName(lst_stress = out_field,
                                         fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         loadstep = loadstep)
                        elif outputFields == 'strain':
                            registerFileName(lst_strain = out_field,
                                         fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         loadstep = loadstep)

                        # mesh
                        in_MESH.append(img_name)

                        # load - 6 components: Exx, Eyy, Ezz, Exy, Exz, Ezz
                        in_LOAD.append(data_macro[loadstep-1, 7:13])
                        
                else:
                
                    if step == None:
                        loadstep = idx[-1]
                    else:
                        loadstep = step
                        
                    if outputFields == 'damage':
                        registerFileName(lst_damage = out_field,
                                     fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                     loadstep = loadstep)
                    elif outputFields == 'stress':
                        registerFileName(lst_stress = out_field,
                                         fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         loadstep = loadstep)
                    elif outputFields == 'strain':
                        registerFileName(lst_strain = out_field,
                                         fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         loadstep = loadstep)
                    elif outputFields == 'elas2crk':
                        registerFileName(lst_stress = in_stress,
                                     fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                     loadstep = idx[0])
                        registerFileName(lst_strain = in_strain,
                                     fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                     loadstep = idx[0])
                        registerFileName(lst_damage = out_field,
                                     fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                     loadstep = loadstep)
                        if PINN == True:
                            registerFileName(lst_strain = out_strain,
                                         fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         loadstep = loadstep)
                            

                    # mesh
                    in_MESH.append(img_name)

                    # load - 6 components: Exx, Eyy, Ezz, Exy, Exz, Ezz
                    in_LOAD.append(data_macro[loadstep-1, 7:13])

        self.in_MESH = in_MESH
        self.in_LOAD = torch.from_numpy(np.stack(in_LOAD))
        self.out_field = out_field
        self.outputFields = outputFields
        self.PINN = PINN
        if outputFields == 'elas2crk':
            self.in_stress = in_stress
            self.in_strain = in_strain
            if PINN == True:
                self.out_strain = out_strain
            
        self.badresults = bad_results

    def __len__(self):
        return len(self.in_MESH)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # input - mesh
        mesh = vtkFieldReader(self.in_MESH[idx], 'tomo_Volume')

        # output - damage / stress
        if self.outputFields=='damage':
            output = np.zeros((251, 251, 1))
            for key in self.out_field:
                output += vtkFieldReader(self.out_field[key][idx], fieldName=vtk_field_name(key))

        elif self.outputFields=='stress':
            output = np.zeros((251, 251, 0))
            for key in self.out_field:
                output = np.concatenate( (output, vtkFieldReader(self.out_field[key][idx],
                                                                 fieldName=vtk_field_name(key))), axis=2 )

        elif self.outputFields=='strain':
            output = np.zeros((251, 251, 0))
            for key in self.out_field:
                output = np.concatenate( (output, vtkFieldReader(self.out_field[key][idx],
                                                                 fieldName=vtk_field_name(key))), axis=2 )
                                                                 
        elif self.outputFields=='elas2crk':
            stress = np.zeros((251, 251, 0))
            for key in self.in_stress:
                stress = np.concatenate( (stress, vtkFieldReader(self.in_stress[key][idx],
                                                                 fieldName=vtk_field_name(key))), axis=2 )
            strain = np.zeros((251, 251, 0))
            for key in self.in_strain:
                strain = np.concatenate( (strain, vtkFieldReader(self.in_strain[key][idx],
                                                                 fieldName=vtk_field_name(key))), axis=2 )
            energy = np.zeros((251, 251, 1))
            energy[:,:,0] = strain[:,:,0] * stress[:,:,0] + strain[:,:,1] * stress[:,:,1] + 2* strain[:,:,2] * stress[:,:,2]
            energy = energy/energy.max()

            output = np.zeros((251, 251, 1))
            for key in self.out_field:
                output += vtkFieldReader(self.out_field[key][idx], fieldName=vtk_field_name(key))
                
            if self.PINN == True:
                for key in self.out_strain:
                    output = np.concatenate( (output, vtkFieldReader(self.out_strain[key][idx],
                                                                 fieldName=vtk_field_name(key))), axis=2 )
        

        # transform - data augmentation
        if self.transform:

            transform_det = self.transform.to_deterministic()
            mesh = transform_det(image=mesh)
            output = transform_det(image=output)
            
            if self.outputFields=='elas2crk':
                energy = transform_det(image=energy)

        # to Tensor
        mesh = torch.moveaxis(torch.from_numpy(mesh), -1, 0)
        output = torch.moveaxis(torch.from_numpy(output), -1, 0)
        if self.outputFields=='elas2crk':
            energy = torch.moveaxis(torch.from_numpy(energy), -1, 0)

        return mesh, energy, self.in_LOAD[idx], output










# ======================================
# functions for saving and loading model
# ======================================

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer, device=None):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load check point
    checkpoint = torch.load(checkpoint_fpath, map_location=device)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()




# ==============================
# learning curve post-processing
# ==============================
# https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py


import traceback

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Extraction function
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data



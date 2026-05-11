#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# base classes for QTI_MLP: QTI_Dataset, QTI_MLP, QTIb_MLP
# corresponding methods and functions
# def of train and test loop
# why do the functions defined outside import automatically?
#-------------------------------------------------------------------------------------
# imports
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.masked import masked_tensor

# Disable prototype warnings and such
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

import nibabel as nib
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import hist

#-------------------------------------------------------------------------------------
# plot settings:
SMALL_SIZE = 8

matplotlib.rc('font', size=SMALL_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
matplotlib.rc('legend', markerscale=SMALL_SIZE)


#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# QTI_Dataset class
#-------------------------------------------------------------------------------------
# prepare QTI data and fit results for regression prediction
class QTI_Dataset(Dataset):
    #-------------------------------------------------------------------------------------
    # Constructor:
    def __init__(self, nii_path, scalar_invars_path=None, mask_path = None, slice_ind = None,\
                 invar_keys = ['FA', 'uFA', 'C_c', 'C_MD'], xps_path = None, btens_der = False,\
                    zscore_output = False):
        
        #--------------------------------------------------------
        # QTI scalar invariants to be fitted:
        self.invar_keys = invar_keys
        
        #--------------------------------------------------------
        # dw-signal input (.nii), affines for writing output:
        self.X, self.nii_affines = read_nii(nii_path)
        
        #--------------------------------------------------------
        # read target (dps.mat), if not present: prediction mode:
        if scalar_invars_path is not None:
            self.target_present = True
            self.y, self.y_headers = read_mat(scalar_invars_path, invar_keys)
        else:
            self.target_present = False
            self.y, self.y_headers = torch.ones_like(self.X[...,0]), [None for _ in range(len(nii_path))]
        
        #--------------------------------------------------------
        # read mask (.nii), if not present, fit all voxels
        if mask_path is not None:
            self.mask = read_mask_nii(mask_path)
        else:
            self.mask = self.create_dummy_mask()
        
        #--------------------------------------------------------
        # option for global output normalization:
        self.zscore_output = zscore_output
        self.zscore = None

        #--------------------------------------------------------
        # absorb bottom/top slices into mask, if specified:
        if slice_ind == None:
            self.slice_ind = [[0, self.X.size()[1]] for _ in range(self.X.size()[0])]
        else:
            self.slice_ind = slice_ind

        self.sel_slices()

        #--------------------------------------------------------
        # set to QTIb_MLP (btens->input)
        if xps_path is not None:
            # select between direct btens input or derived measures here:
            if not btens_der:
                self.btens = read_bt_mat(xps_path)
            else:
                self.btens = read_xps_mat(xps_path)
        else:
            self.btens = None

    #-------------------------------------------------------------------------------------
    # Default methods:
    def __len__(self):
        return len(self.X)

    def __getitem__(self, batch):
        return self.X[batch], self.y[batch]


    #-------------------------------------------------------------------------------------
    # Methods/data preparation:
    #-------------------------------------------------------------------------------------
    # masking:
    # edge slice exclusion
    # adapted to allow flexible exclusion of lower/upper slices per brain
    # ind[0]: 1st included slice, ind[1]: last included slice (0-based)
    def sel_slices(self):
        for i, ind in enumerate(self.slice_ind):
            self.mask[i, :ind[0], ...] = False
            self.mask[i, (ind[1] + 1) :, ...] = False

    def create_dummy_mask(self):
        return torch.ones(self.X.size()[:-1], dtype=torch.bool)

    def apply_mask(self):
        self.X = self.X[self.mask]
        self.y = self.y[self.mask]

    #-------------------------------------------------------------------------------------
    # flattening:
    def flatten_slice_dim(self, stop = -2):
        self.X = torch.flatten(self.X.contiguous(), end_dim=stop) # masked tensor requires .contiguous()
        self.y = torch.flatten(self.y.contiguous(), end_dim=-2)
        self.mask = torch.flatten(self.mask.contiguous())

    #-------------------------------------------------------------------------------------
    # normalization:
    # masked tensor used to compute mean, std while ignoring masked voxels:
    def apply_masked_tensor(self):
        self.X = masked_tensor(self.X, self.mask[...,None].expand(-1,-1,-1,-1, self.X.size()[-1]))
        if len(self.y.size()) > 4:
            self.y = masked_tensor(self.y, self.mask[...,None].expand(-1,-1,-1,-1, self.y.size()[-1]))
        elif len(self.y.size()) == 4:
            self.y = masked_tensor(self.y, self.mask)
    #--------------------------------------------------------
    # brain-wise z-score normalization of dw-signal:
    def zscore_norm_input(self, disable = False):
        nan = float ('nan')   # use nans to track masked values

        if disable:
            # option to disable, convert back to normal tensor:
            self.X = (self.X.to_tensor(nan))
            return
        S_mean = torch.mean(self.X, dim=tuple(range(1, len(self.X.size()))), keepdim = True, dtype=torch.float32)
        S_std = torch.std(self.X, dim=tuple(range(1, len(self.X.size()))), keepdim = True)
        # masked tensors currently do not broadcast for elementwise operations.. convert back and forth:
        # print('\ns_mean =\n {} \nS_std =\n {}\n'.format(torch.squeeze(S_mean.to_tensor(nan)), torch.squeeze(S_std.to_tensor(nan))))
        self.X = (self.X.to_tensor(nan) - S_mean.to_tensor(nan)) / S_std.to_tensor(nan)   # regular Tensors will broadcast
        # self.X = masked_tensor(self.X, ~self.X.isnan())   # convert back to MaskedTensor - removed!
        
    #--------------------------------------------------------
    # brain-wise z-score normalization of target/scalar invars:
    def zscore_norm_labels(self, norm_keys = ['None']):
        norm_y = []
        nan = float ('nan')    # use nans to track masked values

        if self.zscore_output:
            # global normalization over all brains
            start_dim = 0
            self.zscore = torch.cat((torch.zeros(len(self.invar_keys))[None,...], torch.ones(len(self.invar_keys))[None,...]), dim = 0)
        else:
            # brain-wise zscore norm.
            start_dim = 1
        
        for invar, name in enumerate(self.invar_keys):
            # if key present, perform normalization:
            dim = start_dim
            if name in norm_keys:
                if name == 's0':
                    # always z-score norm every brain individually for s0
                    dim = 1 
                    # print('\ns0_mean =\n {} \nS0_std =\n {}\n'.format(torch.squeeze(invar_mean.to_tensor(nan)), torch.squeeze(invar_std.to_tensor(nan))))

                invar_mean = torch.mean(self.y[..., invar], dim=tuple(range(dim, len(self.y.size()) - 1)), keepdim = True, dtype=torch.float32)
                invar_std = torch.std(self.y[..., invar], dim=tuple(range(dim, len(self.y.size()) - 1)), keepdim = True)
                # masked tensors currently do not broadcast for elementwise operations.. convert back and forth:
                norm_y.append((self.y[..., invar].to_tensor(nan) - invar_mean.to_tensor(nan)) / invar_std.to_tensor(nan))   # regular Tensors will broadcast

                # print('\n' + name + ' mean =\n {} \n'.format(torch.squeeze(invar_mean.to_tensor(nan))) + name + ' std =\n {}\n'.format(torch.squeeze(torch.squeeze(invar_std.to_tensor(nan)))))

            else:
                # perform no normalization:
                norm_y.append(self.y[..., invar].to_tensor(nan))

            if self.zscore_output and name != 's0':
                # store zscore:
                self.zscore[0,invar] = torch.squeeze(invar_mean.to_tensor(nan))
                self.zscore[1,invar] = torch.squeeze(invar_std.to_tensor(nan))

        # update self.y:
        norm_y = torch.from_numpy(np.asarray(norm_y, dtype=np.float32))
        norm_y = torch.permute(norm_y, (1, 2, 3, 4, 0))
        self.y = norm_y
        # self.y = masked_tensor(norm_y, ~norm_y.isnan()) # convert back to MaskedTensor - removed!
        del norm_y

    #--------------------------------------------------------
    # brain-wise min-max normalization of target/scalar invars:
    # not adapted for global norm
    def min_max_norm_labels(self, norm_keys = ['None']):
        norm_y = []
        nan = float ('nan')    # use nans to track masked values

        for invar, name in enumerate(self.invar_keys):
            if name in norm_keys:
                # only mean/std/min/max along spatial dims, keep file dim seperate
                invar_min = torch.amin(self.y[..., invar], dim=tuple(range(1, len(self.y.size()) - 1)), keepdim = True)
                invar_max = torch.amax(self.y[..., invar], dim=tuple(range(1, len(self.y.size()) - 1)), keepdim = True)
                if name == 's0':
                    print('\ns0_min =\n {} \ns0_max =\n {}\n'.format(torch.squeeze(invar_min.to_tensor(nan)), torch.squeeze(invar_max.to_tensor(nan))))
                elif name == 'MD':
                    print('\nMD_min =\n {} \nMD_max =\n {}\n'.format(torch.squeeze(invar_min.to_tensor(nan)), torch.squeeze(invar_max.to_tensor(nan))))
            # masked tensors currently do not broadcast for elementwise operations.. convert back and forth:
                norm_y.append((self.y[..., invar].to_tensor(nan) - invar_min.to_tensor(nan)) / (invar_max.to_tensor(nan) - invar_min.to_tensor(nan)))   # regular Tensors will broadcast
            else:
                # if invar not normalized, simply copy
                norm_y.append(self.y[..., invar].to_tensor(nan))

        norm_y = torch.from_numpy(np.asarray(norm_y, dtype=np.float32))
        # reorder to preserve dim order: (file/head, slice z, x, y, n_invar) 
        norm_y = torch.permute(norm_y, (1, 2, 3, 4, 0))
        self.y = norm_y
        del norm_y

    #-------------------------------------------------------------------------------------
    # further input preparation:
    #--------------------------------------------------------
    # remove negative signal:
    def thresh_neg_vals(self):
        self.X[self.X < 0.0] = 0.0 # basically relu lol

    #--------------------------------------------------------
    # method to remove nan values from input and labels
    # useful if clamping of extreme values by nan masking is used
    # rough check for plausible numbers done.
    def mask_nan_vals(self):
        mask_X = torch.count_nonzero(torch.isnan(self.X), dim=-1)
        mask_y = torch.count_nonzero(torch.isnan(self.y), dim=-1)
        mask = torch.logical_and((mask_X == 0), (mask_y == 0))

        self.mask = torch.logical_and(self.mask, mask)

    #--------------------------------------------------------
    # method to remove zero values, e.g. if log signal should be calculated
    # only if no normalization is used, since that introduces neg. values inevitably
    def mask_zero_vals(self):
        mask = torch.count_nonzero(self.X, dim=-1)
        mask = self.X.size()[-1] - mask
        mask = mask == 0

        self.mask = torch.logical_and(self.mask, mask)

    #--------------------------------------------------------
    # option to exclude b0 scan in 1st position
    # do this before normalization!
    def drop_b0(self):
        self.X = self.X[..., 1:]

    #-------------------------------------------------------------------------------------
    # input preparation for btens/QTIb_MLP
    #--------------------------------------------------------
    # append btens to self.X (interleaved after each signal entry):
    def cat_inputs(self):
        if self.btens is not None:
            # extension from QTIp_MLP to add btens as feature
            # replicate btens along missing dims to match X.size(),
            # then cat, not used unless btens becomes feature
            # dim order: (file/head, slice z, x, y, n_scan, 6)
            self.btens = self.btens[:, None, None, None,...].expand(-1, self.X.size()[1],self.X.size()[2],\
                                                            self.X.size()[3], -1, -1)
            # self.btens = self.btens[:, None, None, None,...].expand(self.btens.size()[0],self.X.size()[1],self.X.size()[2],self.X.size()[3], self.btens.size()[1], self.btens.size()[-1])

            # dummy dim for X: torch.Linear: last dim must have size of layer!
            # add dummy dim at axis -2?
            self.X = self.X[..., None]
            # concatenate
            self.X = torch.cat((self.X, self.btens), dim=-1)
            # torch.nn.Linear: retains all dims for input, except the last one:
            # self.X = torch.permute(self.X, (0, 1, 2, 3, 5, 4))
            self.X = torch.flatten(self.X, start_dim=-2, end_dim=-1)
            print(self.X.size())
        else:
            raise Exception("No input for btens has been passed.")

    #--------------------------------------------------------
    # analogous to zscore_norm_input(), btens not masked_tensor
    # should work both before and after expansion of btens, since copy dims in middle
    def zscore_norm_btens(self):
        if self.btens is not None:
            S_mean = torch.mean(self.btens, dim=tuple(range(1, len(self.btens.size()))), keepdim = True, dtype=torch.float32)
            S_std = torch.std(self.btens, dim=tuple(range(1, len(self.btens.size()))), keepdim = True)
            self.btens = (self.btens - S_mean) / S_std
            print(torch.std(self.btens[0]))
        else:
            raise Exception("No input for btens has been passed.")

    #--------------------------------------------------------
    # analogous to min_max_norm_labels(), btens not masked_tensor
    # should work both before and after expansion of btens, since copy dims in middle
    def min_max_norm_btens(self):
        if self.btens is not None:
            btens_min = torch.amin(self.btens, dim=tuple(range(1, len(self.btens.size()))), keepdim = True)
            btens_max = torch.amax(self.btens, dim=tuple(range(1, len(self.btens.size()))), keepdim = True)
            self.btens = (self.btens - btens_min) / (btens_max - btens_min)
            print(torch.amax(self.btens[0]))
        else:
            raise Exception("No input for btens has been passed.")

    #--------------------------------------------------------
    # scale btens for adequate weight in optimization:
    def scale_btens(self, f_scale = 5e-10):
        self.btens = f_scale * self.btens

    #-------------------------------------------------------------------------------------
    # set of methods to transform alt. representation of btens
    #--------------------------------------------------------
    # scale bval for adequate weight in optimization:
    def scale_bval(self, f_scale = 5e-10):
        bval = f_scale * self.btens[..., 0]
        self.btens = torch.cat((bval[..., None], self.btens[..., 1:]), dim=-1)
        del bval

    #--------------------------------------------------------
    # transform bvec to spherical coordinates (angles only, since |bvec| = 1)
    # normalize angles to [0, 1]
    def bvec_cart2sph(self):
        theta = torch.atan(torch.divide(self.btens[..., 2], self.btens[..., 1]))
        theta /= torch.pi/2.0
        phi = torch.acos(torch.divide(self.btens[..., 3], torch.linalg.vector_norm(self.btens[..., 1:4], dim = -1)))
        phi /= torch.pi
        self.btens = torch.cat((torch.unsqueeze(self.btens[..., 0], dim=-1) , theta[..., None], phi[..., None], self.btens[..., 4:]), dim=-1)
        del theta, phi
    
    #--------------------------------------------------------
    # drop b_eta as feature, since very small values
    def btens_drop_b_eta(self):
        self.btens = self.btens[..., :-1]

#-------------------------------------------------------------------------------------
# Functions:
#-------------------------------------------------------------------------------------
# file reading functions:
#--------------------------------------------------------
# dw-signal input as .nii
# always pass list as file_path, even when only one file
def read_nii(file_path):
    if isinstance(file_path, list):
        data = []
        nii_affines = []
        for path in file_path:
            img = nib.load(path)
            data.append(img.get_fdata())
            nii_affines.append(img.affine)
        data = np.asarray(data, dtype=np.float32) # changed dtype from 64 -> 32 (torch default in model)
    # dim order: (file/head, slice z, x, y, n_scan)
    else:
        raise Exception("file_path argument has to be of type list.")
    return torch.permute(torch.from_numpy(data), (0, 3, 1, 2, 4)), nii_affines

#--------------------------------------------------------
# binary brain mask input as .nii
# always pass list as file_path, even when only one file
def read_mask_nii(file_path):
    if isinstance(file_path, list):
        data = []
        for path in file_path:
            img = nib.load(path)
            data.append(img.get_fdata())
        data = np.asarray(data, dtype=bool)
    else:
        raise Exception("file_path argument has to be of type list.")
    return torch.permute(torch.from_numpy(data), (0, 3, 1, 2))

#--------------------------------------------------------
# model parameters/scalar invariants as dps.mat (struct):
# always pass list as file_path, even when only one file
def read_mat(file_path, invar_keys):
    if isinstance(file_path, list):
        data = []
        nii_headers = []
        for path in file_path:
            # data:
            values = []
            container = sio.loadmat(path)
            for key in invar_keys:
                values.append(container['dps'][key][0,0])
            data.append(values)
            # .nii header: not used?
            # taken from Eval_metrics
            try:
                nii_h = container['dps']['nii_h'][0,0][0,0]
                # affines.append(np.transpose(np.concatenate((nii_h['srow_x'], nii_h['srow_y'], nii_h['srow_z'], np.array([0, 0, 0, 1])[:, None]), axis=-1)))
            except:
                nii_h = None
        data = np.asarray(data, dtype=np.float32) # changed dtype from 64 -> 32
    else:
        raise Exception("file_path argument has to be of type list.")
    # dim order: (file/head, slice z, x, y, n_invar)
    return torch.permute(torch.from_numpy(data), (0, 4, 2, 3, 1)), nii_h

#--------------------------------------------------------
 # load btens derived vars (QTIb_MLP)
def read_xps_mat(file_path):
    if isinstance(file_path, list):
        data = []
        for path in file_path:
            values = []
            container = sio.loadmat(path)
            for key in ['b', 'u', 'b_delta', 'b_eta']:
                tmp = container['xps'][key][0,0]
                tmp_size = tmp.shape[-1]
                if tmp_size > 1:
                    for ind in range(tmp_size):
                        values.append(tmp[:, ind])
                else:
                    values.append(np.squeeze(tmp))
            data.append(values)
        data = np.asarray(data, dtype=np.float32) # changed dtype from 64 -> 32
    else:
        raise Exception("file_path argument has to be of type list.")
    # dim order: (file/head , n_scan, len(btens))
    return torch.permute(torch.from_numpy(data), (0, 2, 1))

#--------------------------------------------------------
 # load btens (QTIb_MLP)
# always pass list as file_path, even when only one file
# was recycled from QTIp_MLP to load xps.mat, not dps.mat
def read_bt_mat(file_path):
    if isinstance(file_path, list):
        btens = []
        for path in file_path:
            container = sio.loadmat(path)
            btens.append(container['xps']['bt'][0,0])

        btens = np.asarray(btens, dtype=np.float32) # changed dtype from 64 -> 32
    else:
        raise Exception("file_path argument has to be of type list.")
    # dim order: (file/head , n_scan, len(btens))
    return torch.from_numpy(btens) # try this, btens not an input to opt.

#-------------------------------------------------------------------------------------
# histogram plotting:
#--------------------------------------------------------
# function for plotting histograms of inputs X & y
# only works before flattening, oc.
# flip: plot one n_scan/n_invar index of all files in one hist
# no flip: plot all passed n_scan/n_invar in one hist per file
def plot_hist(data, file_index, last_index, flip=False):
    if len(data.size()) > 5:
        raise Exception("Input has too many dimensions.\n\
        If btens present, select along last dimension accordingly.")   

    fig_list = []
    if not flip:
        for a, ind_1 in enumerate(file_index):
            fig_list.append((plt.subplots()))
            for b, ind_2 in enumerate(last_index):
                fig_list[a][1].hist(torch.flatten(data[ind_1, ..., ind_2].contiguous()),\
                                    bins = 1000, label='file_{}_{}'.format(ind_1, ind_2), alpha = 0.75)
            fig_list[a][1].legend(loc='best', frameon=False)
            fig_list[a][1].set_ylabel(r'Ocurrences')
            fig_list[a][1].set_xlabel(r'Value')
    else:
        for b, ind_2 in enumerate(last_index):
            fig_list.append((plt.subplots()))
            for a, ind_1 in enumerate(file_index):
                fig_list[b][1].hist(torch.flatten(data[ind_1, ..., ind_2].contiguous()),\
                                    bins = 1000, label='file_{}_{}'.format(ind_1, ind_2), alpha = 0.25)
            fig_list[b][1].legend(loc='best', frameon=False)
            fig_list[b][1].set_ylabel(r'Ocurrences')
            fig_list[b][1].set_xlabel(r'Value')

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# QTI_MLP class
#-------------------------------------------------------------------------------------
'''
class QTI_MLP(nn.Module):
    def __init__(self, n_scans, n_invars, layer_norm = True, final_sigmoid = False):
        super().__init__()
        self.n_scans = n_scans
        self.n_invars = n_invars
        self.final_sigmoid = final_sigmoid
        self.layer_norm = layer_norm
        self.layers = nn.Sequential(
            nn.Linear(self.n_scans, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, n_invars))
        if self.final_sigmoid:
            self.layers.append(nn.Sigmoid())

        # old architecture:
        # self.layers = nn.Sequential(nn.Linear(self.n_scans, 108), nn.ReLU(), nn.Linear(108, 216), nn.ReLU(), nn.Linear(216, 108), nn.ReLU(), nn.Linear(108, n_invars))
'''
class QTI_MLP(nn.Module):
    def __init__(self, n_scans, n_invars, layer_norm = True, final_sigmoid = False, bias = True):
        super().__init__()
        #--------------------------------------------------------
        # no. of dw-signals/voxel as features = size of 1st layer
        self.n_scans = n_scans
        #--------------------------------------------------------
        # no. of scalar invars to be fitted/vox = output size
        self.n_invars = n_invars
        #--------------------------------------------------------
        # final sigmoid activation, if [0...1] params estimated seperatly
        self.final_sigmoid = final_sigmoid
        #--------------------------------------------------------
        self.layer_norm = layer_norm
        #--------------------------------------------------------
        # train w or w/o bias neuron
        self.bias = bias
        #--------------------------------------------------------
        self.layers = nn.Sequential()
        if self.layer_norm:
            self.layers.append(nn.LayerNorm(self.n_scans, elementwise_affine=False))

        linear_ReLU_stack = [
            # nn.Dropout(0.2),
            nn.Linear(self.n_scans, 128, bias=bias),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(128, 256, bias=bias),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(256, 128, bias=bias),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(128, 32, bias=bias),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(32, n_invars, bias=bias)
            ]
        
        for layer in linear_ReLU_stack:
            self.layers.append(layer)
        if self.final_sigmoid:
            self.layers.append(nn.Sigmoid())


    def forward(self, x):
        pred = self.layers(x)
        return pred 
'''
        # # must adequately scale logits after sigmoid activation, needs to be custom tailored
        # if self.final_sigmoid:
        #     pred = torch.cat((torch.unsqueeze((pred[:,0])*2800, dim=-1), \
        #             torch.unsqueeze(pred[:,1]*4, dim=-1), pred[:,2:]), dim = -1)
'''

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# QTIb_MLP class (eats btens, too)
#-------------------------------------------------------------------------------------
class QTIb_MLP(nn.Module):
    def __init__(self, n_scans, n_invars, final_sigmoid = False):
        super().__init__()
        self.n_scans = n_scans
        self.n_invars = n_invars
        self.final_sigmoid = final_sigmoid
        self.layers = nn.Sequential(
            nn.Linear(self.n_scans, 512),
            nn.ReLU(),
            nn.Linear(512, 1024), # (512, 512) (512, 512)
            nn.ReLU(),
            nn.Linear(1024, 512), # (512, 256) (512, 128)
            nn.ReLU(),
            nn.Linear(512, 128), # (512, 256) (128, 32)
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, n_invars))
        if self.final_sigmoid:
            self.layers.append(nn.Sigmoid())

    def forward(self, x):
        pred = self.layers(x)
        return pred 
'''       
        # # must adequately scale logits after sigmoid activation, needs to be custom tailored
        # if self.final_sigmoid:
        #     pred = torch.cat((torch.unsqueeze((pred[:,0])*2800, dim=-1), \
        #             torch.unsqueeze(pred[:,1]*4, dim=-1), pred[:,2:]), dim = -1)
'''    
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Training loop:
#-------------------------------------------------------------------------------------
def train_loop(dataloader, model, loss_fn, optimizer, device, batch_size):

    epoch_loss = 0.
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):

        X = X.to(device)
        y = y.to(device)
        #--------------------------------------------------------
        # Compute prediction and loss:
        # Perform forward pass
        pred = model(X)
        # Compute loss
        loss = loss_fn(pred, y) # , sigmoid(y))

        #--------------------------------------------------------
        # Backpropagation: 
        # Perform backward pass
        loss.backward()
        # Perform optimization
        optimizer.step()
        # Zero the gradients
        optimizer.zero_grad()
            
        # return epoch loss/voxel (MSE averaged over each batch)
        epoch_loss += loss.item()

    return epoch_loss/((batch + 1)*batch_size)

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# test loop:
#-------------------------------------------------------------------------------------
def test_loop(dataloader, model, loss_fn, device, batch_size):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    running_vloss = 0.0

    # torch.no_grad(): no gradients are computed during test mode
    # also to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            #--------------------------------------------------------
            pred = model(X)
            vloss = loss_fn(pred, y) # , sigmoid(y))
            #--------------------------------------------------------
            # return epoch loss/voxel (MSE averaged over each batch)
            running_vloss += vloss.item()

    return running_vloss / ((batch + 1)*batch_size)

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# function to reverse global z-score normalization for model output/prediction
# works well, if statistics (mean, std) of training set targets are representative
# of test data as well, allows fitting all quant. params of different scales by one MLP
def rev_zscore(pred, zscore):
    invar_mean = torch.unsqueeze(zscore[0, :], 0).expand(pred.size(0), -1)
    invar_std = torch.unsqueeze(zscore[1, :], 0).expand(pred.size(0), -1)

    return torch.add(torch.multiply(pred, invar_std), invar_mean)


#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
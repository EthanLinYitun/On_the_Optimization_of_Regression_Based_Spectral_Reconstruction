# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:04:43 2020

@author: Yi-Tun Lin
"""

import os
import utils
import numpy as np
import pickle
from data import *
from evaluation_metrics import *

regress_mode = {'type': 'poly', # {"poly", "root-poly", "rbfn"}, also "HSCNN-R" (but only for testing)
                'order': 1,
                'num_centers': 45, # need to match the rbfn training setting (see rbf.py)
                'dim_spec': 31,
                'dim_rgb': 3,
                'target_wavelength': np.arange(400,701,10)
                }

advanced_mode = {'Sparse': False,
                 'Per_Channel': False,
                 'RELS': False,
                 'RELAD': False
                 }

operation_mode = {'Train': True, # Can't train HSCNN-R. Please refer to the original publication: Shi et al. (CVPR Workshop 2018)
                  'Validation': True,
                  'Test': True
                  }

cost_funcs = {'val': mrae,
              'test': [mrae, rmse]
              }

test_modes = {'Mean': np.mean,
              'Pt99': lambda X: np.percentile(X, 99)
              }

directories = {'data': './ICVL_data/',
               'precal': './ICVL_data/precal/',
               'sparse_label': './ICVL_data/sparse/',
               'models': './models/',
               'HSCNN-R': './models/HSCNN-R/',
               'results': './results/'
               }

resources = {'cmf': load_cie64cmf('./resources/', regress_mode['target_wavelength']),
             'rbf_net': [pickle.load(open('./resources/rbf_icvl_train'+i+'.pkl', 'rb')) for i in ['AB', 'CD']],
             'crsval_name_list': [open('./resources/fn_icvl_group_'+i+'.txt').readlines() for i in ['A','B','C','D']],
             'anchors': [loadmat2array('./resources/icvl_anchors.mat', 'anchor_'+ i) for i in ['A', 'B']]
             }


def train(img_list, crsval_mode=0):
    print("  Training Regression Matrix...")
    train_suffix, _ = generate_crsval_suffix(crsval_mode)
    
    if regress_mode['type'] == 'HSCNN-R':
        pass # train and validate by other scripts
    
    elif advanced_mode['RELAD']:
        assert os.path.isdir(directories['sparse_label']), 'Please run sparse.py first' # use pretrained sparse data in the validation step 
    
    elif advanced_mode['Sparse']:
        # Loading pretrained sparse data
        assert os.path.isdir(directories['sparse_label']), 'Please run sparse.py first'
        with open(os.path.join(directories['precal'], 'sparse_all_data'+train_suffix+'.pkl'), 'rb') as handle:
            gt_data = pickle.load(handle)
        nearest_neighbors = np.load(os.path.join(directories['precal'], 'sparse_neighbor_idx'+train_suffix+'.npy')).astype(int)
        num_anchors, num_neighbors = nearest_neighbors.shape
        
        # Training "Multiple" Regression Matrix
        RegMat = []
        for i in range(num_anchors):
            RegMat.append(utils.RegressionMatrix(regress_mode, advanced_mode))
            
        for i in range(num_anchors):
            nearest_idx = nearest_neighbors[i, :]
            gt_data_nearest = {}
            gt_data_nearest['spec'] = gt_data['spec'][nearest_idx, :]
            gt_data_nearest['rgb'] = gt_data['rgb'][nearest_idx, :]
            
            regress_input, regress_output = utils.data_transform(gt_data_nearest, regress_mode, advanced_mode, crsval_mode, resources)
            RegMat[i].update(regress_input, regress_output)
        
        utils.save_model(RegMat, directories['models'], regress_mode, advanced_mode, train_suffix)
    
    else:
        RegMat = utils.RegressionMatrix(regress_mode, advanced_mode)
        for img_name in img_list:
            spec_img = load_icvl_data(directories['data'], img_name[:-1]) # 31 x H x W
            gt_data = {}
            gt_data['spec'], gt_data['rgb'] = utils.cal_gt_data(spec_img, resources['cmf'])        
                
            regress_input, regress_output = utils.data_transform(gt_data, regress_mode, advanced_mode, crsval_mode, resources)
            RegMat.update(regress_input, regress_output)

        utils.save_model(RegMat, directories['models'], regress_mode, advanced_mode, train_suffix)
    

def validate(img_list, crsval_mode=0):    
    train_suffix, val_suffix = generate_crsval_suffix(crsval_mode)    
    print("  Validating Regression Matrix...")
    
    if regress_mode['type'] == 'HSCNN-R':
        pass # train and validate by other scripts
    
    elif advanced_mode['Sparse']:
        
        if advanced_mode['RELAD']:
            # Loading precollected sparse validation data
            with open(os.path.join(directories['precal'], 'sparse_all_data'+train_suffix+'.pkl'), 'rb') as handle:
                gt_data_tr = pickle.load(handle)
            nearest_neighbors_tr = np.load(os.path.join(directories['precal'], 'sparse_neighbor_idx'+train_suffix+'.npy')).astype(int)   
            RegMat = []
            
        else:
            RegMat = utils.load_model(directories['models'], regress_mode, advanced_mode, train_suffix)
        
        # Loading precollected sparse validation data
        with open(os.path.join(directories['precal'], 'sparse_all_data'+val_suffix+'.pkl'), 'rb') as handle:
            gt_data_val = pickle.load(handle)
            
        nearest_neighbors_val = np.load(os.path.join(directories['precal'], 'sparse_neighbor_idx'+val_suffix+'.npy')).astype(int)
        num_anchors_val, num_neighbors_val = nearest_neighbors_val.shape
        
        # Regularizing Regression Matrix
        for i in range(num_anchors_val):
            if i%100 == 50:
                print('    anchor', i)
            nearest_idx_val = nearest_neighbors_val[i, :]
            gt_data_nearest_val = {}
            gt_data_nearest_val['spec'] = gt_data_val['spec'][nearest_idx_val, :]
            gt_data_nearest_val['rgb'] = gt_data_val['rgb'][nearest_idx_val, :]
            
            if advanced_mode['RELAD']:
                nearest_idx_tr = nearest_neighbors_tr[i, :]
                gt_data_nearest_tr = {}
                gt_data_nearest_tr['spec'] = gt_data_tr['spec'][nearest_idx_tr, :]
                gt_data_nearest_tr['rgb'] = gt_data_tr['rgb'][nearest_idx_tr, :]
                
                regress_input_tr, regress_output_tr = utils.data_transform(gt_data_nearest_tr, regress_mode, advanced_mode, crsval_mode, resources)
                regress_input_val, _ = utils.data_transform(gt_data_nearest_val, regress_mode, advanced_mode, crsval_mode, resources)
                
                RegMat.append(utils.RegressionMatrix(regress_mode, advanced_mode))
                for channel in range(RegMat[i].get_dim_regress_output()):
                    RegMat[i].set_gamma(0, channel=channel, regress_input_tr=regress_input_tr, regress_output_tr=regress_output_tr[:, channel])
                    
                RegMat[i] =  utils.regularize(RegMat[i], regress_input_val, gt_data_nearest_val, advanced_mode, 
                                              cost_funcs['val'], resources,
                                              regress_input_tr, regress_output_tr,
                                              regress_mode=regress_mode,
                                              show_graph=False)
                
            else:
                regress_input_val, _ = utils.data_transform(gt_data_nearest_val, regress_mode, advanced_mode, crsval_mode, resources)
                RegMat[i] = utils.regularize(RegMat[i], regress_input_val, gt_data_nearest_val, advanced_mode, 
                                             cost_funcs['val'], resources, show_graph=False)
        
        utils.save_model(RegMat, directories['models'], regress_mode, advanced_mode, train_suffix + val_suffix) 
    
    elif advanced_mode['RELAD']:
        # Loading pretrained sparse data
        with open(os.path.join(directories['precal'], 'sparse_all_data'+train_suffix+'.pkl'), 'rb') as handle:
            gt_tr = pickle.load(handle)
        
        dir_precal_data = os.path.join(directories['precal'], 'all_data'+val_suffix+'.pkl')
        
        if os.path.isfile(dir_precal_data):
            with open(dir_precal_data, 'rb') as handle:
                gt_val = pickle.load(handle)
        else:
            print("  Preparing Validation Images...")
            gt_val = utils.collect_gt_data(directories['data'], img_list, resources['cmf'], 2000)    
            with open(dir_precal_data, 'wb') as handle:
                pickle.dump(gt_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        regress_input_tr, regress_output_tr = utils.data_transform(gt_tr, regress_mode, advanced_mode, crsval_mode, resources)
        regress_input_val, _ = utils.data_transform(gt_val, regress_mode, advanced_mode, crsval_mode, resources)
        
        RegMat = utils.RegressionMatrix(regress_mode, advanced_mode)
        for channel in range(RegMat.get_dim_regress_output()):
            RegMat.set_gamma(0, channel=channel, regress_input_tr=regress_input_tr, regress_output_tr=regress_output_tr[:, channel])
            
        RegMat = utils.regularize(RegMat, regress_input_val, gt_val, advanced_mode, 
                                  cost_funcs['val'], resources,
                                  regress_input_tr, regress_output_tr,
                                  regress_mode=regress_mode,
                                  show_graph=True)
        
        utils.save_model(RegMat, directories['models'], regress_mode, advanced_mode, train_suffix + val_suffix)
    
    else:
        RegMat = utils.load_model(directories['models'], regress_mode, advanced_mode, train_suffix)
        # Check if precollected data exists
        dir_precal_data = os.path.join(directories['precal'], 'all_data'+val_suffix+'.pkl')
        
        if os.path.isfile(dir_precal_data):
            with open(dir_precal_data, 'rb') as handle:
                gt_data = pickle.load(handle)
                
        else:
            print("  Preparing Validation Images...")
            gt_data = utils.collect_gt_data(directories['data'], img_list, resources['cmf'], 2000)    
            with open(dir_precal_data, 'wb') as handle:
                pickle.dump(gt_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        regress_input, _ = utils.data_transform(gt_data, regress_mode, advanced_mode, crsval_mode, resources)
        
        RegMat = utils.regularize(RegMat, regress_input, gt_data, advanced_mode, 
                                  cost_funcs['val'], resources, show_graph=True)
        
        utils.save_model(RegMat, directories['models'], regress_mode, advanced_mode, train_suffix + val_suffix)
        

def test(img_list, crsval_mode=0, file_name=()):
    print("  Testing...")
    train_suffix, val_suffix = generate_crsval_suffix(crsval_mode)  
    if regress_mode['type'] == 'HSCNN-R':
        hscnn_model = load_hscnn_R_model(directories['HSCNN-R'], regress_mode, advanced_mode, crsval_mode)
        
    else:
        RegMat = utils.load_model(directories['models'], regress_mode, advanced_mode, train_suffix + val_suffix)
      
    if advanced_mode['Sparse']:
        for img_name in img_list:
            print("    ", img_name[:-1])
            spec_img = load_icvl_data(directories['data'], img_name[:-1]) # 31 x H x W
            gt_data = {} 
            gt_data['spec'], gt_data['rgb'] = utils.cal_gt_data(spec_img, resources['cmf'])
            
            nearest_anchor = np.load(os.path.join(directories['sparse_label'], img_name[:-5]+'_label.npy')).astype(int)
            active_anchors = np.unique(nearest_anchor).astype(int)
                            
            write_row = [img_name]
            
            regress_input, _ = utils.data_transform(gt_data, regress_mode, advanced_mode, crsval_mode, resources)
            num_data = regress_input.shape[0]
            recovery = {'spec': np.zeros((num_data, regress_mode['dim_spec'])),
                        'rgb': np.zeros((num_data, regress_mode['dim_rgb']))}  
                                
            for i in active_anchors:
                is_nearest = nearest_anchor == i
                rgb_nearest = gt_data['rgb'][is_nearest, :]
                regress_input_nearest = regress_input[is_nearest, :]
                recovery_part = utils.recover(RegMat[i].get_matrix(), regress_input_nearest, advanced_mode, resources, rgb_nearest)

                recovery['spec'][is_nearest, :] = recovery_part['spec']
                recovery['rgb'][is_nearest, :] = recovery_part['rgb']

            for cost_func in cost_funcs['test']:
                cost = cost_func(gt_data, recovery) 
                    
                for tmode_key, tmode_func in test_modes.items():
                    write_row.append(tmode_func(cost))

            if len(file_name):
                write2csvfile(file_name, write_row)
                
    else:
        for img_name in img_list:
            print("    ", img_name[:-1])
            spec_img = load_icvl_data(directories['data'], img_name[:-1]) # 31 x H x W
            gt_data = {} 
            gt_data['spec'], gt_data['rgb'] = utils.cal_gt_data(spec_img, resources['cmf'])
            
            write_row = [img_name]
            
            if regress_mode['type'] == 'HSCNN-R':
                recovery = {}
                recovery = utils.recover_HSCNN_R(gt_data['rgb'], spec_img.shape, resources, hscnn_model)

            else:
                regress_input, _ = utils.data_transform(gt_data, regress_mode, advanced_mode, crsval_mode, resources)
                recovery = {}
                recovery = utils.recover(RegMat.get_matrix(), regress_input, advanced_mode, resources, gt_data['rgb'])               
                
            for cost_func in cost_funcs['test']:
                cost = cost_func(gt_data, recovery)
                    
                for tmode_key, tmode_func in test_modes.items():
                    write_row.append(tmode_func(cost))   
            
            if len(file_name):
                write2csvfile(file_name, write_row)

if __name__ == '__main__':
    
    if operation_mode['Test']:
        file_name = initialize_csvfile(directories['results'], regress_mode, advanced_mode, cost_funcs, test_modes)
            
    for cmode in [1, 2, 3, 4]:
        print("Cross Validation", cmode)
        train_list, val_list, test_list = generate_crsval_imlist(resources['crsval_name_list'], crsval_mode=cmode)
        if operation_mode['Train'] & cmode in [1, 3]:
            train(train_list, crsval_mode=cmode)
            
        if operation_mode['Validation']:
            validate(val_list, crsval_mode=cmode)
            
        if operation_mode['Test']:
            test(test_list, crsval_mode=cmode, file_name=file_name)
        
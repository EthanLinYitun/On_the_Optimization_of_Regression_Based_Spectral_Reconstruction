# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:19:10 2020

@author: Yi-Tun Lin
"""

import poly
import os
import numpy as np
from numpy.linalg import inv, det
import pickle
import matplotlib.pyplot as plt
from data import generate_file_name, load_icvl_data
from scipy.spatial.distance import cdist
import hscnn
  
def cal_gt_data(spectral_img, cmf, num_sampling_points=(), rand=False):
    spec_data = spectral_img.reshape(spectral_img.shape[0], -1).T # Dim_Data x 31    
    if num_sampling_points:
        spec_data = sampling_data(spec_data, num_sampling_points, rand)
    else:
        num_sampling_points = spec_data.shape[0]
    
    all_spec_data = spec_data
    rgb_data = all_spec_data @ cmf
    
    return all_spec_data, rgb_data


def collect_gt_data(dir_data, img_list, cmf, num_sampling_points, rand=True):
    gt_data = {'spec': [],
               'rgb': []}
    
    for img_name in img_list:
        spec_img = load_icvl_data(dir_data, img_name[:-1]) # 31 x H x W
        spec_data, rgb_data = cal_gt_data(spec_img, cmf, num_sampling_points, rand)
        
        gt_data['spec'] = gt_data['spec'] + [spec_data]
        gt_data['rgb']  = gt_data['rgb']  + [rgb_data]
    
    gt_data['spec'] = np.array(gt_data['spec']).swapaxes(0, 2).reshape(spec_data.shape[1], -1).T
    gt_data['rgb']  = np.array(gt_data['rgb']).swapaxes(0, 2).reshape(rgb_data.shape[1], -1).T
    
    return gt_data


def normc(data):
    return data / np.linalg.norm(data, axis=1, keepdims=True)


def knn(data, reference, k, batch_size=None):
    num_reference = reference.shape[0]
    if batch_size:
        pass
    else:
        batch_size = num_reference
    num_batch = num_reference//batch_size
    num_residual = num_reference%batch_size
    out_list = np.zeros((num_reference, k))
    for i in range(num_batch):
        D = cdist(reference[i*batch_size:(i+1)*batch_size, :], data)
        out_list[i*batch_size:(i+1)*batch_size, :] = np.argsort(D, axis=1)[:, :k]
    if num_residual:
        D = cdist(reference[-num_residual:, :], data)
        out_list[-num_residual:, :] = np.argsort(D, axis=1)[:, :k]
    
    return out_list
    

def nearest_neighbor(data, anchors, k):
    
    D = cdist(data, anchors)
    
    return np.argmin(D, axis=1)
    
    
    num_data = data.shape[0]
    idx_list = []
    for i in range(num_data):
        dist = np.sum((anchors - data)**2, axis=1, keepdims=False)
        idx_list.append(np.argmin(dist))
    
    return idx_list

def rgb2poly(rgb_data, poly_order, root):
    
    dim_data, dim_variables = rgb_data.shape
    poly_term = poly.get_polynomial_terms(dim_variables, poly_order, root)
    dim_poly = len(poly_term)
    
    out_mat = np.empty((dim_data, dim_poly))
    
    for term in range(dim_poly):
        new_col = np.ones((dim_data))            # DIM_DATA,
        for var in range(dim_variables):
            variable_vector = rgb_data[:, var]                             # DIM_DATA,
            variable_index_value = poly_term[term][var]
            new_col = new_col * (variable_vector**variable_index_value)
            
        out_mat[:,term] = new_col
    
    return out_mat


def rgb2rbf(rgb_data, rbf_net):
    rbf_net.transformation(rgb_data.T)
    rfb_data = rbf_net.feature.T
    
    return rfb_data


def data_transform(gt_data, regress_mode, advanced_mode, crsval_mode, resources=()):
    # Transformation on RGB data
    if regress_mode['type'] == 'poly':
        regress_input = rgb2poly(gt_data['rgb'], regress_mode['order'], root=False)
        
    elif regress_mode['type'] == 'root-poly':
        regress_input = rgb2poly(gt_data['rgb'], regress_mode['order'], root=True)
        
    elif regress_mode['type'] == 'rbfn':
        if crsval_mode in [1, 2]:
            regress_input = rgb2rbf(gt_data['rgb'], resources['rbf_net'][0])
            
        elif crsval_mode in [3, 4]:
            regress_input = rgb2rbf(gt_data['rgb'], resources['rbf_net'][1])
    
    return regress_input, gt_data['spec']


def recover(regress_matrix, regress_input, advanced_mode, resources, gt_rgb=()):
    
    recovery = {}
    recovery['spec'] = regress_input @ regress_matrix
    recovery['rgb'] = recovery['spec'] @ resources['cmf']
    
    return recovery

def per_channel_recover(regress_matrix, channel, regress_input, advanced_mode, resources, gt_data=()):
    
    recovery_ch = {}
    gt_data_ch = {}
    
    recovery_ch['spec'] = regress_input @ regress_matrix[:, channel] # DIM_Data,
    
    if len(gt_data) == 0:
        gt_data_ch = ()
    else:
        gt_data_ch['spec'] = gt_data['spec'][:, channel].reshape(-1, 1)
    
    recovery_ch['spec'] = recovery_ch['spec'].reshape(-1, 1)
        
    return gt_data_ch, recovery_ch 

def get_regression_parts(data_spec, data_from_rgb, weights=()):
    '''
    Input data_spec with shape ( DIM_DATA, DIM_SPEC )
          data_from_rgb with shape ( DIM_DATA, -1 ), could be data_poly or data_patch
    Output squared_term, body_term
    '''
    
    
    if weights == ():
        squared_term = data_from_rgb.T @ data_from_rgb    # DIM_RGB x DIM_RGB
        body_term = data_from_rgb.T @ data_spec      # DIM_RGB x DIM_SPEC
    else:
        weights = weights.reshape(1, -1)
        
        squared_term = (data_from_rgb.T * weights) @ data_from_rgb    # DIM_RGB x DIM_RGB
        body_term = (data_from_rgb.T * weights) @ data_spec      # DIM_RGB x DIM_SPEC
    
    return squared_term, body_term


class RegressionMatrix():
    def __init__(self, regress_mode, advanced_mode):
        
        self.regress_mode = regress_mode
        self.advanced_mode = advanced_mode
        
        if regress_mode['type'] == 'rbfn':
            self.__dim_regress_input = regress_mode['num_centers'] + 1
            
        elif regress_mode['type'] == 'poly':
            self.__dim_regress_input = len(poly.get_polynomial_terms(regress_mode['dim_rgb'], regress_mode['order'], False))
            
        elif regress_mode['type'] == 'root-poly':
            self.__dim_regress_input = len(poly.get_polynomial_terms(regress_mode['dim_rgb'], regress_mode['order'], True))
        
        self.__dim_regress_output = regress_mode['dim_spec']

        if advanced_mode['RELS']:
            self.__squared_term = [np.zeros((self.__dim_regress_input, self.__dim_regress_input))] * self.__dim_regress_output
            self.__body_term = np.zeros((self.__dim_regress_input, self.__dim_regress_output))
            self.__matrix = np.zeros((self.__dim_regress_input, self.__dim_regress_output))
            self.__gamma_ch = np.zeros(self.__dim_regress_output)
            
        elif advanced_mode['RELAD']:
            self.__squared_term = [np.zeros((self.__dim_regress_input, self.__dim_regress_input))] * self.__dim_regress_output
            self.__body_term = np.zeros((self.__dim_regress_input, self.__dim_regress_output))
            self.__matrix = np.zeros((self.__dim_regress_input, self.__dim_regress_output))
            self.__gamma_ch = np.zeros(self.__dim_regress_output)
            
        elif advanced_mode['Per_Channel']:
            self.__squared_term = np.zeros((self.__dim_regress_input, self.__dim_regress_input))
            self.__body_term = np.zeros((self.__dim_regress_input, self.__dim_regress_output))
            self.__matrix = np.zeros((self.__dim_regress_input, self.__dim_regress_output))
            self.__gamma_ch = np.zeros(self.__dim_regress_output)
            
        else:
            self.__squared_term = np.zeros((self.__dim_regress_input, self.__dim_regress_input))
            self.__body_term = np.zeros((self.__dim_regress_input, self.__dim_regress_output))
            self.__matrix = np.zeros((self.__dim_regress_input, self.__dim_regress_output))
            self.__gamma = 1
            
    def set_gamma(self, gamma, channel=(), regress_input_tr=(), regress_output_tr=(), weights_sqr=(), weights_reg=()):
        if self.advanced_mode['RELS']:
            self.__gamma_ch[channel] = gamma
            self.__matrix[:, channel] = inv( self.__squared_term[channel] + self.__gamma_ch[channel] * np.eye(self.__dim_regress_input) ) @ self.__body_term[:, channel]
        
        elif self.advanced_mode['Per_Channel']:
            self.__gamma_ch[channel] = gamma
            self.__matrix[:, channel] = inv( self.__squared_term + self.__gamma_ch[channel] * np.eye(self.__dim_regress_input) ) @ self.__body_term[:, channel]
        
        elif self.advanced_mode['RELAD']:
            self.__gamma_ch[channel] = gamma
            num_data = regress_input_tr.shape[0]
            
            if len(weights_reg):
                pass
                
            else:
                weights_reg = np.ones(self.__dim_regress_input)
                
            if len(weights_sqr):
                pass
                
            else:
                weights_sqr = np.ones(num_data)

            squared_term, body_term = get_regression_parts(np.ones(num_data), 
                                                           1./regress_output_tr.reshape(num_data, 1) * regress_input_tr,
                                                           weights_sqr)            
            self.__squared_term[channel] = squared_term
            self.__body_term[:, channel] = body_term
            self.__matrix[:, channel] = inv( self.__squared_term[channel] + self.__gamma_ch[channel] * np.diag(weights_reg) ) @ self.__body_term[:, channel]
        else:
            self.__gamma = gamma
            self.__matrix = inv( self.__squared_term + self.__gamma * np.eye(self.__dim_regress_input) ) @ self.__body_term
    
    def get_gamma(self, channel=()):
        if self.advanced_mode['RELS'] or self.advanced_mode['Per_Channel'] or self.advanced_mode['RELAD']:   
            return self.__gamma_ch[channel]
            
        else:
            return self.__gamma
    
    def get_matrix(self):
        return self.__matrix
    
    def get_dim_regress_input(self):
        return self.__dim_regress_input
    
    def get_dim_regress_output(self):
        return self.__dim_regress_output
    
    def reset_weights(self):
        self.__weights_reg = ()
        self.__weights_sqr = ()
        self.__num_data = ()
    
    def test_feasible_gamma(self, gamma, channel=()):
        if self.advanced_mode['RELS'] or self.advanced_mode['RELAD']:
            return det(self.__squared_term[channel] + gamma * np.eye(self.__dim_regress_input)) != 0

        else:
            return det(self.__squared_term + gamma * np.eye(self.__dim_regress_input)) != 0
    
    def update(self, regress_input, regress_output):
        if self.advanced_mode['RELS']:
            num_data = regress_input.shape[0]
            for channel in range(self.__dim_regress_output):
                squared_term, body_term = get_regression_parts(np.ones(num_data), 
                                                               1./regress_output[:, channel].reshape(num_data, 1) * regress_input)            
                self.__squared_term[channel] = self.__squared_term[channel] + squared_term
                self.__body_term[:, channel] = self.__body_term[:, channel] + body_term
        
        elif self.advanced_mode['Per_Channel']:
            squared_term, body_term = get_regression_parts(regress_output, regress_input)
            self.__squared_term = self.__squared_term + squared_term
            for channel in range(self.__dim_regress_output):
                self.__body_term[:, channel] = self.__body_term[:, channel] + body_term[:, channel]
        
        else:
            squared_term, body_term = get_regression_parts(regress_output, regress_input)
            self.__squared_term = self.__squared_term + squared_term
            self.__body_term = self.__body_term + body_term            
            
def save_model(RegMat, dir_model, regress_mode, advanced_mode, model_suffix):
    fn = generate_file_name(regress_mode, advanced_mode)
    with open(os.path.join(dir_model, fn + model_suffix + '.pkl'), 'wb') as handle:
        pickle.dump(RegMat, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(dir_model, regress_mode, advanced_mode, model_suffix):
    fn = generate_file_name(regress_mode, advanced_mode)
    with open(os.path.join(dir_model, fn + model_suffix + '.pkl'), 'rb') as handle:
        return pickle.load(handle)

def sampling_data(data, num_sampling_points, rand=False):
    if rand:
        np.random.shuffle(data)
        
    sampling_points = np.floor(np.linspace(0, len(data), num_sampling_points, endpoint=False)).astype(int)
    return data[sampling_points, :]


def regularize(RegMat, regress_input, gt_data, advanced_mode, cost_func, resources=(), regress_input_tr=(), regress_output_tr=(), regress_mode=(), show_graph=False):
    
    def determine_feasible_gamma(channel=(), max_range=None):
        if max_range:
            pass
        else:
            max_range = 20
            
        for s in range(-max_range, 0, 1):
            if RegMat.test_feasible_gamma(10**s, channel):
                break
        #print('feasible range:', s, 'to', 10)
        return np.logspace(10, s, np.abs(10-s))
    
    def regularizer(test_gammas, channel=(), tolerance = 0.00005, return_best_model=False):
        cost = []
        best_weights_sqr = []
        best_weights_reg = []
        
        for gamma in test_gammas:
            if advanced_mode['Per_Channel'] or advanced_mode['RELS']:
                RegMat.set_gamma(gamma, channel)
                gt_data_ch, recovery_ch = per_channel_recover(RegMat.get_matrix(), channel, regress_input, advanced_mode, resources, gt_data)
                cost.append(np.mean(cost_func(gt_data_ch, recovery_ch)))
            
            elif advanced_mode['RELAD']:
                
                def weight_fun(z):
                    out = np.zeros(len(z))
                    for i in range(len(z)):
                        out[i] = 1. / np.max((z[i], 10**(-6)))
                    return out
                
                pre_cost = np.inf
                cost_diff = np.inf
                t = 0
                t_max = 20
                new_weights_sqr = ()
                new_weights_reg = ()

                while cost_diff >= tolerance and t <= t_max:
                    RegMat.set_gamma(gamma, channel, regress_input_tr, regress_output_tr[:, channel], new_weights_sqr, new_weights_reg)

                    num_data = regress_input_tr.shape[0]
                    res = np.abs(np.ones(num_data) - 1./regress_output_tr[:, channel].reshape(num_data, 1) * regress_input_tr @ RegMat.get_matrix()[:, channel])
                    mar = np.median(res)/0.6745
                    
                    new_weights_sqr = weight_fun(res/mar)
                    new_weights_reg = weight_fun(RegMat.get_matrix()[:, channel])
                    
                    curr_cost = np.mean(res)
                    cost_diff = np.abs(curr_cost - pre_cost)
                    pre_cost = curr_cost
                    #print('iter:',t,', cost =', curr_cost)
                    t = t + 1
                    
                best_weights_sqr.append(new_weights_sqr)
                best_weights_reg.append(new_weights_reg)
                
                gt_data_ch, recovery_ch = per_channel_recover(RegMat.get_matrix(), channel, regress_input, advanced_mode, resources, gt_data)
                cost.append(np.mean(cost_func(gt_data_ch, recovery_ch)))
            
            else:
                RegMat.set_gamma(gamma)        
                recovery = recover(RegMat.get_matrix(), regress_input, advanced_mode, resources, gt_data['rgb'])  
                cost.append(np.mean(cost_func(gt_data, recovery)))
            
        best_gamma = test_gammas[np.argmin(cost)]
        
        if show_graph:
            plt.figure()
            plt.title('Tikhonov parameter search')
            plt.plot(test_gammas, cost)
            plt.scatter(best_gamma, np.min(cost), c='r', marker='o')
            plt.xscale('log')
            plt.show()
        
        if return_best_model:
            return best_gamma, best_weights_sqr[np.argmin(cost)], best_weights_reg[np.argmin(cost)]
        else:
            return best_gamma
    
    if advanced_mode['Per_Channel'] or advanced_mode['RELS']:
        for channel in range(RegMat.get_dim_regress_output()):
            test_gammas = determine_feasible_gamma(channel)
            best_gamma = regularizer(test_gammas, channel)
            test_gammas_fine = best_gamma * np.logspace(-1, 1, 1000)
            best_gamma = regularizer(test_gammas_fine, channel)            
            RegMat.set_gamma(best_gamma, channel)
    
    elif advanced_mode['RELAD']:
        # Initialization        
        for channel in range(RegMat.get_dim_regress_output()):
            print('current regularized channel:', channel)
            test_gammas = determine_feasible_gamma(channel, max_range=10)
            best_gamma, best_weights_sqr, best_weights_reg = regularizer(test_gammas, channel, return_best_model=True)
            RegMat.set_gamma(best_gamma, channel, regress_input_tr, regress_output_tr[:, channel], best_weights_sqr, best_weights_reg)
            
    else:
        test_gammas = determine_feasible_gamma()
        best_gamma = regularizer(test_gammas)
        test_gammas_fine = best_gamma * np.logspace(-1, 1, 1000)
        best_gamma = regularizer(test_gammas_fine)
        RegMat.set_gamma(best_gamma)
    
    return RegMat


def recover_HSCNN_R(rgb, img_shape, resources, model):
    
    dim_spec, height, width = img_shape

    rgb = np.array(rgb).T.reshape(3, height, width) 	# 3 x height x width
    rgb = np.swapaxes(np.swapaxes(rgb, 0, 1), 1, 2 )		# height x 3 x width
    
    curr_rgb = rgb.astype('float32')
    curr_rgb = np.expand_dims(np.transpose(curr_rgb,[2,1,0]), axis=0).copy() 
    
    img_res1 = hscnn.reconstruction(curr_rgb, model)
    img_res2 = np.flip(hscnn.reconstruction(np.flip(curr_rgb, 2).copy(), model), 1) 
   
    img_res3 = (img_res1+img_res2)/2
    
    final_img = np.swapaxes(np.swapaxes(img_res3/4095,0,2),1,2)
        
    recovery = {}
    recovery['spec'] = final_img.reshape(dim_spec,-1).T
    recovery['rgb'] = recovery['spec'] @ resources['cmf']
        
    return recovery
    
    
    
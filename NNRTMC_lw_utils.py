import numpy as np  
import torch
import pickle
import math
import os

def split_train_test_sample(sample_size, test_ratio ,rng):
    training_sample_size = math.ceil(sample_size*(1-test_ratio))
    shuffled_id = rng.choice(sample_size, sample_size, replace=False)
    indice_train = shuffled_id[0:training_sample_size]
    indice_test  = shuffled_id[training_sample_size:sample_size]
    return indice_train, indice_test

def draw_batches(indice_train, batch_size, rng, device, replace=False):
    training_sample_size = indice_train.shape[0]
    n_batch = math.ceil(training_sample_size/batch_size)
    shuffled_id = rng.choice(training_sample_size, training_sample_size, replace=replace)
    shuffled_id = torch.from_numpy(shuffled_id).to(device)
    batch_indice_train = []
    sta=0
    for i in range(n_batch-1):
        end = sta + batch_size
        batch_indice_train.append(shuffled_id[sta:end])
        sta = end
    batch_indice_train.append(shuffled_id[sta:training_sample_size]) 
    return batch_indice_train 

def data_std_normalization(input_array, output_array, nomral_para = None):
    ###################################################### 
    if nomral_para == None:
        ## normalization based on data std
        input_scale     = input_array.std(axis=0)
        # check not varying input
        if np.any(np.isclose(input_scale,0)):
            print(f'Warning: {np.isclose(input_scale,0).sum()} input feature(s) is fixed!')
            input_scale     = np.where(np.isclose(input_scale,0), 1, input_scale)  
        input_scale     = 1/input_scale
        input_offset    = input_array.mean(axis=0)
        output_scale    = output_array.std(axis=0)
        # check not varying input
        if np.any(np.isclose(output_scale,0)):
            print(f'Warning: {np.isclose(output_scale,0).sum()} output feature(s) is fixed!')
            output_scale     = np.where(np.isclose(output_scale,0), 1, output_scale)  
        output_scale    = 1/output_scale
        output_offset   = output_array.mean(axis=0)
        nomral_para = {'input_scale'   : input_scale, 
                       'input_offset'  : input_offset,
                       'output_scale'  : output_scale,
                       'output_offset' : output_offset} 
    # do normalization
    input_array  = (input_array  - nomral_para['input_offset' ])*nomral_para['input_scale' ]
    output_array = (output_array - nomral_para['output_offset'])*nomral_para['output_scale']
    return nomral_para, input_array, output_array 

def data_std_normalization_c(input_array, output_array, nomral_para = None):
    '''
        normalize the input and output via std scale and mean offset
        return the normalized input and output and the normalize
        also return the fixing input indice in normal_para
    '''
    input_array = np.where(np.isnan(input_array),0,input_array)
    ###################################################### 
    if nomral_para == None:
        ## normalization based on data std
        input_scale     = input_array.std(axis=0)
        input_offset    = input_array.mean(axis=0)
        # check not varying input
        range_check = np.isclose(input_array.min(axis=0),input_array.max(axis=0))
        if np.any(range_check):
            print(f'Warning: {range_check.sum()} input feature(s) is fixed!')
            input_scale     = np.where(range_check, 1, input_scale)  
        input_scale      = 1/input_scale
        input_valid_feat = (~range_check)
        output_scale    = output_array.std(axis=0)
        output_offset   = output_array.mean(axis=0)
        # check not varying output
        range_check = np.isclose(output_array.min(axis=0),output_array.max(axis=0))
        if np.any(range_check):
            print(f'Warning: {range_check.sum()} output feature(s) is fixed!')
            output_scale     = np.where(range_check, 1, output_scale)  
        output_scale    = 1/output_scale
        nomral_para = {'input_scale'      : input_scale, 
                       'input_offset'     : input_offset,
                       'output_scale'     : output_scale,
                       'output_offset'    : output_offset,
                       'input_valid_feat' : input_valid_feat} 
    # do normalization
    input_array  = (input_array  - nomral_para['input_offset' ])*nomral_para['input_scale' ]
    output_array = (output_array - nomral_para['output_offset'])*nomral_para['output_scale']
    return nomral_para, input_array, output_array

def data_std_normalization_2xflux(input_array, output_array, nomral_para = None):
    ###################################################### 
    if nomral_para == None:
        ## normalization based on data std
        input_scale     = input_array.std(axis=0)
        # check not varying input
        if np.any(np.isclose(input_scale,0)):
            print(f'Warning: {np.isclose(input_scale,0).sum()} input feature(s) is fixed!')
            input_scale     = np.where(np.isclose(input_scale,0), 1, input_scale)  
        input_scale     = 1/input_scale
        input_offset    = input_array.mean(axis=0)
        output_scale    = output_array.std(axis=0)
        # check not varying input
        if np.any(np.isclose(output_scale,0)):
            print(f'Warning: {np.isclose(output_scale,0).sum()} output feature(s) is fixed!')
            output_scale     = np.where(np.isclose(output_scale,0), 1, output_scale)  
        output_scale    = 1/output_scale
        output_scale[:2] = output_scale[:2]*5 # 2xflux
        output_offset   = output_array.mean(axis=0)
        nomral_para = {'input_scale'   : input_scale, 
                       'input_offset'  : input_offset,
                       'output_scale'  : output_scale,
                       'output_offset' : output_offset} 
    # do normalization
    input_array  = (input_array  - nomral_para['input_offset' ])*nomral_para['input_scale' ]
    output_array = (output_array - nomral_para['output_offset'])*nomral_para['output_scale']
    return nomral_para, input_array, output_array 

def data_std_normalization_10xflux(input_array, output_array, nomral_para = None):
    ###################################################### 
    if nomral_para == None:
        ## normalization based on data std
        input_scale     = input_array.std(axis=0)
        # check not varying input
        if np.any(np.isclose(input_scale,0)):
            print(f'Warning: {np.isclose(input_scale,0).sum()} input feature(s) is fixed!')
            input_scale     = np.where(np.isclose(input_scale,0), 1, input_scale)  
        input_scale     = 1/input_scale
        input_offset    = input_array.mean(axis=0)
        output_scale    = output_array.std(axis=0)
        # check not varying input
        if np.any(np.isclose(output_scale,0)):
            print(f'Warning: {np.isclose(output_scale,0).sum()} output feature(s) is fixed!')
            output_scale     = np.where(np.isclose(output_scale,0), 1, output_scale)  
        output_scale    = 1/output_scale
        output_scale[:2] = output_scale[:2]*10 # 10xflux
        output_offset   = output_array.mean(axis=0)
        nomral_para = {'input_scale'   : input_scale, 
                       'input_offset'  : input_offset,
                       'output_scale'  : output_scale,
                       'output_offset' : output_offset} 
    # do normalization
    input_array  = (input_array  - nomral_para['input_offset' ])*nomral_para['input_scale' ]
    output_array = (output_array - nomral_para['output_offset'])*nomral_para['output_scale']
    return nomral_para, input_array, output_array 

def print_key_results(NNRTMC_solver,input_torch_test, output_array_test, nomral_para): 
######################################################
    # print key results
    truth = output_array_test / nomral_para['output_scale'] + nomral_para['output_offset']
    predi = NNRTMC_solver.predict(input_torch_test)
 
    predi = predi / nomral_para['output_scale'] + nomral_para['output_offset']
    error = predi - truth
    error[:,3:] = error[:,3:]*86400  
    RMSE = ((error**2).mean(axis=0))**0.5
    MAE  = abs(error).mean(axis=0)
    Bias = error.mean(axis=0)
    print('TEST: RMSE, MAE, Bias')
    print(RMSE, MAE, Bias)   

def return_exp_dir(parent_dir, Exp_name): 
    if not os.path.exists(parent_dir): 
        raise Exception('Parent work directory do no exits! Please create: \n' + parent_dir)
    
    exp_dir = parent_dir+Exp_name
    if not os.path.exists(exp_dir): 
        print('First run!')
        print(f'Create experiment dir at: \n {exp_dir}')
        # create work dir for experiment to store results 
        os.mkdir(exp_dir)  
        # copy script to experiment dir for reference
        ossyscmd = f'cp {os.path.abspath(__file__)} {exp_dir}/ '
        print(ossyscmd)
        os.system(ossyscmd) 
    for run_num in range(100): 
        # PATH for the save the restart/result file
        PATH =  exp_dir+f'/restart.{run_num:02d}.pth'
        if not os.path.exists(PATH): 
            return run_num, exp_dir
    raise Exception('Already exist 99 runs!') 
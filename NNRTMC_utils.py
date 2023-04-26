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
    print(f"Total data size: {sample_size}")
    print(f"Test data ratio: {test_ratio}")
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
    sample_size = input_array.shape[0]
    print(f"Total data size: {sample_size}")
    ###################################################### 
    if nomral_para == None:
        ## normalization based on data std
        input_scale     = input_array.std(axis=0)
        # check not varying input
        if np.any(np.isclose(input_scale,0,atol=1e-10)):
            print(f'Warning: {np.isclose(input_scale,0).sum()} input feature(s) is fixed!')
            input_scale     = np.where(np.isclose(input_scale,0), 1, input_scale)  
        input_scale     = 1/input_scale
        input_offset    = input_array.mean(axis=0)
        output_scale    = output_array.std(axis=0)
        # check not varying input
        if np.any(np.isclose(output_scale,0,atol=1e-10)):
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

def data_std_normalization_sw(input_array, output_array, rsdt_array, nomral_para = None):
    sample_size = input_array.shape[0]
    print(f"Total data size: {sample_size}")
    day_ind = np.argwhere(~np.isclose(rsdt_array,0, atol=1e-1)).squeeze()
    # remove data that rsdt == 0 (night, no shortwave transfer) 
    sample_size2 = day_ind.shape[0]
    print(f"Night time will be removed! (rsdt==0)")
    input_array  = input_array [day_ind]
    output_array = output_array[day_ind]
    rsdt_array   = rsdt_array  [day_ind]
    print(f"Total data size (daylight): {rsdt_array.shape}")
    ###################################################### 
    if nomral_para == None:
        ## normalization based on data std
        input_scale     = input_array.std(axis=0)
        # check not varying input
        if np.any(np.isclose(input_scale,0,atol=1e-10)):
            print(f'Warning: {np.isclose( input_scale,0,atol=1e-10).sum()} input feature(s) is fixed!')
            input_scale     = np.where(np.isclose(input_scale,0), 1, input_scale)  
        input_scale     = 1/input_scale
        input_offset    = input_array.mean(axis=0)
        output_scale    = output_array.std(axis=0)
        # check not varying input
        if np.any(np.isclose(output_scale,0,atol=1e-10)):
            print(f'Warning: {np.isclose(output_scale,0, atol=1e-10).sum()} output feature(s) is fixed!')
            output_scale     = np.where(np.isclose(output_scale,0), 1, output_scale)  
        output_scale    = 1/output_scale
        output_offset   = output_array.mean(axis=0)
        nomral_para = {'input_scale'   : input_scale, 
                       'input_offset'  : input_offset,
                       'output_scale'  : output_scale,
                       'output_offset' : output_offset}  
    # do normalization
    input_array_nor  = (input_array  - nomral_para['input_offset' ])*nomral_para['input_scale' ]
    output_array_nor = (output_array - nomral_para['output_offset'])*nomral_para['output_scale']
    return nomral_para, input_array_nor, output_array_nor, rsdt_array, day_ind
 

def print_key_results(pred_output, true_output, normal_para): 
######################################################
    # print key results 
    error = (pred_output - true_output)/ normal_para['output_scale'] + normal_para['output_offset']
    error[:,3:] = error[:,3:]*86400  
    RMSE = ((error**2).mean(axis=0))**0.5
    MAE  = abs(error).mean(axis=0)
    Bias = error.mean(axis=0)
    print('TEST: RMSE, MAE, Bias')
    print(RMSE, '\n', MAE, '\n', Bias)  

 
    
def return_exp_dir(parent_dir, Exp_name, create_dir=True): 
    
    if not os.path.exists(parent_dir): 
        raise Exception('Parent work directory do no exits! Please create: \n' + parent_dir)
    exp_dir = parent_dir+Exp_name
    if create_dir == True:
        if not os.path.exists(exp_dir): 
            print('First run!')
            print(f'Create experiment dir at: \n {exp_dir}')
            # create work dir for experiment to store results 
            os.mkdir(exp_dir)  
            # copy script to experiment dir for reference
            ossyscmd = f'cp {os.path.abspath(__file__)} {exp_dir}/ '
            print(ossyscmd)
            os.system(ossyscmd) 
    for run_num in range(1,100): 
        # PATH for the save the restart/result file
        PATH =  exp_dir+f'/restart.{run_num:02d}.pth'
        # print(PATH)
        if not os.path.exists(PATH): 
            return run_num, exp_dir
    raise Exception('Already exist 99 runs!') 
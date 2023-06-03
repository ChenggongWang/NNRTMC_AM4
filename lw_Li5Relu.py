import numpy as np
import torch
from torch import nn  
import time 
import os  
######################################################
import xarray as xr  
from get_data_lw_AM4_std import get_data_lw_AM4
import argparse, sys

######################################################
# common functions to split the training and test data
# 
from NNRTMC_utils import NNRTMC_NN_lw, split_train_test_sample, \
draw_batches, data_std_normalization_lw, print_key_results_lw, return_exp_dir

torch.set_float32_matmul_precision('high')

######################################################
def custom_trainning(NNRTMC_solver, lr, loss, epochs, batch_size, de_save,
                     input_torch, output_torch,
                     indice_train, indice_test, 
                     eng_loss_frac, device, rng):
    # update lr based on test loss
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        NNRTMC_solver.optimizer, mode='min', factor=0.2, patience=50, threshold=1e-3, 
        threshold_mode='rel', cooldown=50, min_lr=0, eps=1e-08, verbose=True) 
    NNRTMC_solver.optimizer.param_groups[0]['lr'] = lr
    ######################################################
    # set training hyperparameter here
    ######################################################
    sta_time = time.time()
    for t in range(epochs): 
        batch_indice_train = draw_batches(indice_train, batch_size, rng, device, replace=False)
        lossv     = NNRTMC_solver.train(batch_indice_train, input_torch, output_torch, eng_loss_frac)
        lossvtest = NNRTMC_solver.test_loss(indice_test, input_torch,  output_torch)
        lr_scheduler.step(lossvtest[0]+lossvtest[1]) # update lr based on test loss
        if (t % de_save == 0) or (t == epochs-1):
            used_time = time.time() - sta_time  
            print( f"Epoch {t+1:05d} |Train L: {lossv[0]:8.2e} {lossv[1]:8.2e} | Vali. L: {lossvtest[0]:7.1e} {lossvtest[1]:7.1e}  "
                  +f"| ~ {used_time:3.0f}s | eta {int(used_time*((epochs-t)/de_save/60)) :3d} min")
            sta_time = time.time()
            loss.append([[t+1]+lossv+lossvtest]) # append epochs, loss, test loss
            # early stop 
            if NNRTMC_solver.optimizer.param_groups[0]['lr'] < 1e-7:
                print(f"Meet early stop criteria LR = {NNRTMC_solver.optimizer.param_groups[0]['lr']} < 1e-7" )
                print("End training")
                break
    
######################################################
def custom_trainning_ens(NNRTMC_solver, lr, loss, epochs, batch_size, de_save,
                         input_torch, output_torch,
                         indice_train, indice_test, 
                         eng_loss_frac, device, rng):
    for mi in range(len(NNRTMC_solver)): 
        print(f"Model #{mi}")
        loss_mi = []
        custom_trainning(NNRTMC_solver[mi], lr, loss_mi, epochs, batch_size, de_save,\
                         input_torch, output_torch, \
                         indice_train, indice_test, eng_loss_frac, device, rng )
        loss.append(loss_mi)
        
def main(work_dir, Exp_name):
    ######################################################
    # create dir for first run or load restart file
    run_num, exp_dir = return_exp_dir(work_dir, Exp_name)
    # copy script to experiment dir for reference
    try:
        ossyscmd = f'cp {os.path.abspath(__file__)} {exp_dir}/train_script{run_num:02d}.py' 
        os.system(ossyscmd) 
    except: 
        print('copy trainscript failed')
    model_state_dict = []
    # get restart info
    if run_num == 1:  
        nor_para = None
        for mi in range(ensemble_num):
            model_state_dict.append(None)
        lr_sta = 1e-3
    else:   # load restart file
        if run_num >= total_run_num: 
            print('All runs finished. Increase <run_num> if you need to continue to train the model.')
            return 1
        for mi in range(ensemble_num):
            PATH_last =  exp_dir+f'/model{mi}_restart.{run_num-1:02d}.pth'
            restart_data = torch.load(PATH_last)  # load exist results and restart training
            print(f'restart from: {PATH_last}')
            # read training dataset, nor_para, model parameteres
            nor_para = restart_data['nor_para']
            model_state_dict.append(restart_data['model_state_dict'])
        lr_sta = 1e-4

    ######################################################
    # load data from AM4 runs
    filelist = [f'/scratch/gpfs/cw55/AM4/work/CTL2000_train_y2000_stellarcpu_intelmpi_22_768PE/'+
            f'HISTORY/20000101.atmos_8xdaily.tile{_}.nc' for _ in range(1,7)] 
    input_array_ori, output_array_ori = \
    get_data_lw_AM4(filelist, condition=sky_cond, month_sel = None, day_sel = [2,10])   
    
    hybrid_p_sigma_para = xr.open_dataset('/tigress/cw55/data/NNRTMC_dataset/AM4_pk_bk_202207.nc')
    A_k = hybrid_p_sigma_para.ak.values[None,:]
    B_k = hybrid_p_sigma_para.bk.values[None,:]
    
    print(f"Features| Input: {input_array_ori.shape[1]},  Output: {output_array_ori.shape[1]}")
    nor_para, input_array, output_array = \
    data_std_normalization_lw(input_array_ori, output_array_ori, nor_para)  
 

    # divide the training and test data here
    # this would be different if restart the training process (be careful!)
    ind_train, ind_test = split_train_test_sample(output_array.shape[0], test_ratio=0.3, rng=rng) 
    
    ######################################################
    # move all data to GPU to accelerate training
    input_torch  = torch.tensor(input_array,  dtype=torch.float32).to(device)
    output_torch = torch.tensor(output_array, dtype=torch.float32).to(device) 
    
    ######################################################
    # initialize model
    NNRTMC_solver = []
    for mi in range(ensemble_num):
        NNRTMC_solver.append(NNRTMC_NN_lw(device, nor_para, A_k, B_k, 
                             input_array.shape[1],output_array.shape[1],
                             hidden_layer_width, model_state_dict[mi]))
    # training 
    for i in range(run_num, total_run_num+1): 
        loss = []
        batch_size = max(8000, 8000*i**2)
        if eng_loss_frac != None:
            print(f'Train info >> run: {i} lr_sta: {lr_sta:7.1e}, batch size: {batch_size}, eng_loss_frac {eng_loss_frac:7.1e}')
        else:
            print(f'Train info >> run: {i} lr_sta: {lr_sta:7.1e}, batch size: {batch_size}')
        custom_trainning_ens(NNRTMC_solver, lr_sta, loss, epochs, batch_size, de_save,\
                             input_torch, output_torch, \
                             ind_train, ind_test, eng_loss_frac, device, rng )
        ######################################################
        # save model state dict and data normalization info
        data_info = filelist
        for mi in range(len(NNRTMC_solver)):
            loss_array = np.array(loss[mi]).squeeze().T  
            PATH =  exp_dir+f'/model{mi}_restart.{i:02d}.pth'
            NNRTMC_solver[mi].save_model_restart(PATH, loss_array, data_info, nor_para)
            print(f'Model #{mi} is saved at: '+PATH)        
            print_key_results_lw(NNRTMC_solver[mi].predict(input_torch[ind_test,:]), 
                                output_array[ind_test,:], 
                                nor_para)
        lr_sta = 1e-4
        print(f'{Exp_name} Finished: run {i}!')  
        
    print('All runs finished. Increase <run_num> if you need to continue to train the model.')
    return exp_dir

if __name__ == '__main__': 
    torch.cuda.set_device(0) # select gpu_id, default 0 means the first GPU
    device = f'cuda:{torch.cuda.current_device()}'
    # set random generator
    rng = np.random.default_rng(12345)
    torch.manual_seed(12345)
    # rng = np.random.default_rng()
    
    #####################################################
    # set exp name and runs 
    # read sky_cond and eng_loss from terminal command
    parser=argparse.ArgumentParser()
    parser.add_argument("--sky_cond", help="sky condition: af, csaf")
    parser.add_argument("--eng_loss", help="minimize the energy loss: Y/N")
    parser.add_argument("--ensemble_size", help="ensemble_size of NN models")
    args=parser.parse_args()
    sky_cond = args.sky_cond
    eng_loss = args.eng_loss 
    ensemble_num = int(args.ensemble_size) 
    
    total_run_num  = 3
    hidden_layer_width = 256 
    
    Exp_name = f'ens_AM4std_lw_{sky_cond}_LiH4W{hidden_layer_width}ReluSig_E{eng_loss}_D210' 
    work_dir = '/tigress/cw55/work/2022_radi_nn/NN_AM4/work/'
    epochs = 2000
    de_save = 200 
    print(f'>>| EXP: {Exp_name} ')
    print(f'>>| Ensemble size {ensemble_num} | total_run_num: {total_run_num} | epochs per run: {epochs} ')
    
    if eng_loss != 'Y':
        eng_loss_frac = None
    else:
        eng_loss_frac = 1e-4
        
    exp_dir = main(work_dir, Exp_name)
    
    # move slurm log to work dir
    job_id = int(os.environ["SLURM_JOB_ID"])
    ossyscmd = f'cp slurm-{job_id}.out {exp_dir}/' 
    print(ossyscmd)
    os.system(ossyscmd) 

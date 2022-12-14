import numpy as np
import torch
from torch import nn  
import time 
import os  

######################################################
# NN module includes:
# NN model structure 
class NeuralNetwork(nn.Module):
    def __init__(self, input_feature_num):
        super(NeuralNetwork, self).__init__()
        input_feature_num  = input_feature_num
        hidden_layer_width = 256
        output_feature_num = 36
        self.Res_stack = nn.Sequential(
            nn.Linear(input_feature_num, hidden_layer_width),
            nn.BatchNorm1d(hidden_layer_width) ,
            nn.ReLU(),
            nn.Linear(hidden_layer_width, hidden_layer_width),
            nn.BatchNorm1d(hidden_layer_width) ,
            nn.ReLU(),
            nn.Linear(hidden_layer_width, hidden_layer_width),
            nn.BatchNorm1d(hidden_layer_width) ,
            nn.ReLU(),
            nn.Linear(hidden_layer_width, hidden_layer_width),
            nn.BatchNorm1d(hidden_layer_width) ,
            nn.ReLU(),
            nn.Linear(hidden_layer_width, output_feature_num)
        ) 
        
    def forward(self, x): 
        return self.Res_stack(x)

    
class NNRTMC_NN: 
    def __init__(self,  device, nor_para, Ak, Bk, input_feature_num, model_dict = None):
        """
        model_dict_path is the saved NN model state dict.
        """
        self.device = device
        # initial ANN
        self.NN_model = NeuralNetwork(input_feature_num).to(self.device) 
        if model_dict is not None:
            self.NN_model.load_state_dict(model_dict) 
        self.optimizer = torch.optim.Adam(self.NN_model.parameters(), lr=1e-4, 
                                          betas=(0.9, 0.999), weight_decay=1e-4) 
        self.loss_fn   = torch.nn.MSELoss()
        self.nor_para  = {key: torch.tensor(value).to(self.device) for key, value in nor_para.items()} 
        
        # parameters
        self.Ak = torch.tensor(Ak).to(self.device)
        self.Bk = torch.tensor(Bk).to(self.device)
        
        self.C_p = 1004       # J/kg/K 
        self.g   = 9.8        # m/s^2 
        
    def train(self, train_batch_indice, input_torch, output_torch, eng_loss='N', optimizer=None):
        if optimizer is None: 
            optimizer = self.optimizer
        self.NN_model.train() # enter training mode
        for i, batch_ids in enumerate(train_batch_indice):
            X, Y = input_torch[batch_ids,:], output_torch[batch_ids,:]
            # Compute prediction error
            Y_pred = self.NN_model(X) 
            loss_data = self.loss_fn(Y_pred, Y)
            loss_ener = self.loss_energy(X, Y_pred)
            loss = loss_data
            if eng_loss=='Y': loss += loss_ener*1e-3
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return [loss_data.item(), loss_ener.item()]
    
    def test_loss(self, test_indice, input_torch, output_torch):
        self.NN_model.eval() # enter evaluation mode
        with torch.no_grad():
            X, Y = input_torch [test_indice,:], output_torch[test_indice,:]
            Y_pred = self.NN_model(X) 
            loss_data = self.loss_fn(Y_pred, Y).item() 
            loss_ener = self.loss_energy(X, Y_pred).item() 
        return [loss_data, loss_ener]
 
    def loss_energy(self,  input_data, output_pred ): 
        F_net, sum_Cphr_gdp = self.energy_flux_HR(input_data , output_pred) 
        return self.loss_fn(F_net,sum_Cphr_gdp) 

    def energy_flux_HR(self, Input, Output):  
        F_toa_up = Output[:,2]/self.nor_para['output_scale'][2] + self.nor_para['output_offset'][2]
        F_sfc_do = Output[:,0]/self.nor_para['output_scale'][0] + self.nor_para['output_offset'][0]
        F_sfc_up = Output[:,1]/self.nor_para['output_scale'][1] + self.nor_para['output_offset'][1]
        F_net = F_sfc_up - F_sfc_do - F_toa_up
        HR = Output[:,3:]/self.nor_para['output_scale'][3:] + self.nor_para['output_offset'][3:]            #  K/s 
        ps = Input[:,None,0]/self.nor_para['input_scale'][0] + self.nor_para['input_offset'][0]
        P_lev = self.return_dP_AM4_plev(ps)                #  Pa 
        dP = (P_lev[:,1:] - P_lev[:,:33])
        sum_Cphr_gdp = self.C_p/self.g * (HR*dP).sum(axis=-1) 
        return F_net, sum_Cphr_gdp
    
    def predict(self, input_X):
        self.NN_model.eval() # enter evaluation mode
        with torch.no_grad():
            pred = self.NN_model(input_X).cpu() # analyize on cpu
        return pred
 
    def return_dP_AM4_plev(self, ps): 
        """
        ps: Pa

        return: 
        """ 
        p_int = self.Ak + self.Bk*ps
        # if not np.all(np.diff(p_int)>0):
        #     raise Exception(f'Input [ps] is not valid. Check units!')
        return p_int
    
    def save_model_restart(self, PATH, loss_array, data_info, nomral_para):
        '''
        Saving & Loading Model for Inference
        Save/Load state_dict (Recommended)
        Load: 
        NNRTMC_solver= NNRTMC(device, model_dict_path = model_state_dict)
        '''
        # get a cpu copy of state dict
        model_state_dict = {key: value.cpu() for key, value in self.NN_model.state_dict().items()}
        # Save:
        torch.save({ 'data_info'       : data_info,
                     'loss_array'      : loss_array,
                     'nor_para'        : nomral_para,
                     'model_state_dict': model_state_dict
                   } , PATH) 

######################################################
# common functions to split the training and test data
# 
from NNRTMC_lw_utils import  split_train_test_sample, \
draw_batches, data_std_normalization, print_key_results, return_exp_dir

    
######################################################
def custom_trainning(NNRTMC_solver, lr, loss, epochs, batch_size, de_save,
                     input_torch, output_torch, indice_train, indice_test, 
                     eng_loss, device, rng):
    # update lr based on test loss
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        NNRTMC_solver.optimizer, mode='min', factor=0.2, patience=100, threshold=1e-3, 
        threshold_mode='rel', cooldown=100, min_lr=0, eps=1e-08, verbose=True) 
    NNRTMC_solver.optimizer.param_groups[0]['lr'] = lr
    ######################################################
    # set training hyperparameter here
    ######################################################
    sta_time = time.time()
    for t in range(epochs): 
        batch_indice_train = draw_batches(indice_train, batch_size, rng, device, replace=False)
        lossv     = NNRTMC_solver.train(batch_indice_train, input_torch, output_torch, eng_loss=eng_loss)
        lossvtest = NNRTMC_solver.test_loss(indice_test, input_torch,  output_torch)
        lr_scheduler.step(lossvtest[0]+lossvtest[1]) # update lr based on test loss
        if t % de_save == 0:
            used_time = time.time() - sta_time  
            print( f"Epoch {t+1:06d} |train loss: {lossv[0]:8.2e} {lossv[1]:8.2e} | test loss: {lossvtest[0]:8.2e} {lossvtest[1]:8.2e}  "
                  +f"| ~ {used_time:3.0f}s | eta {int(used_time*((epochs-t)/de_save/60)) :3d} min")
            sta_time = time.time()
            loss.append([[t+1]+lossv+lossvtest]) # append epochs, loss, test loss
            # early stop 
            if NNRTMC_solver.optimizer.param_groups[0]['lr'] < 1e-7:
                print(f"Meet early stop criteria LR = {NNRTMC_solver.optimizer.param_groups[0]['lr']} < 1e-7" )
                print("End training")
                break
                
import xarray as xr 
from get_AM4_data_lw import get_AM4_data_lw
import argparse, sys

if __name__ == '__main__': 
    torch.cuda.set_device(0) # select gpu_id, default 0 means the first GPU
    device = f'cuda:{torch.cuda.current_device()}'
    # set random generator
    rng = np.random.default_rng(12345)
    torch.manual_seed(12345)
    # rng = np.random.default_rng()
    
    ######################################################
    # set exp name and runs
    # read sky_cond and eng_loss from run command
    parser=argparse.ArgumentParser()
    parser.add_argument("--sky_cond", help="sky condition: af, csaf")
    parser.add_argument("--eng_loss", help="minimize the energy loss: Y/N")
    args=parser.parse_args()
    sky_cond = args.sky_cond
    eng_loss = args.eng_loss 
    
    Exp_name = f'lw_{sky_cond}_Li10Relu_E{eng_loss}' 
    work_dir = '/tigress/cw55/work/2022_radi_nn/NN_AM4/work/'
    total_run_num  = 4
    epochs = 2000
    de_save = 200
    
    ######################################################
    # create dir for first run or load restart file
    run_num, exp_dir = return_exp_dir(work_dir, Exp_name)
    # copy script to experiment dir for reference
    try:
        ossyscmd = f'cp {os.path.abspath(__file__)} {exp_dir}/train_script{run_num:02d}.py' 
        os.system(ossyscmd) 
    except: pass 
    
    # get data and do normalization
    if run_num == 1:  
        nor_para = None
        model_state_dict = None
        lr_sta = 1e-3
    else:   # load restart file
        PATH_last =  exp_dir+f'/restart.{run_num-1:02d}.pth'
        restart_data = torch.load(PATH_last)  # load exist results and restart training
        print(f'restart from: {PATH_last}')
        # read training dataset, nor_para, model parameteres
        nor_para = restart_data['nor_para']
        model_state_dict = restart_data['model_state_dict']
        lr_sta = 1e-4

    ######################################################
    # load data from AM4 runs
    out_filelist = [f'/scratch/gpfs/cw55/NNRTMC_data/AM4_v2/20000101.fluxes.tile{_}.nc' for _ in range(1,7)]
    inp_filelist = [f'/scratch/gpfs/cw55/NNRTMC_data/AM4_v2/20000101.new_offline_input.tile{_}.nc' for _ in range(1,7)]
    input_array_ori, output_array_ori = \
    get_AM4_data_lw(out_filelist, inp_filelist, condition=sky_cond, month_sel = None, day_sel = [1, 7]) 
    
    hybrid_p_sigma_para = xr.open_dataset('/tigress/cw55/data/NNRTMC_dataset/AM4_pk_bk_202207.nc')
    A_k = hybrid_p_sigma_para.ak.values[None,:]
    B_k = hybrid_p_sigma_para.bk.values[None,:]
    
    print(f"Features| Input: {input_array_ori.shape[1]},  Output: {output_array_ori.shape[1]}")
    nor_para, input_array, output_array   = data_std_normalization(input_array_ori, output_array_ori, nor_para)
     
    # divide the training and test data here
    # this would be different if restart the training process (be careful!)
    ind_train, ind_test = split_train_test_sample(output_array.shape[0], test_ratio=0.3, rng=rng) 
    
    ######################################################
    # move all data to GPU to accelerate training
    input_torch  = torch.tensor(input_array,  dtype=torch.float32).to(device)
    output_torch = torch.tensor(output_array, dtype=torch.float32).to(device) 
    
    ######################################################
    # initialize model
    NNRTMC_solver = NNRTMC_NN(device, nor_para, A_k, B_k, input_array.shape[1], model_state_dict)  
    # training 
    for i in range(run_num, total_run_num+1): 
        loss = []
        batch_size = max(8000, 8000*i**2)
        print(f'Train info >> run: {i} lr_sta: {lr_sta:7.1e}, batch size: {batch_size}')
        custom_trainning(NNRTMC_solver, lr_sta, loss, epochs, batch_size, de_save,\
                         input_torch, output_torch, \
                         ind_train, ind_test, eng_loss, device, rng )
        loss_array = np.array(loss).squeeze().T  
        ######################################################
        # save model state dict and data normalization info
        data_info = out_filelist+inp_filelist
        PATH =  exp_dir+f'/restart.{i:02d}.pth'
        print('OUTPUT is saved at: '+PATH)
        NNRTMC_solver.save_model_restart(PATH, loss_array, data_info, nor_para)
        print_key_results( NNRTMC_solver, input_torch[ind_test,:], output_array[ind_test,:], nor_para)
        lr_sta = 1e-4
        print(f'{Exp_name} Finished: run {i}!')  
        
    print('All runs finished. Increase <run_num> if you need to continue to train the model.')

    # move slurm log to work dir
    job_id = int(os.environ["SLURM_JOB_ID"])
    ossyscmd = f'cp slurm-{job_id}.out {exp_dir}/' 
    print(ossyscmd)
    os.system(ossyscmd) 
    

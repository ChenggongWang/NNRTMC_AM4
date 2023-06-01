import numpy as np  
import pickle
import math
import os
import torch
from torch import nn   

torch.set_float32_matmul_precision('high')

######################################################
# NN module includes:
# NN model structure 
class NeuralNetwork(nn.Module):
    def __init__(self, input_feature_num, output_feature_num, hidden_layer_width):
        super(NeuralNetwork, self).__init__()
        input_feature_num  = input_feature_num
        hidden_layer_width = hidden_layer_width
        output_feature_num = output_feature_num
        self.Stack = nn.Sequential(
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
            nn.Linear(hidden_layer_width, output_feature_num),
            nn.Sigmoid()
        ) 
        
    def forward(self, x): 
        return self.Stack(x)


    
class NNRTMC_NN: 
    def __init__(self,  device, nor_para, Ak, Bk, 
                 input_feature_num, output_feature_num, hidden_layer_width, model_dict = None):
        """
        model_dict_path is the saved NN model state dict.
        """
        self.device = device
        # initial ANN
        self.NN_model = NeuralNetwork(input_feature_num, output_feature_num, hidden_layer_width).to(self.device)  
        
        if model_dict is not None:
            self.NN_model.load_state_dict(model_dict) 
        self.optimizer = torch.optim.Adam(self.NN_model.parameters(), lr=1e-4, 
                                          betas=(0.9, 0.999), weight_decay=1e-5) 
        self.loss_fn   = torch.nn.MSELoss()
        self.nor_para  = {key: torch.tensor(value).to(self.device) for key, value in nor_para.items()} 
        
        # parameters
        self.Ak = torch.tensor(Ak).to(self.device)
        self.Bk = torch.tensor(Bk).to(self.device)
        self.C_p = 1004.64      # Specific heat capacity [J/kg/K] 
        self.g   = 9.8          # GAV [m/s^2 ] 
        self.sigma = 5.6734e-8  # Stefan-Boltzmann constant [W/m^2/K^4]
        
    
    def predict(self, input_X):
        self.NN_model.eval() # enter evaluation mode
        with torch.no_grad():
            pred = self.NN_model(input_X).cpu().numpy() # analyize on cpu
        return pred
 
    def return_dP_AM4_plev(self, ps): 
        """
        ps: Pa

        return: 
        """ 
        P_lev = self.Ak + self.Bk*ps
        # if not np.all(np.diff(p_int)>0):
        #     raise Exception(f'Input [ps] is not valid. Check units!') 
        dP = (P_lev[:,1:] - P_lev[:,:33])
        return dP
    
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

class NNRTMC_NN_sw(NNRTMC_NN): 
    def train(self, train_batch_indice, input_torch, output_torch, 
              rsdt_torch, eng_loss_frac=None, optimizer=None):
        if optimizer is None: 
            optimizer = self.optimizer
        self.NN_model.train() # enter training mode
        for i, batch_ids in enumerate(train_batch_indice):
            X, Y, rsdt = input_torch [batch_ids,:], \
                         output_torch[batch_ids,:], \
                         rsdt_torch  [batch_ids  ]
            # Compute prediction error
            Y_pred = self.NN_model(X) 
            loss_data = self.loss_fn(Y_pred, Y)
            loss_ener = self.loss_energy(X, Y_pred, rsdt)
            loss = loss_data
            if eng_loss_frac != None: 
                loss += loss_ener*eng_loss_frac
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return [loss_data.item(), loss_ener.item()]
     
    def test_loss(self, test_indice, input_torch, output_torch, rsdt_torch):
        self.NN_model.eval() # enter evaluation mode
        with torch.no_grad():
            X, Y, rsdt = input_torch [test_indice,:], \
                         output_torch[test_indice,:], \
                         rsdt_torch  [test_indice  ]
            Y_pred = self.NN_model(X) 
            loss_data = self.loss_fn(Y_pred, Y).item() 
            loss_ener = self.loss_energy(X, Y_pred, rsdt).item() 
        return [loss_data, loss_ener]
 
    def loss_energy(self,  input_data, output_pred, rsdt): 
        F_net, sum_Cphr_gdp = self.energy_flux_HR(input_data , output_pred, rsdt) 
        return self.loss_fn(F_net,sum_Cphr_gdp) 

    def energy_flux_HR(self, Input, Output, rsdt): 
        # ! rsdt is the last input var
        Out_unnor = Output/self.nor_para['output_scale'] + self.nor_para['output_offset']
        F_sw_toa_do = rsdt
        F_sw_toa_up = Out_unnor[:,0]
        F_sw_sfc_do = Out_unnor[:,1]
        F_sw_sfc_up = Out_unnor[:,2]
        HR = Out_unnor[:,3:]    #  K/s 
        F_net = F_sw_toa_do*(1 - F_sw_toa_up + F_sw_sfc_up - F_sw_sfc_do) 
        ps = Input[:,None,0]/self.nor_para['input_scale'][0] + self.nor_para['input_offset'][0] 
        dP = self.return_dP_AM4_plev(ps)    #  Pa 
        sum_Cphr_gdp = self.C_p/self.g * (HR*dP).sum(axis=-1) * F_sw_toa_do
        return F_net, sum_Cphr_gdp
    
class NNRTMC_NN_lw(NNRTMC_NN): 
    def train(self, train_batch_indice, input_torch, output_torch, eng_loss_frac=None, optimizer=None):
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
            if eng_loss_frac != None: 
                loss += loss_ener*eng_loss_frac
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
        Out_unnor = Output/self.nor_para['output_scale'] + self.nor_para['output_offset']
        F_sfc_do = Out_unnor[:,0]
        F_toa_up = Out_unnor[:,1]
        Ts = Input[:,34]/self.nor_para['input_scale'][34] + self.nor_para['input_offset'][34]
        F_sfc_up = self.sigma*Ts**4  # formula from AM4
        F_net = F_sfc_up - F_sfc_do - F_toa_up
        HR = Out_unnor[:,2:]    #  K/s 
        ps = Input[:,None,0]/self.nor_para['input_scale'][0] + self.nor_para['input_offset'][0]
        dP = self.return_dP_AM4_plev(ps)    #  Pa 
        sum_Cphr_gdp = self.C_p/self.g * (HR*dP).sum(axis=-1) 
        return F_net, sum_Cphr_gdp
    
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

def data_std_normalization_lw(input_array, output_array, nomral_para = None):
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
        output_scale    = (output_array.max(axis=0)-output_array.min(axis=0))
        # check not varying input
        if np.any(np.isclose(output_scale,0,atol=1e-10)):
            print(f'Warning: {np.isclose(output_scale,0).sum()} output feature(s) is fixed!')
            output_scale     = np.where(np.isclose(output_scale,0), 1, output_scale) 
        # Scale output to ~[0.05, 0.95]
        output_offset   = output_array.min(axis=0) - 0.05 * output_scale
        output_scale    = 1/(1.1 * output_scale)
        nomral_para = {'input_scale'   : input_scale, 
                       'input_offset'  : input_offset,
                       'output_scale'  : output_scale,
                       'output_offset' : output_offset} 
    # do normalization
    input_array  = (input_array  - nomral_para['input_offset' ])*nomral_para['input_scale' ]
    output_array = (output_array - nomral_para['output_offset'])*nomral_para['output_scale']
    return nomral_para, input_array, output_array 

def data_std_normalization_sw(input_array, output_array, rsdt_array, nomral_para = None, remove_night=True):
    sample_size = input_array.shape[0]
    print(f"Total data size: {sample_size}")
    day_ind = np.argwhere(~np.isclose(rsdt_array,0, atol=1e-1)).squeeze()
    # remove data that rsdt == 0 (night, no shortwave transfer) 
    sample_size2 = day_ind.shape[0]
    if remove_night == True:
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
    # print key results  # offset in pred and true cancels
    error = (pred_output - true_output)/ normal_para['output_scale'] 
    # print('error.shape')
    # print(error.shape)
    error[:,3:] = error[:,3:]*86400  
    RMSE = ((error**2).mean(axis=0))**0.5
    MAE  = abs(error).mean(axis=0)
    Bias = error.mean(axis=0)
    print('Validation: RMSE')  
    print(np.array2string(RMSE,formatter={'float_kind':lambda x: "%9.2e" % x}))
    print('Validation:       MAE')  
    print(np.array2string(MAE ,formatter={'float_kind':lambda x: "%9.2e" % x}))
    print('Validation:            Bias')  
    print(np.array2string(Bias,formatter={'float_kind':lambda x: "%9.2e" % x}))

def print_key_results_lw(pred_output, true_output, normal_para): 
######################################################
    # print key results  # offset in pred and true cancels
    error = (pred_output - true_output)/ normal_para['output_scale'] 
    # print('error.shape')
    # print(error.shape)
    error[:,2:] = error[:,2:]*86400  
    RMSE = ((error**2).mean(axis=0))**0.5
    MAE  = abs(error).mean(axis=0)
    Bias = error.mean(axis=0)
    print('Validation: RMSE')  
    print(np.array2string(RMSE,formatter={'float_kind':lambda x: "%9.2e" % x}))
    print('Validation:       MAE')  
    print(np.array2string(MAE ,formatter={'float_kind':lambda x: "%9.2e" % x}))
    print('Validation:            Bias')  
    print(np.array2string(Bias,formatter={'float_kind':lambda x: "%9.2e" % x}))

def return_exp_dir(parent_dir, Exp_name, create_dir=True): 
    
    if not os.path.exists(parent_dir): 
        raise Exception('Parent work directory do no exits! Please create: \n' + parent_dir)
    exp_dir = parent_dir+Exp_name
    if create_dir == True:
        if not os.path.exists(exp_dir): 
            print('>>| First run! '+f'Create experiment dir at: \n{exp_dir}')
            # create work dir for experiment to store results 
            os.mkdir(exp_dir)  
            # copy script to experiment dir for reference
            ossyscmd = f'cp {os.path.abspath(__file__)} {exp_dir}/ '
            print(ossyscmd)
            os.system(ossyscmd) 
    for run_num in range(1,100): 
        # PATH for the save the restart/result file
        PATH =  exp_dir+f'/model0_restart.{run_num:02d}.pth'
        # print(PATH)
        if not os.path.exists(PATH): 
            return run_num, exp_dir
    raise Exception('Already exist 99 runs!') 

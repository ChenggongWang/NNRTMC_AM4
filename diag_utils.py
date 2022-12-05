import math
import torch
import numpy as np
import xarray as xr
import subprocess

def batch_index_sta_end(sample_size, batch_size):
    '''
    divide the whole sample into batchs
    return sta and end indices
    '''
    n_batch = math.ceil(sample_size/batch_size)
    batch_ind = []
    sta=0
    for i in range(n_batch-1):
        end = sta + batch_size
        batch_ind.append([sta, end])
        sta = end
    batch_ind.append([sta,sample_size])
    return batch_ind

def pred_NN_batch(input_arr, output_arr, model, nor_para, device):
    '''call NN to predict in batch to reduce memory use'''
    sample_size = input_arr.shape[0]
    batch_size = 10000*100 # 1 million columns in one batch
    batch_ind = batch_index_sta_end(sample_size, batch_size)
    pred_out_array = np.empty_like(output_arr)
    eng_err_array = np.empty_like(output_arr[:,0])
    for sta, end in batch_ind:
        # print(sta,end)
        # move all data to GPU
        input_torch  = torch.tensor(input_arr[sta:end,:], dtype=torch.float32).to(device)
        output_torch = torch.tensor(output_arr[sta:end,:], dtype=torch.float32).to(device)
        # get output from NN
        pred_out_batch  = model.predict(input_torch)
        # check energy error
        F_net, sum_Cphr_gdp = model.energy_flux_HR(input_torch , pred_out_batch.to(device)) 
        eng_err_array[sta:end]  =  (F_net - sum_Cphr_gdp).cpu().numpy() 
        # denormalize
        pred_out_array[sta:end,:] = pred_out_batch.cpu().numpy()/nor_para['output_scale'] + nor_para['output_offset'] 
        del input_torch, output_torch # release GPU memory
        torch.cuda.empty_cache()       
    return pred_out_array, eng_err_array

def create_6tiles_sw(ds_coords, predi, error, eng_err, exp_dir, output_name): 
    '''save sw output as 6 nc files'''
    print('Creating 6 tiles ... ', end=' ')
    time    = ds_coords[0]['time']
    pfull   = ds_coords[0]['pfull']
    grid_yt = ds_coords[0]['grid_yt']
    grid_xt = ds_coords[0]['grid_xt']
    txy_size = time.shape[0]*grid_xt.shape[0]*grid_yt.shape[0]
    if (predi.shape[0]/txy_size) !=6:
        raise Exception('wrong size of output, pls check data dimension')
    for i in range(6):
        output_nn_ti = predi[txy_size*i:txy_size*(i+1)]
        output_nn_ti = output_nn_ti.reshape(time.shape[0],
                                            grid_yt.shape[0],
                                            grid_xt.shape[0],
                                            -1)
        output_err_ti = error[txy_size*i:txy_size*(i+1)]
        output_err_ti = output_err_ti.reshape(time.shape[0],
                                              grid_yt.shape[0],
                                              grid_xt.shape[0],
                                              -1)
        eng_err_ti = eng_err[txy_size*i:txy_size*(i+1)]
        eng_err_ti = eng_err_ti.reshape(time.shape[0],
                                        grid_yt.shape[0],
                                        grid_xt.shape[0] )
        ds = xr.Dataset( 
                        {'rsut':      (["time", "grid_yt", "grid_xt"], output_nn_ti[:,:,:,0]),
                         'rsds':      (["time", "grid_yt", "grid_xt"], output_nn_ti[:,:,:,1]),
                         'rsus':      (["time", "grid_yt", "grid_xt"], output_nn_ti[:,:,:,2]),
                         'tntrs':     (["time", "grid_yt", "grid_xt", "pfull"], output_nn_ti[:,:,:,3:]),
                         'err_rsut':  (["time", "grid_yt", "grid_xt"], output_err_ti[:,:,:,0]),
                         'err_rsds':  (["time", "grid_yt", "grid_xt"], output_err_ti[:,:,:,1]),
                         'err_rsus':  (["time", "grid_yt", "grid_xt"], output_err_ti[:,:,:,2]),
                         'err_tntrs': (["time", "grid_yt", "grid_xt", "pfull"], output_err_ti[:,:,:,3:]),
                         'err_eng':   (["time", "grid_yt", "grid_xt"], eng_err_ti),
                        },
                        coords=ds_coords[i]
                        )
        var_4d_name = ['tntrs', 'err_tntrs']
        var_3d_name = ['rsut', 'rsds','rsus',
                       'err_rsut', 'err_rsds','err_rsus','err_eng']
        for _var in var_4d_name:
            ds[_var] = ds[_var].transpose("time", "pfull", "grid_yt", "grid_xt")
        for _var in var_3d_name:
            ds[_var] = ds[_var].transpose("time", "grid_yt", "grid_xt")
        encoding={}
        for _var in list(ds.keys()):
            encoding[_var] = {"dtype":"float32", "_FillValue":None} 
        ds.to_netcdf(exp_dir+f'/NN_pred/{output_name}.tile{i+1}.nc', encoding=encoding)
        var_list = list(ds.keys())
        ds.close()
    print('Done.')
    return var_list

def create_6tiles_lw(ds_coords, predi, error, eng_err, exp_dir, output_name): 
    '''save lw output as 6 nc files'''
    print('Creating 6 tiles ... ', end='  ')
    time    = ds_coords[0]['time']
    pfull   = ds_coords[0]['pfull']
    grid_yt = ds_coords[0]['grid_yt']
    grid_xt = ds_coords[0]['grid_xt']
    txy_size = time.shape[0]*grid_xt.shape[0]*grid_yt.shape[0]
    if (predi.shape[0]/txy_size) !=6:
        raise Exception('wrong size of output, pls check data dimension')
    for i in range(6):
        output_nn_ti = predi[txy_size*i:txy_size*(i+1)]
        output_nn_ti = output_nn_ti.reshape(time.shape[0],
                                            grid_yt.shape[0],
                                            grid_xt.shape[0],
                                            -1)
        output_err_ti = error[txy_size*i:txy_size*(i+1)]
        output_err_ti = output_err_ti.reshape(time.shape[0],
                                              grid_yt.shape[0],
                                              grid_xt.shape[0],
                                              -1)
        eng_err_ti = eng_err[txy_size*i:txy_size*(i+1)]
        eng_err_ti = eng_err_ti.reshape(time.shape[0],
                                        grid_yt.shape[0],
                                        grid_xt.shape[0] )
        ds = xr.Dataset( 
                        {'rlds':      (["time", "grid_yt", "grid_xt"], output_nn_ti[:,:,:,0]),
                         'rlus':      (["time", "grid_yt", "grid_xt"], output_nn_ti[:,:,:,1]),
                         'rlut':      (["time", "grid_yt", "grid_xt"], output_nn_ti[:,:,:,2]),
                         'tntrl':     (["time", "grid_yt", "grid_xt", "pfull"], output_nn_ti[:,:,:,3:]),
                         'err_rlds':  (["time", "grid_yt", "grid_xt"], output_err_ti[:,:,:,0]),
                         'err_rlus':  (["time", "grid_yt", "grid_xt"], output_err_ti[:,:,:,1]),
                         'err_rlut':  (["time", "grid_yt", "grid_xt"], output_err_ti[:,:,:,2]),
                         'err_tntrl': (["time", "grid_yt", "grid_xt", "pfull"], output_err_ti[:,:,:,3:]),
                         'err_eng':   (["time", "grid_yt", "grid_xt"], eng_err_ti),
                        },
                        coords=ds_coords[i]
                        )
        var_4d_name = ['tntrl', 'err_tntrl']
        var_3d_name = ['rlds', 'rlus','rlut',
                       'err_rlds', 'err_rlus','err_rlut', 'err_eng']
        for _var in var_4d_name:
            ds[_var] = ds[_var].transpose("time", "pfull", "grid_yt", "grid_xt")
        for _var in var_3d_name:
            ds[_var] = ds[_var].transpose("time", "grid_yt", "grid_xt")
        encoding={}
        for _var in list(ds.keys()):
            encoding[_var] = {"dtype":"float32", "_FillValue":None} 
        ds.to_netcdf(exp_dir+f'/NN_pred/{output_name}.tile{i+1}.nc', encoding=encoding)
        var_list = list(ds.keys())
        ds.close()
    print('Done.')
    return var_list

def regrid_6tile2latlon(var_list,exp_dir,root,output_name):
    '''run fregrid in subshell'''
    print('Calling fregrid in subshell ... ', end='  ')
    var_list_str = ','.join(var_list)
    cmd = f"cd {exp_dir}/NN_pred;"\
         +f"cp {root}/regrid/* . ;"\
         +f"bash run_fregrid.sh {output_name} {var_list_str};"\
         +f"rm {output_name}.tile*"
    tmp=subprocess.run([cmd], shell=True, capture_output=True)
    if tmp.stderr.decode("utf-8") != '': 
        raise Exception(tmp.stderr.decode("utf-8")) 
    # print(tmp.stdout.decode("utf-8"))
    regrid_file_path = exp_dir+f'/NN_pred/{output_name}.nc'
    print('Done.')
    return regrid_file_path
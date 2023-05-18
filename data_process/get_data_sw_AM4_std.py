import xarray as xr
import numpy as np
import time

def get_data_sw_AM4(filelist, condition='cs', month_sel = None, day_sel = None, return_coords = False):
    sta_time = time.time()
    # sample data by month and day
    if month_sel == None:
        month_sel = [1,2,3,4,5,6,7,8,9,10,11,12] 
    if day_sel == None: 
        day_sel = [15] 
    print("Data files:")
    print(filelist)
    print(f"Data selection:\n    Month: {month_sel} \n    Day: {day_sel} \nReading data...", end=' ')
        
    inp_var_name_csaf   = ['nn_ta'  ,'nn_ts', 'nn_sphum' ,'nn_o3' ,
                           'cosz', 'visible_direct_albedo','visible_diffuse_albedo',
                           'infrared_direct_albedo','infrared_diffuse_albedo' ]
    inp_var_name_cloud  = ['stratiform_droplet_number' ,'stratiform_cloud_fraction' ,
                           'stratiform_liquid_content' ,'stratiform_ice_content'    ,
                           'shallow_droplet_number'    ,'shallow_cloud_fraction'    ,
                           'shallow_liquid_content'    ,'shallow_ice_content'       ] 
    # print(condition)
    if condition == 'cs':
        inp_var_name = inp_var_name_csaf 
    elif condition == 'all':
        inp_var_name = inp_var_name_csaf + inp_var_name_cloud 
    else: 
        raise Exception(f"condition {condition} is not configured") 
        
    out_var_name_cs = ['swup_toa_clr', 'swdn_sfc_clr', 'swup_sfc_clr', 'tdt_sw_clr']
    out_var_name    = ['swup_toa',     'swdn_sfc',     'swup_sfc',     'tdt_sw']
    if condition == 'cs':
        out_var_name = out_var_name_cs
    elif condition == 'all':
        pass
    else: raise Exception(f"condition {condition} is not configured")
    
    input_array_list = []
    output_array_list = []
    rsdt_array_list = []

    id_list_drop = []
    input_array_list_drop = []
    tiles_coords = []
    # at least 6 tiles
    for tile_i in range(len(filelist)): 
        print (tile_i, end=' ')
        # read data from files  
        # inputs
        ds = xr.open_dataset(filelist[tile_i])
        time_sel = ds.time.dt.month.isin(month_sel)&ds.time.dt.day.isin(day_sel)
        ds_inp = ds.isel(time=time_sel) 
        ds_out = ds_inp
        #pressure
        var_ps = ds_inp['nn_plevel'].isel(phalf=-1).load() # this will be included in input by default 
        input_sf =[[var_ps.stack(txy=("time","grid_yt", "grid_xt")).values],] 
        for _var in inp_var_name:  # for all vars
            tmp = ds_inp[_var].stack(txy=("time","grid_yt", "grid_xt")).fillna(0).values
            if len(tmp.shape)==1:
                input_sf.append(tmp[None,:]) # addtional dim
            else:
                input_sf.append(tmp)
        # concatenate all varibile into one big matrix
        input_sf = np.concatenate(input_sf)
        
        # outputs 
        # save rsdt as a seperate data, phalf[0] = TOA
        rsdt = ds_out['swdn_toa'].stack(txy=("time","grid_yt", "grid_xt")).fillna(0).values
        # inverse of rsdt
        rsdt_r = np.where(np.isclose(rsdt,0,rtol=1e-05, atol=1e-1,), 0, 1/rsdt)
        output_sf = []
        for _var in out_var_name:  # for all vars
            tmp = ds_out[_var].stack(txy=("time","grid_yt", "grid_xt")).fillna(0).values
            if len(tmp.shape)==1:
                tmp = (tmp[None,:]) # addtional dim 
            # normalized by rsdt
            tmp = tmp*rsdt_r
            output_sf.append(tmp)
            
        # concatenate all varibile into one big matrix
        output_sf = np.concatenate(output_sf) 
        # get all tiles/files from different files
        input_array_list.append(input_sf)
        output_array_list.append(output_sf)
        rsdt_array_list.append(rsdt)
        tiles_coords.append(ds_out.coords)
    #concatenate all tiles/files and transpose the matrix
    input_array_ori  = np.concatenate( input_array_list,axis=1).T
    output_array_ori = np.concatenate(output_array_list,axis=1).T
    rsdt_array_ori   = np.concatenate(rsdt_array_list)
    print(f'\nRead data done. Use time: {time.time() - sta_time: 3.0f}s')
    if return_coords:
        return input_array_ori, output_array_ori, rsdt_array_ori, tiles_coords
    else:
        return input_array_ori, output_array_ori, rsdt_array_ori

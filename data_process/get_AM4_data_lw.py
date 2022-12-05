import xarray as xr
import numpy as np 

def get_AM4_data_lw(out_filelist, inp_filelist, condition='csaf', month_sel = None, day_sel = None, return_coords = False):
    # sample data by month and day
    if month_sel == None:
        month_sel = [1,2,3,4,5,6,7,8,9,10,11,12]
    if day_sel == None: 
        day_sel = [1]
    print("Data files:")
    print(out_filelist, inp_filelist)
    print(f"Data selection:\n    Month: {month_sel} \n    Day: {day_sel} \nReading data...", end=' ')
    
    input_array_list = []
    output_array_list = [] 
    tiles_coords = []
    for tile_i in range(len(out_filelist)): 
        print (tile_i, end=' ')
        # read data from files  
        # inputs
        ds = xr.open_dataset(inp_filelist[tile_i])
        time_sel = ds.time.dt.month.isin(month_sel)&ds.time.dt.day.isin(day_sel)
        ds_inp = ds.isel(time=time_sel)
        var_ps = ds_inp['level_pressure'].isel(phalf=-1) # this will be included in input by default
        inp_var_name_csaf   = ['level_temperature'  ,'surface_temperature', 'water_vapor' ,'ozone' ]
        inp_var_name_cloud  = ['stratiform_droplet_number' ,'stratiform_cloud_fraction' ,
                               'stratiform_liquid_content' ,'stratiform_ice_content'    ,
                               'shallow_droplet_number'    ,'shallow_cloud_fraction'    ,
                               'shallow_liquid_content'    ,'shallow_ice_content'       ,
                               'strat_size_drop'           ,'shallow_size_drop'          ]
        # inp_var_name_aersol   = ['stratiform_droplet_number' ,'stratiform_cloud_fraction' ,
        #                        'stratiform_liquid_content' ,'stratiform_ice_content'    ,
        #                        'shallow_droplet_number'    ,'shallow_cloud_fraction'    ,
        #                        'shallow_liquid_content'    ,'shallow_ice_content'       ,
        #                        'strat_size_drop'           ,'shallow_size_drop'          ]
        if condition == 'csaf':
            inp_var_name = inp_var_name_csaf 
        elif condition == 'af':
            inp_var_name = inp_var_name_csaf + inp_var_name_cloud 
        else: raise Exception(f"condition {condition} is not configured")
        
        input_sf =[[var_ps.stack(txy=("time","grid_yt", "grid_xt")).values],]
        for _var in inp_var_name:  # for all vars
            tmp = ds_inp[_var].stack(txy=("time","grid_yt", "grid_xt")).fillna(0).values
            if len(tmp.shape)==1:
                input_sf.append(tmp[None,:]) # addtional dim
            else:
                input_sf.append(tmp)
        input_sf = np.concatenate(input_sf)
        
        # outputs
        ds = xr.open_dataset(out_filelist[tile_i])
        ds_out = ds.isel(time=time_sel)
        out_var_list_csaf   = [ds_out['rldcsaf'  ].isel(phalf=-1), 
                               ds_out['rlucsaf'  ].isel(phalf=-1),
                               ds_out['rlucsaf'  ].isel(phalf=0 ),
                               ds_out['tntrlcsaf'] ] # no aerosols
        out_var_list_af     = [ds_out['rldaf'    ].isel(phalf=-1), 
                               ds_out['rluaf'    ].isel(phalf=-1),
                               ds_out['rluaf'    ].isel(phalf=0 ),
                               ds_out['tntrlaf'  ] ] # no aerosols 
        if condition == 'csaf':
            out_var_list = out_var_list_csaf 
        elif condition == 'af':
            out_var_list = out_var_list_af  
        else: raise Exception(f"condition {condition} is not configured")
        output_sf = [] 
        for _var in out_var_list:  # for all vars
            tmp = _var.stack(txy=("time","grid_yt", "grid_xt")).fillna(0).values
            if len(tmp.shape)==1:
                output_sf.append(tmp[None,:]) # addtional dim
            else:
                output_sf.append(tmp)
        output_sf = np.concatenate(output_sf) 
        input_array_list.append(input_sf)
        output_array_list.append(output_sf)
        tiles_coords.append(ds_out['tntrlcsaf'].coords)
    #concatenate all tiles/files and transpose the matrix
    input_array_ori  = np.concatenate( input_array_list,axis=1).T
    output_array_ori = np.concatenate(output_array_list,axis=1).T
    print('Done.')
    if return_coords:
        return input_array_ori, output_array_ori, tiles_coords
    else:
        return input_array_ori, output_array_ori
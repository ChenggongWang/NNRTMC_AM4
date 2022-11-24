import xarray as xr
import numpy as np 
# def get_AM4_data_sw(out_filelist, inp_filelist, condition='csaf', month_sel = None, day_sel = None):
#     # sample data by month and day
#     if month_sel == None:
#         month_sel = [1,2,3,4,5,6,7,8,9,10,11,12] 
#     if day_sel == None: 
#         day_sel = [1,11,21] 
#     print(f"Data selection:\nMonth: {month_sel} \nDay: {day_sel}")
    
    
#     input_array_list = []
#     output_array_list = []

#     id_list_drop = []
#     input_array_list_drop = []
#     # at least 6 tiles
#     for tile_i in range(len(out_filelist)): 
#         # read data from files  
#         ds = xr.open_dataset(inp_filelist[tile_i])
#         time_sel = ds.time.dt.month.isin(month_sel)&ds.time.dt.day.isin(day_sel)
#         ds_inp = ds.isel(time=time_sel)
#         ds = xr.open_dataset(out_filelist[tile_i])
#         ds_out = ds.isel(time=time_sel)
#         var_ps = ds_inp['level_pressure'].isel(phalf=-1) # this will be included in input by default
#         inp_var_name_csaf   = ['level_temperature'  ,'surface_temperature', 'water_vapor' ,'ozone',
#                                'cosine_zenith',
#                                'visible_direct_albedo','visible_diffuse_albedo',
#                                'infrared_direct_albedo','infrared_diffuse_albedo',
#                               ]
        
#         inp_var_name_cloud  = ['stratiform_droplet_number' ,'stratiform_cloud_fraction' ,
#                                'stratiform_liquid_content' ,'stratiform_ice_content'    ,
#                                'shallow_droplet_number'    ,'shallow_cloud_fraction'    ,
#                                'shallow_liquid_content'    ,'shallow_ice_content'       ,
#                                'strat_size_drop'           ,'shallow_size_drop'          ]
#         # inp_var_name_aersol   = ['stratiform_droplet_number' ,'stratiform_cloud_fraction' ,
#         #                        'stratiform_liquid_content' ,'stratiform_ice_content'    ,
#         #                        'shallow_droplet_number'    ,'shallow_cloud_fraction'    ,
#         #                        'shallow_liquid_content'    ,'shallow_ice_content'       ,
#         #                        'strat_size_drop'           ,'shallow_size_drop'          ]
#         # ! rsdt will be in input list
#         out_var_name_csaf   = [ 'rsutcsaf','rsdscsaf', 'rsuscsaf' ,'tntrscsaf' ]
#         out_var_name_af     = [ 'rsutaf'  ,'rsdsaf'  , 'rsusaf'   ,'tntrsaf' ]
#         out_var_name_cs     = [ 'rsutcs'  ,'rsdscs'  , 'rsuscs'   ,'tntrscs' ]
#         if condition == 'csaf':
#             inp_var_name = inp_var_name_csaf
#             out_var_name = out_var_name_csaf
#         elif condition == 'af':
#             inp_var_name = inp_var_name_csaf + inp_var_name_cloud
#             out_var_name = out_var_name_af
#         else: raise Exception(f"condition {condition} is not configured")
        
#         input_sf =[[ var_ps.stack(txy=("time","grid_xt", "grid_yt")).values],]
#         for _var in inp_var_name:  # for all vars
#             tmp = ds_inp[_var].stack(txy=("time","grid_xt", "grid_yt")).fillna(0).values
#             if len(tmp.shape)==1:
#                 input_sf.append(tmp[None,:]) # addtional dim
#             else:
#                 input_sf.append(tmp)
#         # ! add rsdt to input list
#         tmp = ds_out['rsdt'].stack(txy=("time","grid_xt", "grid_yt")).fillna(0).values
#         input_sf.append(tmp[None,:]) # addtional dim
#         # concatenate all varibile into one big matrix
#         input_sf = np.concatenate(input_sf)
        
#         output_sf = []
#         for _var in out_var_name:  # for all vars
#             tmp = ds_out[_var].stack(txy=("time","grid_xt", "grid_yt")).fillna(0).values
#             if len(tmp.shape)==1:
#                 output_sf.append(tmp[None,:]) # addtional dim
#             else:
#                 output_sf.append(tmp)
#         # concatenate all varibile into one big matrix
#         output_sf = np.concatenate(output_sf) 
#         # get all tiles/files from different files
#         input_array_list.append(input_sf)
#         output_array_list.append(output_sf)
#     #concatenate all tiles/files and transpose the matrix
#     input_array_list  = np.concatenate( input_array_list,axis=1).T
#     output_array_list = np.concatenate(output_array_list,axis=1).T
#     return input_array_list, output_array_list

def get_AM4_data_sw(out_filelist, inp_filelist, condition='csaf', month_sel = None, day_sel = None, return_coords = False):
    # sample data by month and day
    if month_sel == None:
        month_sel = [1,2,3,4,5,6,7,8,9,10,11,12] 
    if day_sel == None: 
        day_sel = [1] 
    print(f"Data selection:\n    Month: {month_sel} \n    Day: {day_sel} \nReading data...", end=' ')
    
    input_array_list = []
    output_array_list = []

    id_list_drop = []
    input_array_list_drop = []
    tiles_coords = []
    # at least 6 tiles
    for tile_i in range(len(out_filelist)): 
        print (tile_i, end=' ')
        # read data from files  
        ds = xr.open_dataset(inp_filelist[tile_i])
        time_sel = ds.time.dt.month.isin(month_sel)&ds.time.dt.day.isin(day_sel)
        ds_inp = ds.isel(time=time_sel)
        ds = xr.open_dataset(out_filelist[tile_i])
        ds_out = ds.isel(time=time_sel)
        var_ps = ds_inp['level_pressure'].isel(phalf=-1) # this will be included in input by default
        inp_var_name_csaf   = ['level_temperature'  ,'surface_temperature', 'water_vapor' ,'ozone',
                               'cosine_zenith',
                               'visible_direct_albedo','visible_diffuse_albedo',
                               'infrared_direct_albedo','infrared_diffuse_albedo',
                              ]
        
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
        # ! rsdt will be in input list
        out_var_name_csaf   = [ 'rsutcsaf','rsdscsaf', 'rsuscsaf' ,'tntrscsaf' ]
        out_var_name_af     = [ 'rsutaf'  ,'rsdsaf'  , 'rsusaf'   ,'tntrsaf' ]
        out_var_name_cs     = [ 'rsutcs'  ,'rsdscs'  , 'rsuscs'   ,'tntrscs' ]
        if condition == 'csaf':
            inp_var_name = inp_var_name_csaf
            out_var_name = out_var_name_csaf
        elif condition == 'af':
            inp_var_name = inp_var_name_csaf + inp_var_name_cloud
            out_var_name = out_var_name_af
        else: raise Exception(f"condition {condition} is not configured")
        
        input_sf =[[ var_ps.stack(txy=("time","grid_yt", "grid_xt")).values],]
        for _var in inp_var_name:  # for all vars
            tmp = ds_inp[_var].stack(txy=("time","grid_yt", "grid_xt")).fillna(0).values
            if len(tmp.shape)==1:
                input_sf.append(tmp[None,:]) # addtional dim
            else:
                input_sf.append(tmp)
        # ! add rsdt to input list
        tmp = ds_out['rsdt'].stack(txy=("time","grid_yt", "grid_xt")).fillna(0).values
        input_sf.append(tmp[None,:]) # addtional dim
        # concatenate all varibile into one big matrix
        input_sf = np.concatenate(input_sf)
        
        output_sf = []
        for _var in out_var_name:  # for all vars
            tmp = ds_out[_var].stack(txy=("time","grid_yt", "grid_xt")).fillna(0).values
            if len(tmp.shape)==1:
                output_sf.append(tmp[None,:]) # addtional dim
            else:
                output_sf.append(tmp)
        # concatenate all varibile into one big matrix
        output_sf = np.concatenate(output_sf) 
        # get all tiles/files from different files
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
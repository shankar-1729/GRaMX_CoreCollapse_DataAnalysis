import os
import openpmd_api as io
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import time

os.system("whoami")

movie_flag = False

#//////////////////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////////////////////

def get_iteration_number_list(data_dir):

    basename = None
    for f in os.listdir(data_dir):
        g = re.match(r'(.*)\.it\d+\.bp4?$', f)
        if g:
            basename = g.group(1)
            break
    assert basename is not None, "Could not find any appropriate .bp files in data-dir."
        

    fname = f"{data_dir}/{basename}.it%08T.bp4"
    print("reading:",fname)

    series = io.Series(fname, io.Access.read_only)
    print("openPMD version:", series.openPMD)
    if series.contains_attribute("author"):
        print("Author: ",series.author)

    sys.stdout.flush()
    #print("Available iterations:")
    iteration_number = []
    count = 0
    for index in series.iterations:
        iteration_number.append(index)
        #print("Available iteration[{}] = {}".format(count, index))
        count = count + 1
    return iteration_number
    
#//////////////////////////////////////////////////////////////////////////////////////////////////////////
        



def get_3d_data(group, gf, data_dir, input_iteration, level, verbose):
    
    if verbose:
        print(" ")
        print("group = {}, gf = {}".format(group, gf))

    basename = None
    for f in os.listdir(data_dir):
        g = re.match(r'(.*)\.it\d+\.bp4?$', f)
        if g:
            basename = g.group(1)
            break
    assert basename is not None, "Could not find any appropriate .bp files in data-dir."
        

    fname = f"{data_dir}/{basename}.it%08T.bp4"
    if verbose:
        print("reading:",fname)

    series = io.Series(fname, io.Access.read_only)
    if verbose:
        print("openPMD version:", series.openPMD)
    if series.contains_attribute("author"):
        if verbose:
            print("Author: ",series.author)

    #only load the given available index for now
    selected_iteration =  input_iteration   #TODO: May need to change this   
    itr = series.iterations[selected_iteration]
    if verbose:
        print("itr: {}\n".format(itr))

    time = (itr.time - 71928)/203.0
    if verbose:
        print("time = {}".format(time))
    

    #load mesh "hydrobase_entropy_lev01" (record)
    variable_mesh = itr.meshes["{}_lev{}".format(group, str(level).zfill(2))]
    if verbose:
        print("{}_mesh_lev{}: {}".format(group, str(level).zfill(2), variable_mesh))

    #load components of mesh "hydrobase_entropy_lev01" (record components)
    variable = variable_mesh[gf]
    if verbose:
        print("{}: {}".format(gf, variable)) 

    #print("variable.unit_SI: ", variable.unit_SI)
    #print("variable.shape: {}\n".format(variable.shape))


    #-------------------------------------------------------------------------
    #Extract the actual array bounds
    #Access data via chunk (chunk is equivalent to box)
    level_min_idx_x = 1e100
    level_max_idx_x = -10
    level_min_idx_y = 1e100
    level_max_idx_y = -10
    level_min_idx_z = 1e100
    level_max_idx_z = -10
    for chunk in variable.available_chunks():
        #print("")
        #print(chunk)
        #print("extent: ", chunk.extent, " offset: ", chunk.offset)
        if chunk.offset[2] < level_min_idx_x:
            level_min_idx_x = chunk.offset[2]
        if chunk.offset[2] + chunk.extent[2] > level_max_idx_x:
            level_max_idx_x = chunk.offset[2] + chunk.extent[2]    
        if chunk.offset[1] < level_min_idx_y:
            level_min_idx_y = chunk.offset[1]
        if chunk.offset[1] + chunk.extent[1] > level_max_idx_y:
            level_max_idx_y = chunk.offset[1] + chunk.extent[1] 
        if chunk.offset[0] < level_min_idx_z:
            level_min_idx_z = chunk.offset[0]
        if chunk.offset[0] + chunk.extent[0] > level_max_idx_z:
            level_max_idx_z = chunk.offset[0] + chunk.extent[0] 

    level_max_idx_z = level_max_idx_z - 1
    level_max_idx_y = level_max_idx_y - 1
    level_max_idx_x = level_max_idx_x - 1

    if verbose:
        print("level_min_idx: [z={}, y={}, x={}]".format(level_min_idx_z, level_min_idx_y, level_min_idx_x))
        print("level_max_idx: [z={}, y={}, x={}]".format(level_max_idx_z, level_max_idx_y, level_max_idx_x))

    size_z = level_max_idx_z - level_min_idx_z + 1
    size_y = level_max_idx_y - level_min_idx_y + 1
    size_x = level_max_idx_x - level_min_idx_x + 1
    if verbose:
        print("size: z = {}, y = {}, x = {}".format(size_z, size_y, size_x))

    dz = round(variable_mesh.grid_spacing[0])
    dy = round(variable_mesh.grid_spacing[1])
    dx = round(variable_mesh.grid_spacing[2])
    
    if level == 7 or level == 8:
        dz = variable_mesh.grid_spacing[0]
        dy = variable_mesh.grid_spacing[1]
        dx = variable_mesh.grid_spacing[2]
        
    z0 = -dz*(level_max_idx_z-level_min_idx_z+1)//2    
    y0 = -dy*(level_max_idx_y-level_min_idx_y+1)//2   
    x0 = -dx*(level_max_idx_x-level_min_idx_x+1)//2   
    if gf == "hydrobase_rho":
        print("\ndz = {}, dy = {}, dx = {}, z0 = {}, y0 = {}, x0 = {}".format(dz, dy, dx, z0, y0, x0))

    #To calculate the actual coordinates,we need the following:
    #We need (x0, y0, z0), (dx, dy, dz) and (size_x, size_y, size_z) = variable_3d.shape 
    xv = np.linspace(x0, (size_x-1)*dx+x0, size_x)
    yv = np.linspace(y0, (size_y-1)*dy+y0, size_y)
    yv = np.linspace(z0, (size_z-1)*dz+z0, size_z)
    
    #-------------------------------------------------------------------------

    #After registering a data chunk such as variable_slice_xz for loading, it MUST NOT be modified or deleted until the flush() step is performed! You must not yet access variable_slice_xz!  
    
    #variable_slice_xz = variable[:, level_min_idx_y+y_slice_index, :]   #(z, y, x)
    variable_3d = variable[(level_min_idx_z):(level_max_idx_z+1), (level_min_idx_y):(level_max_idx_y+1), (level_min_idx_x):(level_max_idx_x+1)]   #(z, y, x)
    
    
    #We now flush the registered data chunks and fill them with actual data from the I/O backend.
    series.flush()
    extent = variable_3d.shape
    if gf == "hydrobase_rho":
        print("extent: {}".format(extent))
    
    #print("testing variable..: variable_3d = ", variable_3d[0, 0, 0]) 
    #print(variable_3d[size_z-1,:,0])
                
    # The iteration can be closed in order to help free up resources.
    itr.close()
    
    return selected_iteration, time, x0, y0, z0, dx, dy, dz, variable_3d



#//////////////////////////////////////////////////////////////////////////////////////////////////////////
def get_derived_vars_3d(data_dir, input_iteration, level, verbose):
    selected_iteration, time, x0, y0, z0, dx, dy, dz, rho_3d = \
            get_3d_data("hydrobase_rho", "hydrobase_rho", data_dir, input_iteration, level, verbose)
    selected_iteration, time, x0, y0, z0, dx, dy, dz, press_3d = \
            get_3d_data("hydrobase_press", "hydrobase_press", data_dir, input_iteration, level, verbose)
    selected_iteration, time, x0, y0, z0, dx, dy, dz, eps_3d = \
            get_3d_data("hydrobase_eps", "hydrobase_eps", data_dir, input_iteration, level, verbose)
                    
    selected_iteration, time, x0, y0, z0, dx, dy, dz, velx_3d = \
            get_3d_data("hydrobase_vel", "hydrobase_velx", data_dir, input_iteration, level, verbose)
    selected_iteration, time, x0, y0, z0, dx, dy, dz, vely_3d = \
            get_3d_data("hydrobase_vel", "hydrobase_vely", data_dir, input_iteration, level, verbose)
    selected_iteration, time, x0, y0, z0, dx, dy, dz, velz_3d = \
            get_3d_data("hydrobase_vel", "hydrobase_velz", data_dir, input_iteration, level, verbose)        
    
    selected_iteration, time, x0, y0, z0, dx, dy, dz, bvecx_3d = \
            get_3d_data("hydrobase_bvec", "hydrobase_bvecx", data_dir, input_iteration, level, verbose)
    selected_iteration, time, x0, y0, z0, dx, dy, dz, bvecy_3d = \
            get_3d_data("hydrobase_bvec", "hydrobase_bvecy", data_dir, input_iteration, level, verbose)
    selected_iteration, time, x0, y0, z0, dx, dy, dz, bvecz_3d = \
            get_3d_data("hydrobase_bvec", "hydrobase_bvecz", data_dir, input_iteration, level, verbose)
    
    selected_iteration, time, x0, y0, z0, dx, dy, dz, gxx_3d = \
            get_3d_data("admbase_metric", "admbase_gxx", data_dir, input_iteration, level, verbose)
    selected_iteration, time, x0, y0, z0, dx, dy, dz, gxy_3d = \
            get_3d_data("admbase_metric", "admbase_gxy", data_dir, input_iteration, level, verbose)
    selected_iteration, time, x0, y0, z0, dx, dy, dz, gxz_3d = \
            get_3d_data("admbase_metric", "admbase_gxz", data_dir, input_iteration, level, verbose)
    selected_iteration, time, x0, y0, z0, dx, dy, dz, gyy_3d = \
            get_3d_data("admbase_metric", "admbase_gyy", data_dir, input_iteration, level, verbose)        
    selected_iteration, time, x0, y0, z0, dx, dy, dz, gyz_3d = \
            get_3d_data("admbase_metric", "admbase_gyz", data_dir, input_iteration, level, verbose)        
    selected_iteration, time, x0, y0, z0, dx, dy, dz, gzz_3d = \
            get_3d_data("admbase_metric", "admbase_gzz", data_dir, input_iteration, level, verbose) 
                   
    selected_iteration, time, x0, y0, z0, dx, dy, dz, alp_3d = \
            get_3d_data("admbase_lapse", "admbase_alp", data_dir, input_iteration, level, verbose)                   
    selected_iteration, time, x0, y0, z0, dx, dy, dz, betax_3d = \
            get_3d_data("admbase_shift", "admbase_betax", data_dir, input_iteration, level, verbose)           
    selected_iteration, time, x0, y0, z0, dx, dy, dz, betay_3d = \
            get_3d_data("admbase_shift", "admbase_betay", data_dir, input_iteration, level, verbose)
    selected_iteration, time, x0, y0, z0, dx, dy, dz, betaz_3d = \
            get_3d_data("admbase_shift", "admbase_betaz", data_dir, input_iteration, level, verbose)
    #-------------------------------------------------------------------------------------------------
    #Average the 3d vertex-centered data to cell-centers 
    gxx_3d_ccc = np.zeros((gxx_3d.shape[0]-1, gxx_3d.shape[1]-1, gxx_3d.shape[2]-1))
    gxy_3d_ccc = np.zeros((gxy_3d.shape[0]-1, gxy_3d.shape[1]-1, gxy_3d.shape[2]-1))
    gxz_3d_ccc = np.zeros((gxz_3d.shape[0]-1, gxz_3d.shape[1]-1, gxz_3d.shape[2]-1))
    gyy_3d_ccc = np.zeros((gyy_3d.shape[0]-1, gyy_3d.shape[1]-1, gyy_3d.shape[2]-1))
    gyz_3d_ccc = np.zeros((gyz_3d.shape[0]-1, gyz_3d.shape[1]-1, gyz_3d.shape[2]-1))
    gzz_3d_ccc = np.zeros((gzz_3d.shape[0]-1, gzz_3d.shape[1]-1, gzz_3d.shape[2]-1))
    
    alp_3d_ccc = np.zeros((gxx_3d.shape[0]-1, gxx_3d.shape[1]-1, gxx_3d.shape[2]-1))
    betax_3d_ccc = np.zeros((gxx_3d.shape[0]-1, gxx_3d.shape[1]-1, gxx_3d.shape[2]-1))
    betay_3d_ccc = np.zeros((gxx_3d.shape[0]-1, gxx_3d.shape[1]-1, gxx_3d.shape[2]-1))
    betaz_3d_ccc = np.zeros((gxx_3d.shape[0]-1, gxx_3d.shape[1]-1, gxx_3d.shape[2]-1))
    
    rad_cutoff = np.zeros((gzz_3d.shape[0]-1, gzz_3d.shape[1]-1, gzz_3d.shape[2]-1))
    
    '''
    #This loops takes the majority of processing time
    for i in range(gxx_3d.shape[0]-1):
      for j in range(gxx_3d.shape[1]-1):
        for k in range(gxx_3d.shape[2]-1):
          gxx_3d_ccc[i, j, k] = 0.125*(gxx_3d[i, j, k] + gxx_3d[i, j+1, k] + \
                                       gxx_3d[i, j, k+1] + gxx_3d[i, j+1, k+1] + \
                                       gxx_3d[i+1, j, k] + gxx_3d[i+1, j+1, k] + \
                                       gxx_3d[i+1, j, k+1] + gxx_3d[i+1, j+1, k+1])
          gxy_3d_ccc[i, j, k] = 0.125*(gxy_3d[i, j, k] + gxy_3d[i, j+1, k] + \
                                       gxy_3d[i, j, k+1] + gxy_3d[i, j+1, k+1] + \
                                       gxy_3d[i+1, j, k] + gxy_3d[i+1, j+1, k] + \
                                       gxy_3d[i+1, j, k+1] + gxy_3d[i+1, j+1, k+1])
          gxz_3d_ccc[i, j, k] = 0.125*(gxz_3d[i, j, k] + gxz_3d[i, j+1, k] + \
                                       gxz_3d[i, j, k+1] + gxz_3d[i, j+1, k+1] + \
                                       gxz_3d[i+1, j, k] + gxz_3d[i+1, j+1, k] + \
                                       gxz_3d[i+1, j, k+1] + gxz_3d[i+1, j+1, k+1])
          gyy_3d_ccc[i, j, k] = 0.125*(gyy_3d[i, j, k] + gyy_3d[i, j+1, k] + \
                                       gyy_3d[i, j, k+1] + gyy_3d[i, j+1, k+1] + \
                                       gyy_3d[i+1, j, k] + gyy_3d[i+1, j+1, k] + \
                                       gyy_3d[i+1, j, k+1] + gyy_3d[i+1, j+1, k+1])
          gyz_3d_ccc[i, j, k] = 0.125*(gyz_3d[i, j, k] + gyz_3d[i, j+1, k] + \
                                       gyz_3d[i, j, k+1] + gyz_3d[i, j+1, k+1] + \
                                       gyz_3d[i+1, j, k] + gyz_3d[i+1, j+1, k] + \
                                       gyz_3d[i+1, j, k+1] + gyz_3d[i+1, j+1, k+1])
          gzz_3d_ccc[i, j, k] = 0.125*(gzz_3d[i, j, k] + gzz_3d[i, j+1, k] + \
                                       gzz_3d[i, j, k+1] + gzz_3d[i, j+1, k+1] + \
                                       gzz_3d[i+1, j, k] + gzz_3d[i+1, j+1, k] + \
                                       gzz_3d[i+1, j, k+1] + gzz_3d[i+1, j+1, k+1])
    '''
    
    #-------------------------------------------------------------------------------------
    #calculation of 1D density profiles to calculate radius cutoff for NS mass calculation
    #-------------------------------------------------------------------------------------
    size_x = rho_3d.shape[2]
    size_y = rho_3d.shape[1]
    size_z = rho_3d.shape[0]
    
    xv = np.linspace(x0, (size_x-1)*dx+x0, size_x)
    x_slice = 0.0
    x_slice_index = np.where(abs(xv-x_slice)<=dx)[0][0]
    
    yv = np.linspace(y0, (size_y-1)*dy+y0, size_y)
    y_slice = 0.0
    y_slice_index = np.where(abs(yv-y_slice)<=dy)[0][0]
    
    zv = np.linspace(z0, (size_z-1)*dz+z0, size_z)
    z_slice = 0.0
    z_slice_index = np.where(abs(zv-z_slice)<=dz)[0][0] 
    
    #This gives rho_1d profiles in physical units
    rho_1d_x = rho_3d[z_slice_index, y_slice_index, :]/1.61930347e-18
    rho_1d_y = rho_3d[z_slice_index, :, x_slice_index]/1.61930347e-18
    rho_1d_z = rho_3d[:, y_slice_index, x_slice_index]/1.61930347e-18
    
    #print("rho_1d_x: {}".format(rho_1d_x), flush=True)
    #print("rho_1d_y: {}".format(rho_1d_y), flush=True)
    #print("rho_1d_z: {}".format(rho_1d_z), flush=True)
    
    rho_cutoff = 1e11
    #-----------------------------------------------------
    #calculation of x-radius cutoff
    #-----------------------------------------------------
    idx_x_minus = np.where(rho_1d_x > rho_cutoff)[0][0]
    idx_x_plus = np.where(rho_1d_x > rho_cutoff)[0][-1]
    rad_x_minus = x0 + dx*idx_x_minus
    rad_x_plus = x0 + dx*idx_x_plus
    print( idx_x_minus, abs(rad_x_minus-dx), "{:.4E}".format(rho_1d_x[idx_x_minus-1]), abs(rad_x_minus), "{:.4E}".format(rho_1d_x[idx_x_minus]) )
    print( idx_x_plus, rad_x_plus, "{:.4E}".format(rho_1d_x[idx_x_plus]), rad_x_plus+dx, "{:.4E}".format(rho_1d_x[idx_x_plus+1]) )
    
    #interpolation to determine r where rho=1e11 g/cm^3
    rad_x_minus_final = (rad_x_minus-dx) + dx*(rho_cutoff-rho_1d_x[idx_x_minus-1])/(rho_1d_x[idx_x_minus]-rho_1d_x[idx_x_minus-1])
    rad_x_minus_final = abs(rad_x_minus_final)
    rad_x_plus_final = rad_x_plus + dx*(rho_cutoff-rho_1d_x[idx_x_plus])/(rho_1d_x[idx_x_plus+1]-rho_1d_x[idx_x_plus]) 
    print("rad_x_minus_final = {}, rad_x_plus_final = {}".format(rad_x_minus_final, rad_x_plus_final))
    rad_x = (rad_x_minus_final + rad_x_plus_final)/2.0
    
    
    #-----------------------------------------------------
    #calculation of y-radius cutoff
    #-----------------------------------------------------
    idx_y_minus = np.where(rho_1d_y > rho_cutoff)[0][0]
    idx_y_plus = np.where(rho_1d_y > rho_cutoff)[0][-1]
    rad_y_minus = y0 + dy*idx_y_minus
    rad_y_plus = y0 + dy*idx_y_plus
    print( idx_y_minus, abs(rad_y_minus-dy), "{:.4E}".format(rho_1d_y[idx_y_minus-1]), abs(rad_y_minus), "{:.4E}".format(rho_1d_y[idx_y_minus]) )
    print( idx_y_plus, rad_y_plus, "{:.4E}".format(rho_1d_y[idx_y_plus]), rad_y_plus+dy, "{:.4E}".format(rho_1d_y[idx_y_plus+1]) )
    
    #interpolation to determine r where rho=1e11 g/cm^3
    rad_y_minus_final = (rad_y_minus-dy) + dy*(rho_cutoff-rho_1d_y[idx_y_minus-1])/(rho_1d_y[idx_y_minus]-rho_1d_y[idx_y_minus-1])
    rad_y_minus_final = abs(rad_y_minus_final)
    rad_y_plus_final = rad_y_plus + dy*(rho_cutoff-rho_1d_y[idx_y_plus])/(rho_1d_y[idx_y_plus+1]-rho_1d_y[idx_y_plus]) 
    print("rad_y_minus_final = {}, rad_y_plus_final = {}".format(rad_y_minus_final, rad_y_plus_final))
    rad_y = (rad_y_minus_final + rad_y_plus_final)/2.0
    
    #-----------------------------------------------------
    #calculation of z-radius cutoff
    #-----------------------------------------------------
    idx_z_minus = np.where(rho_1d_z > rho_cutoff)[0][0]
    idx_z_plus = np.where(rho_1d_z > rho_cutoff)[0][-1]
    rad_z_minus = z0 + dz*idx_z_minus
    rad_z_plus = z0 + dz*idx_z_plus
    print( idx_z_minus, abs(rad_z_minus-dz), "{:.4E}".format(rho_1d_z[idx_z_minus-1]), abs(rad_z_minus), "{:.4E}".format(rho_1d_z[idx_z_minus]) )
    print( idx_z_plus, rad_z_plus, "{:.4E}".format(rho_1d_z[idx_z_plus]), rad_z_plus+dz, "{:.4E}".format(rho_1d_z[idx_z_plus+1]) )
    
    #interpolation to determine r where rho=1e11 g/cm^3
    rad_z_minus_final = (rad_z_minus-dz) + dz*(rho_cutoff-rho_1d_z[idx_z_minus-1])/(rho_1d_z[idx_z_minus]-rho_1d_z[idx_z_minus-1])
    rad_z_minus_final = abs(rad_z_minus_final)
    rad_z_plus_final = rad_z_plus + dz*(rho_cutoff-rho_1d_z[idx_z_plus])/(rho_1d_z[idx_z_plus+1]-rho_1d_z[idx_z_plus]) 
    print("rad_z_minus_final = {}, rad_z_plus_final = {}".format(rad_z_minus_final, rad_z_plus_final))
    rad_z = (rad_z_minus_final + rad_z_plus_final)/2.0

    #------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------
    print("rad_x = {}".format(rad_x))
    print("rad_y = {}".format(rad_y))
    print("rad_z = {}".format(rad_z))
    
    mass_within_radius_M = (rad_x+rad_y+rad_z)/3.0
    mass_within_radius_km = mass_within_radius_M*1.477    
    
    #mass_within_radius_km = 50.0
    #mass_within_radius_M = mass_within_radius_km/1.477
    print("Calculating total mass within radius = {} km = {} M".format(mass_within_radius_km, mass_within_radius_M))
    
    #Don't average for now (until I figure out parallelization)
    print("Started metric calculation...", flush=True)
    for i in range(gxx_3d.shape[0]-1): #z
      for j in range(gxx_3d.shape[1]-1): #y
        for k in range(gxx_3d.shape[2]-1): #x
          xx = k*dx+x0 #TODO TODO: Recheck the coordinate calculation again
          yy = j*dy+y0
          zz = i*dz+z0
          rad = np.sqrt(xx*xx + yy*yy + zz*zz)
          #TODO: Radius within which we calculate total mass
          if rad <= mass_within_radius_M: 
            rad_cutoff[i, j, k] = 1.0 
          
          gxx_3d_ccc[i, j, k] = gxx_3d[i, j, k]                                      
          gxy_3d_ccc[i, j, k] = gxy_3d[i, j, k]                                      
          gxz_3d_ccc[i, j, k] = gxz_3d[i, j, k]                                     
          gyy_3d_ccc[i, j, k] = gyy_3d[i, j, k] 
          gyz_3d_ccc[i, j, k] = gyz_3d[i, j, k] 
          gzz_3d_ccc[i, j, k] = gzz_3d[i, j, k]
          alp_3d_ccc[i, j, k] = alp_3d[i, j, k]
          betax_3d_ccc[i, j, k] = betax_3d[i, j, k]
          betay_3d_ccc[i, j, k] = betay_3d[i, j, k]
          betaz_3d_ccc[i, j, k] = betaz_3d[i, j, k] 
    
          
    #-------------------------------------------------------------------------------------------------
    #Calculate variables
    velxlow = gxx_3d_ccc * velx_3d + gxy_3d_ccc * vely_3d + gxz_3d_ccc * velz_3d;
    velylow = gxy_3d_ccc * velx_3d + gyy_3d_ccc * vely_3d + gyz_3d_ccc * velz_3d;
    velzlow = gxz_3d_ccc * velx_3d + gyz_3d_ccc * vely_3d + gzz_3d_ccc * velz_3d;
     
    v2 = velxlow * velx_3d + velylow * vely_3d + velzlow * velz_3d;
    w = 1.0 / np.sqrt(1.0 - v2);
    
    sdetg = -gxz_3d_ccc * gxz_3d_ccc * gyy_3d_ccc + \
            2.0 * gxy_3d_ccc * gxz_3d_ccc * gyz_3d_ccc - \
            gxx_3d_ccc * gyz_3d_ccc * gyz_3d_ccc - \
            gxy_3d_ccc * gxy_3d_ccc * gzz_3d_ccc + \
            gxx_3d_ccc * gyy_3d_ccc * gzz_3d_ccc 
    
    sdetg = np.sqrt(sdetg)
   
    dens = sdetg*w*rho_3d*rad_cutoff
    
    #Calculation of Bernoulli criterion
    bvecxlow = gxx_3d_ccc * bvecx_3d + gxy_3d_ccc * bvecy_3d + gxz_3d_ccc * bvecz_3d;
    bvecylow = gxy_3d_ccc * bvecx_3d + gyy_3d_ccc * bvecy_3d + gyz_3d_ccc * bvecz_3d;
    bveczlow = gxz_3d_ccc * bvecx_3d + gyz_3d_ccc * bvecy_3d + gzz_3d_ccc * bvecz_3d; 
      
    Bdotv = velxlow * bvecx_3d + velylow * bvecy_3d + velzlow * bvecz_3d;
    b2 = (bvecx_3d * bvecxlow + bvecy_3d * bvecylow + bvecz_3d * bveczlow) / (w*w) + Bdotv*Bdotv;
    
    ut = w*(velxlow*betax_3d_ccc + velylow*betay_3d_ccc + velzlow*betaz_3d_ccc - alp_3d_ccc) 
    bernoulli = - (1.0 + eps_3d + (press_3d + b2) / rho_3d) * ut   #bernoulli = -h*ut
    
    unbound_flag = np.zeros((gzz_3d.shape[0]-1, gzz_3d.shape[1]-1, gzz_3d.shape[2]-1))
       
    print("Started unbound calculation...", flush=True)
    #For unbound material: h*ut < -1 => -h*ut>1 => bernoulli>1
    for i in range(gxx_3d.shape[0]-1): #z
      for j in range(gxx_3d.shape[1]-1): #y
        for k in range(gxx_3d.shape[2]-1): #x
            if bernoulli[i, j, k] > 1:
                unbound_flag[i, j, k] = 1.0
                
    unbound_dens = sdetg*w*rho_3d*unbound_flag
        
    return selected_iteration, time, x0, y0, z0, dx, dy, dz, w, sdetg, dens, unbound_dens
#//////////////////////////////////////////////////////////////////////////////////////////////////////////        

from timeit import default_timer as timer        


#------------------------------------------------------------------
'''
Available data range:
CCSN_12000km: output-0000 to output-0106
Ref6_40: output-0000 to output-0055 (except output-0043)
'''
#------------------------------------------------------------------               


#for output_number in range(106, 107):
for output_number in range(30, 31):   
    sim_name = "CCSN_12000km"   
    #sim_name = "Ref6_40"
    
    parfile_name = "CCSN_12000km"
    verbose = False
    
    #data_dir = "/gpfs/alpine/ast154/scratch/sshanka/simulations/{}/output-{}/{}/".format(sim_name, str(output_number).zfill(4), parfile_name)
    data_dir = "/lustre/orion/ast154/scratch/sshanka/simulations/{}/output-{}/{}/".format(sim_name, str(output_number).zfill(4), parfile_name)
    
    if output_number == 43 and sim_name == "Ref6_40":
        continue
    
    iteration_numbers = get_iteration_number_list(data_dir)
    print("Running output-{}:: found iterations {}".format(str(output_number).zfill(4), iteration_numbers), flush=True)   
    
    #TODO: Try to use multithreading
    for input_iteration in iteration_numbers:
        
        if input_iteration != iteration_numbers[0]:
            continue         
        
        #----------------------------------------------------------------------------
        start = timer()
        level = 6  #which refinement level to load
        selected_iteration, time, x0, y0, z0, dx, dy, dz, w, sdetg, dens, unbound_dens = \
            get_derived_vars_3d(data_dir, input_iteration, level, verbose)
            
        print("At level {}: total mass = {}".format(level, dx*dy*dz*np.sum(dens)));
        print("At level {}: ejecta mass = {}".format(level, dx*dy*dz*np.sum(unbound_dens)));
        
        end = timer()
        time_elapsed = end - start
        print("Time elapsed = {} seconds = {} minutes".format(round(time_elapsed), round(time_elapsed/60.0)))
        
        print("Finished iteration {}, t = {}\n".format(input_iteration, time))
        sys.stdout.flush()
        #----------------------------------------------------------------------------
        
        
        

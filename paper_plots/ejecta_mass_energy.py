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
    
    #Don't average for now (until I figure out parallelization)
    '''
    for i in range(gxx_3d.shape[0]-1): #z
      for j in range(gxx_3d.shape[1]-1): #y
        for k in range(gxx_3d.shape[2]-1): #x
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
    '''          
    
    print("Started metric calculation...", flush=True)
    #strip the last element of metric to make its shape same as hydro variables
    #We use vectorized processing rather than using for loop: speedup is tremendous
    gxx_3d_ccc = gxx_3d[:-1, :-1, :-1].copy()                                  
    gxy_3d_ccc = gxy_3d[:-1, :-1, :-1].copy()                                    
    gxz_3d_ccc = gxz_3d[:-1, :-1, :-1].copy()                                 
    gyy_3d_ccc = gyy_3d[:-1, :-1, :-1].copy() 
    gyz_3d_ccc = gyz_3d[:-1, :-1, :-1].copy() 
    gzz_3d_ccc = gzz_3d[:-1, :-1, :-1].copy()
    alp_3d_ccc = alp_3d[:-1, :-1, :-1].copy()
    betax_3d_ccc = betax_3d[:-1, :-1, :-1].copy()
    betay_3d_ccc = betay_3d[:-1, :-1, :-1].copy()
    betaz_3d_ccc = betaz_3d[:-1, :-1, :-1].copy()
    
    del gxx_3d, gxy_3d, gxz_3d, gyy_3d, gyz_3d, gzz_3d, alp_3d, betax_3d, betay_3d, betaz_3d      
    #-------------------------------------------------------------------------------------------------
    
    print("Started Bernoulli calculation...", flush=True)
    #Calculate variables
    velxlow = gxx_3d_ccc * velx_3d + gxy_3d_ccc * vely_3d + gxz_3d_ccc * velz_3d;
    velylow = gxy_3d_ccc * velx_3d + gyy_3d_ccc * vely_3d + gyz_3d_ccc * velz_3d;
    velzlow = gxz_3d_ccc * velx_3d + gyz_3d_ccc * vely_3d + gzz_3d_ccc * velz_3d;
    
    bvecxlow = gxx_3d_ccc * bvecx_3d + gxy_3d_ccc * bvecy_3d + gxz_3d_ccc * bvecz_3d;
    bvecylow = gxy_3d_ccc * bvecx_3d + gyy_3d_ccc * bvecy_3d + gyz_3d_ccc * bvecz_3d;
    bveczlow = gxz_3d_ccc * bvecx_3d + gyz_3d_ccc * bvecy_3d + gzz_3d_ccc * bvecz_3d;
     
    v2 = velxlow * velx_3d + velylow * vely_3d + velzlow * velz_3d;
    w = 1.0 / np.sqrt(1.0 - v2);
    
    sdetg = -gxz_3d_ccc * gxz_3d_ccc * gyy_3d_ccc + \
            2.0 * gxy_3d_ccc * gxz_3d_ccc * gyz_3d_ccc - \
            gxx_3d_ccc * gyz_3d_ccc * gyz_3d_ccc - \
            gxy_3d_ccc * gxy_3d_ccc * gzz_3d_ccc + \
            gxx_3d_ccc * gyy_3d_ccc * gzz_3d_ccc 
    
    sdetg = np.sqrt(sdetg)
    
    del gxx_3d_ccc, gxy_3d_ccc, gxz_3d_ccc, gyy_3d_ccc, gyz_3d_ccc, gzz_3d_ccc
    del velx_3d, vely_3d, velz_3d 
        
    #Calculation of Bernoulli criterion    
    Bdotv = velxlow * bvecx_3d + velylow * bvecy_3d + velzlow * bvecz_3d;
    b2 = (bvecx_3d * bvecxlow + bvecy_3d * bvecylow + bvecz_3d * bveczlow) / (w*w) + Bdotv*Bdotv;
    
    ut = w*(velxlow*betax_3d_ccc + velylow*betay_3d_ccc + velzlow*betaz_3d_ccc - alp_3d_ccc) 
    enthalpy_h = 1.0 + eps_3d + (press_3d + b2) / rho_3d
    bernoulli = - enthalpy_h* ut   #bernoulli = -h*ut
    
    print("Started unbound calculation...", flush=True)
    unbound_flag = np.zeros((rho_3d.shape[0], rho_3d.shape[1], rho_3d.shape[2]))
    #We use vectorized processing rather than using 'for' loop: speedup is tremendous
    unbound_flag[bernoulli > 1.0] = 1.0
    
    '''
    unbound_flag = np.zeros((rho_3d.shape[0], rho_3d.shape[1], rho_3d.shape[2]))
    #For unbound material: h*ut < -1 => -h*ut>1 => bernoulli>1
    for i in range(rho_3d.shape[0]): #z
      for j in range(rho_3d.shape[1]): #y
        for k in range(rho_3d.shape[2]): #x
            if bernoulli[i, j, k] > 1:                      
                unbound_flag[i, j, k] = 1.0                  
    '''
    
    del bernoulli, ut, bvecxlow, bvecylow, bveczlow, velxlow, velylow, velzlow, betax_3d_ccc, betay_3d_ccc, betaz_3d_ccc
    
    #mass = (rho*wlorentz)*(sdetg*dx*dy*dz)
    #KE = 0.5*(rho*wlorentz)*(v^2)*(sdetg*dx*dy*dz)
    #IE = eps*(rho*wlorentz)*(sdetg*dx*dy*dz)
    #ME = (0.5*bcom_sq)*(sdetg*dx*dy*dz)            
    unbound_dens = sdetg*w*rho_3d*unbound_flag
    #unbound_KE = 0.5*sdetg*w*rho_3d*v2*unbound_flag #TODO TODO
    unbound_KE = sdetg*(w-1)*rho_3d*unbound_flag 
    unbound_IE = abs(eps_3d)*sdetg*w*rho_3d*unbound_flag  
    unbound_ME = 0.5*b2*sdetg*unbound_flag
    
    #TODO: Look at eq 24 of: https://arxiv.org/pdf/2311.04989.pdf
    
    #-------------------------------------------------------------------------------
    #calculate energy using kuroda2023 method
    kuroda_unbound_flag = np.zeros((rho_3d.shape[0], rho_3d.shape[1], rho_3d.shape[2]))
    
    #Muller2012: https://iopscience.iop.org/article/10.1088/0004-637X/756/1/84/pdf (see eq 2)
    #e_bind = alp_3d_ccc*(rho_3d*enthalpy_h*w*w - (press_3d + 0.5*b2)) - rho_3d*w 
    
    #Kuroda2023: https://arxiv.org/pdf/2309.05161.pdf (see eq 4 and 5)
    #definition of tau is not clear, so I will use same definition as cons.tau
    tau = rho_3d*enthalpy_h*w*w - (press_3d + 0.5*b2) - w*Bdotv*w*Bdotv 
    e_bind = alp_3d_ccc*tau - rho_3d*w 
    
    kuroda_unbound_flag[e_bind > 0.0] = 1.0
    kuroda_explosion_energy = e_bind*sdetg*kuroda_unbound_flag
    #-------------------------------------------------------------------------------
    
    return selected_iteration, time, x0, y0, z0, dx, dy, dz, w, sdetg, unbound_dens, unbound_KE, unbound_IE, unbound_ME, kuroda_explosion_energy
#//////////////////////////////////////////////////////////////////////////////////////////////////////////        

from timeit import default_timer as timer        


#------------------------------------------------------------------
'''
Available data range:
CCSN_12000km: output-0000 to output-0106
Ref6_40: output-0000 to output-0055 (except output-0043)
'''
#------------------------------------------------------------------               

#TODO
sim_name = "CCSN_12000km"   
#sim_name = "Ref6_40"
#sim_name = "Ref6_40_AST191"

import os 

#buffering=1 #means flush output after every line
#TODO
#f = open("ejecta_mass_energy/ejecta_mass_energy_abs_eps_{}.txt".format(sim_name), "w", buffering=1) 
f = open("ejecta_mass_energy/test.txt".format(sim_name), "w", buffering=1) 
f.write("#o/p  it       t_pb[ms]       level        ejecta_mass[M]      ejecta_KE[erg]     ejecta_IE[erg]     ejecta_ME[erg]     ejecta_TE[erg]   exp_energy_kuroda[erg]\n")

#TODO
for output_number in range(90, 91):      
    parfile_name = "CCSN_12000km"
    verbose = False
    
    #TODO TODO
    #data_dir = "/lustre/orion/ast191/scratch/sshanka/simulations/Ref6_40/output-{}/{}/".format(str(output_number).zfill(4), parfile_name)
    data_dir = "/lustre/orion/ast154/scratch/sshanka/simulations/{}/output-{}/{}/".format(sim_name, str(output_number).zfill(4), parfile_name)
    
    #--------------------------------------------------------------------------------
    print("-------------------------------------------------------------------------")    
    #Does directory exist
    directory_exists = os.path.exists(data_dir)
    if(not directory_exists):
        print("Directory {} doesn't exist".format(data_dir))    
        continue
    
    #Does file exist
    file_exists = False
    for fname in os.listdir(data_dir):
        if fname.endswith('.bp4'):
            print("Found file output-{}/{}".format(str(output_number).zfill(4), fname))
            file_exists = True
    print("output-{}: file exists = {}".format(str(output_number).zfill(4), file_exists))
    #--------------------------------------------------------------------------------
    
    if(not file_exists):
        continue
    
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
        
        level = -1
        if sim_name == "Ref6_40" or sim_name == "Ref6_40_AST191":
            level = 5
        if sim_name == "CCSN_12000km":
            level = 6      
        
        selected_iteration, time, x0, y0, z0, dx, dy, dz, w, sdetg, \
        unbound_dens, unbound_KE, unbound_IE, unbound_ME, kuroda_explosion_energy = \
                 get_derived_vars_3d(data_dir, input_iteration, level, verbose)
        
        ejecta_mass = dx*dy*dz*np.sum(unbound_dens)  
        ejecta_KE = dx*dy*dz*np.sum(unbound_KE)*1.7877e54 
        ejecta_IE = dx*dy*dz*np.sum(unbound_IE)*1.7877e54  
        ejecta_ME = dx*dy*dz*np.sum(unbound_ME)*1.7877e54 
        ejecta_TE =  ejecta_KE + ejecta_IE + ejecta_ME 
        kuroda_total_energy = dx*dy*dz*np.sum(kuroda_explosion_energy)*1.7877e54 
        
        print("At level {}: ejecta mass = {} M_sun".format(level, ejecta_mass));
        print("At level {}: ejecta Kinetic Energy = {} erg".format(level, ejecta_KE));
        print("At level {}: ejecta Internal Energy = {} erg".format(level, ejecta_IE));
        print("At level {}: ejecta Magnetic Energy = {} erg".format(level, ejecta_ME));
        print("At level {}: ejecta Total Energy = {} erg".format(level, ejecta_TE));
        print("At level {}: kuroda explosion Energy = {} erg".format(level, kuroda_total_energy));
        
        end = timer()
        time_elapsed = end - start
        print("Time elapsed = {} seconds = {} minutes".format(round(time_elapsed), round(time_elapsed/60.0)))
        
        print("Finished iteration {}, t_pb = {} ms".format(input_iteration, time))
        print("---------------------------------------------------------------\n")
        sys.stdout.flush()
        
        f.write("{}  {}  {}  {}     {}     {}     {}     {}     {}     {}\n".format(output_number, input_iteration, time, level, ejecta_mass, ejecta_KE, ejecta_IE, ejecta_ME, ejecta_TE, kuroda_total_energy))
        #----------------------------------------------------------------------------

f.close()        
        
        

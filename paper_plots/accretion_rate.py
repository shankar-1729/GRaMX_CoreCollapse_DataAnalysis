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
                    
    selected_iteration, time, x0, y0, z0, dx, dy, dz, velx_3d = \
            get_3d_data("hydrobase_vel", "hydrobase_velx", data_dir, input_iteration, level, verbose)
    selected_iteration, time, x0, y0, z0, dx, dy, dz, vely_3d = \
            get_3d_data("hydrobase_vel", "hydrobase_vely", data_dir, input_iteration, level, verbose)
    selected_iteration, time, x0, y0, z0, dx, dy, dz, velz_3d = \
            get_3d_data("hydrobase_vel", "hydrobase_velz", data_dir, input_iteration, level, verbose)        
    
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
                   
    #-------------------------------------------------------------------------------------------------
    #Average the 3d vertex-centered data to cell-centers 
    gxx_3d_ccc = np.zeros((gxx_3d.shape[0]-1, gxx_3d.shape[1]-1, gxx_3d.shape[2]-1))
    gxy_3d_ccc = np.zeros((gxy_3d.shape[0]-1, gxy_3d.shape[1]-1, gxy_3d.shape[2]-1))
    gxz_3d_ccc = np.zeros((gxz_3d.shape[0]-1, gxz_3d.shape[1]-1, gxz_3d.shape[2]-1))
    gyy_3d_ccc = np.zeros((gyy_3d.shape[0]-1, gyy_3d.shape[1]-1, gyy_3d.shape[2]-1))
    gyz_3d_ccc = np.zeros((gyz_3d.shape[0]-1, gyz_3d.shape[1]-1, gyz_3d.shape[2]-1))
    gzz_3d_ccc = np.zeros((gzz_3d.shape[0]-1, gzz_3d.shape[1]-1, gzz_3d.shape[2]-1))
    
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
    
    rad_cutoff_50km = np.zeros((gzz_3d.shape[0]-1, gzz_3d.shape[1]-1, gzz_3d.shape[2]-1)) 
    rad_cutoff_70km = np.zeros((gzz_3d.shape[0]-1, gzz_3d.shape[1]-1, gzz_3d.shape[2]-1))  
    rad_cutoff_90km = np.zeros((gzz_3d.shape[0]-1, gzz_3d.shape[1]-1, gzz_3d.shape[2]-1))      
   
    mass_within_50km_M = 50.0/1.477
    mass_within_70km_M = 70.0/1.477
    mass_within_90km_M = 90.0/1.477
    #print("Calculating total mass within radius = 50.0 km = {} M".format(mass_within_50km_M))
    
    #Don't average for now (until I figure out parallelization)
    print("Started metric calculation...", flush=True)
    for i in range(gxx_3d.shape[0]-1): #z
      for j in range(gxx_3d.shape[1]-1): #y
        for k in range(gxx_3d.shape[2]-1): #x
          xx = k*dx+x0 #TODO TODO: Recheck the coordinate calculation again
          yy = j*dy+y0
          zz = i*dz+z0
          rad = np.sqrt(xx*xx + yy*yy + zz*zz)
          if rad <= mass_within_50km_M: 
            rad_cutoff_50km[i, j, k] = 1.0
          if rad <= mass_within_70km_M: 
            rad_cutoff_70km[i, j, k] = 1.0 
          if rad <= mass_within_90km_M: 
            rad_cutoff_90km[i, j, k] = 1.0  
          
          gxx_3d_ccc[i, j, k] = gxx_3d[i, j, k]                                      
          gxy_3d_ccc[i, j, k] = gxy_3d[i, j, k]                                      
          gxz_3d_ccc[i, j, k] = gxz_3d[i, j, k]                                     
          gyy_3d_ccc[i, j, k] = gyy_3d[i, j, k] 
          gyz_3d_ccc[i, j, k] = gyz_3d[i, j, k] 
          gzz_3d_ccc[i, j, k] = gzz_3d[i, j, k]
              
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
   
    dens_50km = sdetg*w*rho_3d*rad_cutoff_50km
    dens_70km = sdetg*w*rho_3d*rad_cutoff_70km
    dens_90km = sdetg*w*rho_3d*rad_cutoff_90km
    
    return selected_iteration, time, x0, y0, z0, dx, dy, dz, w, sdetg, dens_50km, dens_70km, dens_90km
#//////////////////////////////////////////////////////////////////////////////////////////////////////////        

from timeit import default_timer as timer        


#------------------------------------------------------------------
'''
Available data range:
CCSN_12000km: output-0000 to output-0106
Ref6_40: output-0000 to output-0055 (except output-0043)
'''
#------------------------------------------------------------------               

sim_name = "CCSN_12000km"   #TODO
#sim_name = "Ref6_40"       #TODO

#"buffering=1" means flush output after every line
f = open("accretion_rate/accretion_rate_{}.txt".format(sim_name), "w", buffering=1) 
#f = open("accretion_rate/test.txt".format(sim_name), "w", buffering=1) 
f.write("#o/p  it       t_pb[ms]       level        mass_50km[M]        mass_70km[M]    mass_90km[M]\n") 


for output_number in range(0, 107): #TODO     
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
        
        level = -1
        if sim_name == "Ref6_40":
            level = 5
        if sim_name == "CCSN_12000km":
            level = 6    
        
        selected_iteration, time, x0, y0, z0, dx, dy, dz, w, sdetg, dens_50km, dens_70km, dens_90km = \
            get_derived_vars_3d(data_dir, input_iteration, level, verbose)
        
        mass_50km = dx*dy*dz*np.sum(dens_50km) 
        mass_70km = dx*dy*dz*np.sum(dens_70km)  
        mass_90km = dx*dy*dz*np.sum(dens_90km)   
        
        print("At level {}: mass enclosed within 50 km = {} M_sun".format(level, mass_50km));
        print("At level {}: mass enclosed within 70 km = {} M_sun".format(level, mass_70km));
        print("At level {}: mass enclosed within 90 km = {} M_sun".format(level, mass_90km));
        
        end = timer()
        time_elapsed = end - start
        print("Time elapsed = {} seconds = {} minutes".format(round(time_elapsed), round(time_elapsed/60.0)))
        
        print("Finished iteration {}, t = {}".format(input_iteration, time))
        print("---------------------------------------------------------------\n")
        sys.stdout.flush()
        
        f.write("{}  {}  {}  {}     {}    {}    {}\n".format(output_number, input_iteration, time, level, mass_50km, mass_70km, mass_90km))  
        #----------------------------------------------------------------------------

f.close()        
        
        

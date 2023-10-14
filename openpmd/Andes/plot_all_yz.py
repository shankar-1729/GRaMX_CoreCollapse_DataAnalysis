import os
import openpmd_api as io
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors

os.system("whoami")

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

    #print("Available iterations:")
    iteration_number = []
    count = 0
    for index in series.iterations:
        iteration_number.append(index)
        #print("Available iteration[{}] = {}".format(count, index))
        count = count + 1
    return iteration_number
    
#//////////////////////////////////////////////////////////////////////////////////////////////////////////
        
def get_data(group, gf, x_slice, data_dir, input_iteration):
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
    print("reading:",fname)

    series = io.Series(fname, io.Access.read_only)
    print("openPMD version:", series.openPMD)
    if series.contains_attribute("author"):
        print("Author: ",series.author)

    #only load the first available index for now
    selected_iteration =  input_iteration   #TODO: May need to change this   
    itr = series.iterations[selected_iteration]
    print("itr: {}\n".format(itr))

    time = (itr.time - 71928)/203.0
    print("time = {}".format(time))
    
    #which refinement level to load
    level = 5

    #load mesh "hydrobase_entropy_lev01" (record)
    variable_mesh = itr.meshes["{}_lev{}".format(group, str(level).zfill(2))]
    print("{}_mesh_lev{}: {}".format(group, str(level).zfill(2), variable_mesh))

    #load components of mesh "hydrobase_entropy_lev01" (record components)
    variable = variable_mesh[gf]
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

    print("level_min_idx: [z={}, y={}, x={}]".format(level_min_idx_z, level_min_idx_y, level_min_idx_x))
    print("level_max_idx: [z={}, y={}, x={}]".format(level_max_idx_z, level_max_idx_y, level_max_idx_x))

    size_z = level_max_idx_z - level_min_idx_z + 1
    size_y = level_max_idx_y - level_min_idx_y + 1
    size_x = level_max_idx_x - level_min_idx_x + 1
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
    print("dz = {}, dy = {}, dx = {}, z0 = {}, y0 = {}, x0 = {}".format(dz, dy, dx, z0, y0, x0))

    xv = np.linspace(x0, (size_x-1)*dx+x0, size_x)
    x_slice_index = np.where(abs(xv-x_slice)<=dx)[0][0]
    print("x_slice = {}, x_slice_index = {}".format(x_slice, x_slice_index))
    #-------------------------------------------------------------------------

    #After registering a data chunk such as variable_slice_yz for loading, it MUST NOT be modified or deleted until the flush() step is performed! You must not yet access variable_slice_yz!  
    variable_slice_yz = variable[:, :, level_min_idx_x+x_slice_index]   #(z, y, x)
    #variable_slice_xz = variable[:, (level_min_idx_y+level_max_idx_y)//2, :]
    #variable_slice_xy = variable[(level_min_idx_z+level_max_idx_z)//2, :, :]

    #We now flush the registered data chunks and fill them with actual data from the I/O backend.
    series.flush()

    #We can now work with the newly loaded data 
    extent = variable_slice_yz.shape
    #print("extent: {}\n".format(extent))

    #/////////////////////////////////////////////////////////////////

    #Let's use variable_slice_yz
    variable_2d_yz = np.zeros((size_z, size_y), dtype=np.double) #ignoring x since we took slice along x
    for chunk in variable.available_chunks():
        idx0_z = chunk.offset[0] - level_min_idx_z
        idx0_y = chunk.offset[1] - level_min_idx_y
        idx0_x = chunk.offset[2] - level_min_idx_x
        
        idx1_z = idx0_z + chunk.extent[0] 
        idx1_y = idx0_y + chunk.extent[1] 
        idx1_x = idx0_x + chunk.extent[2] 
        
        #print("idx0 = [{}, {}, {}], idx1 = [{}, {}, {}]".format(idx0_z, idx0_y, idx0_x, idx1_z-1, idx1_y-1, idx1_x-1))
        
        for k in range(chunk.extent[0]):
            for j in range(chunk.extent[1]):
                variable_2d_yz[idx0_z + k, idx0_y + j] = variable_slice_yz[chunk.offset[0] + k, chunk.offset[1] + j]

    
    # The iteration can be closed in order to help free up resources.
    itr.close()
    
    print("{}: min = {}, max = {}".format(gf, np.min(variable_2d_yz), np.max(variable_2d_yz)))
    
    #print(variable_2d_yz[32, 32])
    return selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, level  

#//////////////////////////////////////////////////////////////////////////////////////////////////////////

def get_plasma_beta(x_slice, data_dir, input_iteration):
    #Plot the plasma-beta
    selected_iteration, time, y0, z0, dy, dz, press_2d_yz = get_data("hydrobase_press", "hydrobase_press", x_slice, data_dir, input_iteration)

    selected_iteration, time, y0, z0, dy, dz, bvecx_2d_yz = get_data("hydrobase_bvec", "hydrobase_bvecx", x_slice, data_dir, input_iteration)
    selected_iteration, time, y0, z0, dy, dz, bvecy_2d_yz = get_data("hydrobase_bvec", "hydrobase_bvecy", x_slice, data_dir, input_iteration)
    selected_iteration, time, y0, z0, dy, dz, bvecz_2d_yz = get_data("hydrobase_bvec", "hydrobase_bvecz", x_slice, data_dir, input_iteration)

    selected_iteration, time, y0, z0, dy, dz, velx_2d_yz = get_data("hydrobase_vel", "hydrobase_velx", x_slice, data_dir, input_iteration)
    selected_iteration, time, y0, z0, dy, dz, vely_2d_yz = get_data("hydrobase_vel", "hydrobase_vely", x_slice, data_dir, input_iteration)
    selected_iteration, time, y0, z0, dy, dz, velz_2d_yz = get_data("hydrobase_vel", "hydrobase_velz", x_slice, data_dir, input_iteration)

    selected_iteration, time, y0, z0, dy, dz, gxx_2d_yz = get_data("admbase_metric", "admbase_gxx", x_slice, data_dir, input_iteration)
    selected_iteration, time, y0, z0, dy, dz, gxy_2d_yz = get_data("admbase_metric", "admbase_gxy", x_slice, data_dir, input_iteration)
    selected_iteration, time, y0, z0, dy, dz, gxz_2d_yz = get_data("admbase_metric", "admbase_gxz", x_slice, data_dir, input_iteration)
    selected_iteration, time, y0, z0, dy, dz, gyy_2d_yz = get_data("admbase_metric", "admbase_gyy", x_slice, data_dir, input_iteration)
    selected_iteration, time, y0, z0, dy, dz, gyz_2d_yz = get_data("admbase_metric", "admbase_gyz", x_slice, data_dir, input_iteration)
    selected_iteration, time, y0, z0, dy, dz, gzz_2d_yz = get_data("admbase_metric", "admbase_gzz", x_slice, data_dir, input_iteration)
    
    selected_iteration, time, y0, z0, dy, dz, rho_2d_yz = get_data("hydrobase_rho", "hydrobase_rho", x_slice, data_dir, input_iteration)

    #-------------------------------------------------------------------------------------------------
    #Average the 2d vertex-centered data to cell-centers (this is approx, averaging along x is missing)
    gxx_2d_yz_ccc = np.zeros((gxx_2d_yz.shape[0]-1, gxx_2d_yz.shape[1]-1))
    gxy_2d_yz_ccc = np.zeros((gxy_2d_yz.shape[0]-1, gxy_2d_yz.shape[1]-1))
    gxz_2d_yz_ccc = np.zeros((gxz_2d_yz.shape[0]-1, gxz_2d_yz.shape[1]-1))
    gyy_2d_yz_ccc = np.zeros((gyy_2d_yz.shape[0]-1, gyy_2d_yz.shape[1]-1))
    gyz_2d_yz_ccc = np.zeros((gyz_2d_yz.shape[0]-1, gyz_2d_yz.shape[1]-1))
    gzz_2d_yz_ccc = np.zeros((gzz_2d_yz.shape[0]-1, gzz_2d_yz.shape[1]-1))
    for i in range(gxx_2d_yz.shape[0]-1):
      for j in range(gxx_2d_yz.shape[1]-1):
          gxx_2d_yz_ccc[i, j] = 0.25*(gxx_2d_yz[i, j] + gxx_2d_yz[i+1, j] + gxx_2d_yz[i, j+1] + gxx_2d_yz[i+1, j+1])
          gxy_2d_yz_ccc[i, j] = 0.25*(gxy_2d_yz[i, j] + gxy_2d_yz[i+1, j] + gxy_2d_yz[i, j+1] + gxy_2d_yz[i+1, j+1])
          gxz_2d_yz_ccc[i, j] = 0.25*(gxz_2d_yz[i, j] + gxz_2d_yz[i+1, j] + gxz_2d_yz[i, j+1] + gxz_2d_yz[i+1, j+1])
          gyy_2d_yz_ccc[i, j] = 0.25*(gyy_2d_yz[i, j] + gyy_2d_yz[i+1, j] + gyy_2d_yz[i, j+1] + gyy_2d_yz[i+1, j+1])
          gyz_2d_yz_ccc[i, j] = 0.25*(gyz_2d_yz[i, j] + gyz_2d_yz[i+1, j] + gyz_2d_yz[i, j+1] + gyz_2d_yz[i+1, j+1])
          gzz_2d_yz_ccc[i, j] = 0.25*(gzz_2d_yz[i, j] + gzz_2d_yz[i+1, j] + gzz_2d_yz[i, j+1] + gzz_2d_yz[i+1, j+1])
        
    #-------------------------------------------------------------------------------------------------
    #Calculate plasma-beta
    velxlow = gxx_2d_yz_ccc * velx_2d_yz + gxy_2d_yz_ccc * vely_2d_yz + gxz_2d_yz_ccc * velz_2d_yz;
    velylow = gxy_2d_yz_ccc * velx_2d_yz + gyy_2d_yz_ccc * vely_2d_yz + gyz_2d_yz_ccc * velz_2d_yz;
    velzlow = gxz_2d_yz_ccc * velx_2d_yz + gyz_2d_yz_ccc * vely_2d_yz + gzz_2d_yz_ccc * velz_2d_yz;

    bvecxlow = gxx_2d_yz_ccc * bvecx_2d_yz + gxy_2d_yz_ccc * bvecy_2d_yz + gxz_2d_yz_ccc * bvecz_2d_yz;
    bvecylow = gxy_2d_yz_ccc * bvecx_2d_yz + gyy_2d_yz_ccc * bvecy_2d_yz + gyz_2d_yz_ccc * bvecz_2d_yz;
    bveczlow = gxz_2d_yz_ccc * bvecx_2d_yz + gyz_2d_yz_ccc * bvecy_2d_yz + gzz_2d_yz_ccc * bvecz_2d_yz; 
      
    Bdotv = velxlow * bvecx_2d_yz + velylow * bvecy_2d_yz + velzlow * bvecz_2d_yz;
      
    v2 = velxlow * velx_2d_yz + velylow * vely_2d_yz + velzlow * velz_2d_yz;
    w = 1.0 / np.sqrt(1.0 - v2); 

    b2 = (bvecx_2d_yz * bvecxlow + bvecy_2d_yz * bvecylow + bvecz_2d_yz * bveczlow) / (w*w) + Bdotv*Bdotv; 

    plasma_beta = 2.0*press_2d_yz/b2
    #p_over_bvec_sqr = 2.0*press_2d_yz/(bvecx_2d_yz*bvecx_2d_yz + bvecy_2d_yz*bvecy_2d_yz + bvecz_2d_yz*bvecz_2d_yz)
    
    
    magnetisation = b2/rho_2d_yz
    return selected_iteration, time, y0, z0, dy, dz, plasma_beta, magnetisation, b2, gxx_2d_yz_ccc
    
#//////////////////////////////////////////////////////////////////////////////////////////////////////////

def plot_data(selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, gf, vmin_set, vmax_set, norm, level):

    #Prepare for plotting in yz-plane
    #Make first index along y and second index along z
    
    variable_2d_yz = np.transpose(variable_2d_yz)

    size_y_plot = variable_2d_yz.shape[0]
    size_z_plot = variable_2d_yz.shape[1]

    # Create a linear array for each axis
    yv = np.linspace(y0, (size_y_plot-1)*dy+y0, size_y_plot)
    zv = np.linspace(z0, (size_z_plot-1)*dz+z0, size_z_plot)

    # Sanity
    assert yv.shape[0] == size_y_plot
    assert zv.shape[0] == size_z_plot

    z = np.zeros(variable_2d_yz.shape)
    y = np.zeros(variable_2d_yz.shape)
    #print(variable_2d_yz.shape, yv.shape[0], zv.shape[0])
    for i in range(yv.shape[0]):
        for j in range(zv.shape[0]):
            z[i,j] = zv[j]
            y[i,j] = yv[i]

    if vmin_set is None:
        vmin = np.min(variable_2d_yz)
    else:
        vmin = vmin_set

    if vmax_set is None: 
        vmax = np.max(variable_2d_yz)
    else:
        vmax = vmax_set
         
    print("vmin = {}, vmax = {}".format(vmin, vmax)) 

    plt.figure(figsize=(14,18))
    plt.rcParams['font.size'] = 26

    if norm == "linear":     
        #plt.pcolor(1.477*y, 1.477*z,variable_2d_yz,vmin=vmin,vmax=vmax, cmap="plasma")
        #plt.pcolor(1.477*y, 1.477*z,variable_2d_yz,vmin=vmin,vmax=vmax)
        plt.pcolor(y, z,variable_2d_yz,vmin=vmin,vmax=vmax)
    elif norm == "log":
        #plt.pcolor(1.477*y, 1.477*z, variable_2d_yz, cmap="plasma", norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        plt.pcolor(y, z, variable_2d_yz, cmap="plasma", norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    elif norm == "log_abs":
        variable_2d_yz_abs= abs(variable_2d_yz)
        vmin_abs = np.min(variable_2d_yz_abs) 
        vmax_abs = np.max(variable_2d_yz_abs) 
        #plt.pcolor(1.477*y, 1.477*z, variable_2d_yz_abs, cmap="plasma", norm=colors.LogNorm(vmin=vmin_abs, vmax=vmax_abs))
        plt.pcolor(y, z, variable_2d_yz_abs, cmap="plasma", norm=colors.LogNorm(vmin=vmin_abs, vmax=vmax_abs))
    else:
        assert False, "unknown norm type"
    plt.colorbar()
    #plt.xlim(-210*1.477, 210*1.477)
    #plt.ylim(-400*1.477, 400*1.477)
    #plt.xlim(-50, 50)
    #plt.ylim(-400, 400)
    plt.title("gf = {}, it = {}, t = {}, x = {}\n min = {}, max = {}".format(gf, selected_iteration, time, x_slice, np.min(variable_2d_yz), np.max(variable_2d_yz)))
           
    plt.xlabel("y (km)")
    plt.ylabel("z (km)")
    plt.savefig("all_output/{}_rf{}_yz_it{}.png".format(gf, level, selected_iteration))
    plt.close()

#//////////////////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////////////////////

x_slice = 0.0 #-38.0 #89.0 #310.0 #111.0
#data_dir = "/gpfs/alpine/ast154/scratch/sshanka/simulations/debug_production7/output-0000/CCSN_12000km/"
#data_dir = "/gpfs/alpine/ast154/scratch/sshanka/simulations/debug_production11/output-0011/CCSN_12000km/"
#norm = "linear", "log", "log_abs"


for output_number in range(80, 119):
#for output_number in range(17, 18):
    sim_name = "CCSN_12000km"   
    #sim_name = "debug_production3"
    parfile_name = "CCSN_12000km"
    data_dir = "/gpfs/alpine/ast154/scratch/sshanka/simulations/{}/output-{}/{}/".format(sim_name, str(output_number).zfill(4), parfile_name)

    iteration_numbers = get_iteration_number_list(data_dir)
    print(iteration_numbers)

    for input_iteration in iteration_numbers:
        
        '''
        #plot entropy
        group = "hydrobase_entropy"
        gf = "hydrobase_entropy"
        norm = "linear"
        vmin_set = 0.0 #None
        vmax_set = 20.0 #None
        selected_iteration, time, y0, z0, dy, dz, variable_2d_yz = \
        get_data(group, gf, x_slice, data_dir, input_iteration)
        plot_data(selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, gf, vmin_set, vmax_set, norm)
        '''
        
        #plot Ye
        group = "hydrobase_ye"
        gf = "hydrobase_ye"
        norm = "linear"
        vmin_set = None
        vmax_set = None
        selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, level = \
        get_data(group, gf, x_slice, data_dir, input_iteration)
        plot_data(selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, gf, vmin_set, vmax_set, norm, level)
        
        '''
        
        #plot temperature
        group = "hydrobase_temperature"
        gf = "hydrobase_temperature"
        norm = "linear"
        vmin_set = 0.0
        vmax_set = 4.0
        selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, level = \
        get_data(group, gf, x_slice, data_dir, input_iteration)
        plot_data(selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, gf, vmin_set, vmax_set, norm, level)
        
        #plot entropy
        group = "hydrobase_entropy"
        gf = "hydrobase_entropy"
        norm = "linear"
        vmin_set = 0.0
        vmax_set = 20.0
        selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, level = \
        get_data(group, gf, x_slice, data_dir, input_iteration)
        plot_data(selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, gf, vmin_set, vmax_set, norm, level)
        
        #plot rho
        group = "hydrobase_rho"
        gf = "hydrobase_rho"
        norm = "log"
        vmin_set = None
        vmax_set = None
        selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, level = \
        get_data(group, gf, x_slice, data_dir, input_iteration)
        plot_data(selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, gf, vmin_set, vmax_set, norm, level)
        
        
        #plot press
        group = "hydrobase_press"
        gf = "hydrobase_press"
        norm = "log"
        vmin_set = None
        vmax_set = None
        selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, level = \
        get_data(group, gf, x_slice, data_dir, input_iteration)
        plot_data(selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, gf, vmin_set, vmax_set, norm, level)
        '''

        
        '''
        #plot Ye
        group = "hydrobase_ye"
        gf = "hydrobase_ye"
        norm = "linear"
        vmin_set = None
        vmax_set = None
        selected_iteration, time, y0, z0, dy, dz, variable_2d_yz = \
        get_data(group, gf, x_slice, data_dir, input_iteration)
        plot_data(selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, gf, vmin_set, vmax_set, norm)
        
        
        '''
        
        '''
        #plot diffusivity
        group = "grhydrox_diffusivity"
        gf = "grhydrox_diffusivity"
        norm = "linear"
        vmin_set = None
        vmax_set = None
        selected_iteration, time, y0, z0, dy, dz, variable_2d_yz = \
        get_data(group, gf, x_slice, data_dir, input_iteration)
        plot_data(selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, gf, vmin_set, vmax_set, norm)
         
        
        '''
        
        '''
        #plot Ye
        group = "hydrobase_ye"
        gf = "hydrobase_ye"
        norm = "linear"
        vmin_set = None
        vmax_set = None
        selected_iteration, time, y0, z0, dy, dz, variable_2d_yz = \
        get_data(group, gf, x_slice, data_dir, input_iteration)
        plot_data(selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, gf, vmin_set, vmax_set, norm)
        
        #plot diffusivity
        group = "grhydrox_diffusivity"
        gf = "grhydrox_diffusivity"
        norm = "linear"
        vmin_set = None
        vmax_set = None
        selected_iteration, time, y0, z0, dy, dz, variable_2d_yz = \
        get_data(group, gf, x_slice, data_dir, input_iteration)
        plot_data(selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, gf, vmin_set, vmax_set, norm)
        '''
        
        '''
        
        
        
        
        '''
        
        '''
        #plot entropy
        group = "neutrinoleakage_neutrinoleakage_abs"
        gf = "neutrinoleakage_abs_number"
        norm = "linear"
        vmin_set = None
        vmax_set = None
        selected_iteration, time, y0, z0, dy, dz, variable_2d_yz = \
        get_data(group, gf, x_slice, data_dir, input_iteration)
        plot_data(selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, gf, vmin_set, vmax_set, norm)
        '''
        
        '''
        #plot entropy
        group = "neutrinoleakage_neutrinoleakage_abs"
        gf = "neutrinoleakage_abs_energy"
        norm = "linear"
        vmin_set = None
        vmax_set = None
        selected_iteration, time, y0, z0, dy, dz, variable_2d_yz = \
        get_data(group, gf, x_slice, data_dir, input_iteration)
        plot_data(selected_iteration, time, y0, z0, dy, dz, variable_2d_yz, gf, vmin_set, vmax_set, norm)
        '''
        
        
        '''   
        #Plot plasma-beta
        norm = "log"
        vmin_set = None
        vmax_set = None
        selected_iteration, time, y0, z0, dy, dz, plasma_beta, magnetisation, b2, gxx = \
        get_plasma_beta(x_slice, data_dir, input_iteration)
        plot_data(selected_iteration, time, y0, z0, dy, dz, gxx, "gxx", None, None, "linear")
        plot_data(selected_iteration, time, y0, z0, dy, dz, plasma_beta, "plasma_beta", None, 100.0, norm)
        plot_data(selected_iteration, time, y0, z0, dy, dz, magnetisation, "magnetisation", 0.001, None, norm)
        #plot_data(selected_iteration, time, y0, z0, dy, dz, b2, "b2", None, None, norm)
        '''


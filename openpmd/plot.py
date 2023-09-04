import os
import openpmd_api as io
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors

os.system("whoami")


data_dir = "/gpfs/alpine/ast154/scratch/sshanka/simulations/CCSN_12000km/output-0059/CCSN_12000km"
#data_dir = "/home/sshanka/simulations/CCSN_12000km/output-0000/CCSN_12000km/"

group = "hydrobase_temperature"
gf = "hydrobase_temperature"
#norm = "log"
norm = "linear"
#norm = "log_abs"

x_slice = -100

vmin_set = None
vmax_set = None
#vmin_set = 1.0
vmax_set = 1.0

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

#-----------------------------------------------------
#Simple code
print("Available iterations:")
iteration_number = []
count = 0
for index in series.iterations:
    iteration_number.append(index)
    print("Available iteration[{}] = {}".format(count, index))
    count = count + 1
    
#only load the first available index for now    
itr = series.iterations[iteration_number[1]]
print("itr: {}\n".format(itr))

#which refinement level to load
level = 6

#load mesh "hydrobase_entropy_lev01" (record)
variable_mesh = itr.meshes["{}_lev{}".format(group, str(level).zfill(2))]
print("{}_mesh_lev{}: {}\n".format(group, str(level).zfill(2), variable_mesh))

#load components of mesh "hydrobase_entropy_lev01" (record components)
variable = variable_mesh[gf]
print("{}: {}\n".format(gf, variable)) 

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
z0 = -dz*(level_max_idx_z-level_min_idx_z+1)//2    
y0 = -dy*(level_max_idx_y-level_min_idx_y+1)//2   
x0 = -dx*(level_max_idx_x-level_min_idx_x+1)//2   
print("dz = {}, dy = {}, dx = {}, z0 = {}, y0 = {}, x0 = {}".format(dz, dy, dx, z0, y0, x0))

xv = np.linspace(x0, (size_x-1)*dx+x0, size_x)
x_slice_index = np.where(abs(xv-x_slice)<=dx)[0][0]
print("x_slice = {}, x_slice_index = {}".format(x_slice, x_slice_index))
#-------------------------------------------------------------------------

#After registering a data chunk such as variable_slice_yz for loading, it MUST NOT be modified or deleted until the flush() step is performed! You must not yet access variable_slice_yz!  
variable_slice_yz = variable[:, :, (level_min_idx_x+level_max_idx_x)//2]   #(z, y, x)
variable_slice_xz = variable[:, (level_min_idx_y+level_max_idx_y)//2, :]
variable_slice_xy = variable[(level_min_idx_z+level_max_idx_z)//2, :, :]

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

#print(variable_2d_yz[32, 32])


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

plt.figure(figsize=(10,10))

if norm == "linear":     
    plt.pcolor(y,z,variable_2d_yz,vmin=vmin,vmax=vmax, cmap="plasma")
elif norm == "log":
    plt.pcolor(y,z, variable_2d_yz, cmap="plasma", norm=colors.LogNorm(vmin=vmin, vmax=vmax))
elif norm == "log_abs":
    variable_2d_yz_abs= abs(variable_2d_yz)
    vmin_abs = np.min(variable_2d_yz_abs) 
    vmax_abs = np.max(variable_2d_yz_abs) 
    plt.pcolor(y,z, variable_2d_yz_abs, cmap="plasma", norm=colors.LogNorm(vmin=vmin_abs, vmax=vmax_abs))
else:
    assert False, "unknown norm type"
plt.colorbar()
plt.savefig("{}_yz.png".format(gf))
plt.close()


#/////////////////////////////////////////////////////////////////

#Let's use variable_slice_xz
variable_2d_xz = np.zeros((size_z, size_x), dtype=np.double) #ignoring x since we took slice along x
for chunk in variable.available_chunks():
    idx0_z = chunk.offset[0] - level_min_idx_z
    idx0_x = chunk.offset[2] - level_min_idx_x
    
    for k in range(chunk.extent[0]):
        for i in range(chunk.extent[2]):
            variable_2d_xz[idx0_z + k, idx0_x + i] = variable_slice_xz[chunk.offset[0] + k, chunk.offset[2] + i]

#print(variable_2d_xz[32, 32]) 

#/////////////////////////////////////////////////////////////////

#Let's use variable_slice_xy
variable_2d_xy = np.zeros((size_y, size_x), dtype=np.double) #ignoring x since we took slice along x
for chunk in variable.available_chunks():
    idx0_y = chunk.offset[1] - level_min_idx_y
    idx0_x = chunk.offset[2] - level_min_idx_x
    
    for j in range(chunk.extent[1]):
        for i in range(chunk.extent[2]):
            variable_2d_xy[idx0_y + j, idx0_x + i] = variable_slice_xy[chunk.offset[1] + j, chunk.offset[2] + i]

#print(variable_2d_xy[32, 32])             

#/////////////////////////////////////////////////////////////////


# The iteration can be closed in order to help free up resources.
itr.close()


    
    
    
    
    
    
    
    
    

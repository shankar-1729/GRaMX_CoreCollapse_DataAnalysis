import os
import openpmd_api as io
import re
import numpy as np

os.system("whoami")


#data_dir = "/home/sshanka/miscellaneous/openpmd/data/"
data_dir = "/home/sshanka/simulations/CCSN_12000km/output-0000/CCSN_12000km/"

thorn = "hydrobase"
gf_name = "entropy"

print("THORN:",thorn)
print("GF:",gf_name)

#global gf_name, thorn
#gf_name = gf_name.lower()
#thorn = thorn.lower()
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
#load iteration 400
itr = series.iterations[10]
print("itr: {}\n".format(itr))

#load mesh "hydrobase_entropy_lev01" (record)
entropy_mesh = itr.meshes["hydrobase_entropy_lev02"]
print("entropy_mesh_lev03: {}\n".format(entropy_mesh))

#load components of mesh "hydrobase_entropy_lev01" (record components)
entropy = entropy_mesh["hydrobase_entropy"]
print("hydrobase_entropy: {}\n".format(entropy)) 

print("entropy.unit_SI: ", entropy.unit_SI)
print("entropy.shape: {}\n".format(entropy.shape))

#After registering a data chunk such as entropy_slice_yz for loading, it MUST NOT be modified or deleted until the flush() step is performed! You must not yet access entropy_slice_yz!  
entropy_slice_yz = entropy[:, :, entropy.shape[2]//2]   #(z, y, x)
entropy_slice_xz = entropy[:, entropy.shape[1]//2, :]
entropy_slice_xy = entropy[entropy.shape[0]//2, :, :]

#We now flush the registered data chunks and fill them with actual data from the I/O backend.
series.flush()

#We can now work with the newly loaded data 
extent = entropy_slice_yz.shape
print("extent: {}\n".format(extent))

#/////////////////////////////////////////////////////////////////
#Access data via chunk (chunk is equivalent to box)
level_min_idx_x = 1e100
level_max_idx_x = -10
level_min_idx_y = 1e100
level_max_idx_y = -10
level_min_idx_z = 1e100
level_max_idx_z = -10
for chunk in entropy.available_chunks():
    print("")
    print(chunk)
    print("extent: ", chunk.extent, " offset: ", chunk.offset)
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


#Let's use entropy_slice_yz
entropy_2d_yz = np.zeros((size_z, size_y), dtype=np.double) #ignoring x since we took slice along x
for chunk in entropy.available_chunks():
    idx0_z = chunk.offset[0] - level_min_idx_z
    idx0_y = chunk.offset[1] - level_min_idx_y
    idx0_x = chunk.offset[2] - level_min_idx_x
    
    idx1_z = idx0_z + chunk.extent[0] 
    idx1_y = idx0_y + chunk.extent[1] 
    idx1_x = idx0_x + chunk.extent[2] 
    
    print("idx0 = [{}, {}, {}], idx1 = [{}, {}, {}]".format(idx0_z, idx0_y, idx0_x, idx1_z-1, idx1_y-1, idx1_x-1))
    
    for k in range(chunk.extent[0]):
        for j in range(chunk.extent[1]):
            entropy_2d_yz[idx0_z + k, idx0_y + j] = entropy_slice_yz[chunk.offset[0] + k, chunk.offset[1] + j]

print(entropy_2d_yz[32, 32])

#Let's use entropy_slice_xz
entropy_2d_xz = np.zeros((size_z, size_x), dtype=np.double) #ignoring x since we took slice along x
for chunk in entropy.available_chunks():
    idx0_z = chunk.offset[0] - level_min_idx_z
    idx0_x = chunk.offset[2] - level_min_idx_x
    
    for k in range(chunk.extent[0]):
        for i in range(chunk.extent[2]):
            entropy_2d_xz[idx0_z + k, idx0_x + i] = entropy_slice_xz[chunk.offset[0] + k, chunk.offset[2] + i]

print(entropy_2d_xz[32, 32]) 

#Let's use entropy_slice_xy
entropy_2d_xy = np.zeros((size_y, size_x), dtype=np.double) #ignoring x since we took slice along x
for chunk in entropy.available_chunks():
    idx0_y = chunk.offset[1] - level_min_idx_y
    idx0_x = chunk.offset[2] - level_min_idx_x
    
    for j in range(chunk.extent[1]):
        for i in range(chunk.extent[2]):
            entropy_2d_xy[idx0_y + j, idx0_x + i] = entropy_slice_xy[chunk.offset[1] + j, chunk.offset[2] + i]

print(entropy_2d_xy[32, 32])             

'''
entropy_array = np.zeros((size_z, size_y, size_x), dtype=np.double)
for chunk in entropy.available_chunks():
    idx0_z = chunk.offset[0] - level_min_idx_z
    idx0_y = chunk.offset[1] - level_min_idx_y
    idx0_x = chunk.offset[2] - level_min_idx_x
    
    idx1_z = idx0_z + chunk.extent[0] 
    idx1_y = idx0_y + chunk.extent[1] 
    idx1_x = idx0_x + chunk.extent[2] 
    
    print("idx0 = [{}, {}, {}], idx1 = [{}, {}, {}]".format(idx0_z, idx0_y, idx0_x, idx1_z-1, idx1_y-1, idx1_x-1))
    
    for k in range(chunk.extent[0]):
        for j in range(chunk.extent[1]):
            for i in range(chunk.extent[2]):
                entropy_array[idx0_z + k, idx0_y + j, idx0_x + i] = entropy[chunk.offset[0] + k, chunk.offset[1] + j, chunk.offset[2] + i]
                print(k, j, i, entropy[chunk.offset[0] + k, chunk.offset[1] + j, chunk.offset[2] + i])

print(entropy_array[1, 0, 0])    
'''
#/////////////////////////////////////////////////////////////////

# The iteration can be closed in order to help free up resources.
itr.close()

'''
for i in range(entropy_slice_yz.shape[0]):
    for j in range(entropy_slice_yz.shape[1]):
        if entropy_slice_yz[i, j] > 0.0:
            print( "[{}, {}] = {}    ".format(i, j, entropy_slice_yz[i, j]) )
'''    

#-----------------------------------------------------

'''
for index in series.iterations:
    print("index: ", index)
    
    iters = series.iterations[index]
    print("series.iterations[{}]: {}".format(index, iters))
    
    #each "iters" has lev00 to lev0n meshes available, named "hydrobase_entropy_lev0n" (total "nlevels" meshes)
    #In this case, each iteration has 5 meshes, and each mesh has 60 available chunks.
    
    for gf in iters.meshes:
        uu = iters.meshes[gf]
               
        if gf != "hydrobase_entropy_lev01":
            continue
        
        print("")
        print("iters.meshes[{}]: {}".format(gf, uu))
        print("grid spacing: {}".format(uu.grid_spacing))
        
        data_raw = uu["hydrobase_entropy"]
        print("data_raw: {}, shape = {}".format(data_raw, data_raw.shape))
        
        
        #each "data_raw" has 60 available chunks
        for chunk in data_raw.available_chunks():
            print("chunk: {}".format(chunk))
            print("offset: {}".format(chunk.offset))
            print("extent: {}".format(chunk.extent))    
'''    
    
    
    
    
    
    
    
    
    
    
    
    
    

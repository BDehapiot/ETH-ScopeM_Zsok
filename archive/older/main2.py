#%% Imports -------------------------------------------------------------------

import nd2
import cv2
import csv
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.morphology import disk, binary_dilation

#%% Parameters ----------------------------------------------------------------

sigma = 1
minProm = 10
minDist = 3
roiRadius = 6 
bgCoeff = 1.02

# -----------------------------------------------------------------------------

showProt = 'Mlp1'

#%% Initialize ----------------------------------------------------------------

data_path = Path('D:\local_Zsok\data')
csv_path = Path('D:\local_Zsok\conds.csv')

# Read cond.csv
with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    conds = [tuple(row) for row in reader]
    
# Get stack names
stack_names = []
for stack_path in data_path.iterdir():
    if stack_path.is_file():
        stack_names.append(stack_path.name)
        
#%% 

# stack_name = stack_names[1]

# # Open stack
# stack = nd2.imread(Path(data_path) / stack_name)     

# # Get condition
# condID = stack_name[4:7]
# for i, tpl in enumerate(conds):
#     if tpl[0] == condID:
#         prot = conds[i][1]
#         tp = int(conds[i][2])

# # Standard projection (stdProj)
# stdProj = np.std(stack, axis=0)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
# opened = cv2.morphologyEx(stdProj, cv2.MORPH_OPEN, kernel)
# stdProj = cv2.subtract(stdProj, opened, dst=stdProj)
# stdProj = gaussian(stdProj, sigma)

# # Sum projection (sumProj)
# sumProj = np.sum(stack, axis=0).astype('float32')
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
# opened = cv2.morphologyEx(sumProj, cv2.MORPH_OPEN, kernel)
# sumProj = cv2.subtract(sumProj, opened * bgCoeff, dst=sumProj)
# sumProj = gaussian(sumProj, sigma)

# # Local max detection
# coords = peak_local_max(
#     stdProj, min_distance=minDist, threshold_abs=minProm
#     ).astype(int)
   
# # Local max measurments
# dot_data = []
# for coord in coords:
#     y = coord[0]; x = coord[1]
#     if (y - roiRadius >= 1 and y + roiRadius <= sumProj.shape[0] -1 and 
#         x - roiRadius >= 1 and x + roiRadius <= sumProj.shape[1] -1):       

#         ROI = sumProj[
#             y - roiRadius:y + roiRadius + 1,
#             x - roiRadius:x + roiRadius + 1
#             ] * disk(roiRadius)
                
#         dot_data.append((stack_name, prot, tp, x, y, np.mean(ROI)))

# # Background measurment
# bgROI = np.zeros_like(sumProj)
# bgROI[coords[:,0], coords[:,1]] = 1
# bgROI = binary_dilation(bgROI, footprint=disk(roiRadius * 2))
# bgROI = np.invert(bgROI)
# bg = np.mean(sumProj * bgROI)
        
# stack_data = (stack_name, prot, tp, stdProj, sumProj, dot_data)


# # Display results in napari
# import napari
# viewer = napari.Viewer()
# viewer.add_image(bgROI)
# points_layer = viewer.add_points(
#     coords, 
#     size=12,
#     edge_width=0.1,
#     edge_color='red',
#     face_color='transparent',
#     opacity = 0.5,
#     )
    
#%% Process -------------------------------------------------------------------
    
def process(stack_name):

    # Open stack
    stack = nd2.imread(Path(data_path) / stack_name)     
    
    # Get condition
    condID = stack_name[4:7]
    for i, tpl in enumerate(conds):
        if tpl[0] == condID:
            prot = conds[i][1]
            tp = int(conds[i][2])

    # Standard projection (stdProj)
    stdProj = np.std(stack, axis=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    opened = cv2.morphologyEx(stdProj, cv2.MORPH_OPEN, kernel)
    stdProj = cv2.subtract(stdProj, opened, dst=stdProj)
    stdProj = gaussian(stdProj, sigma)
    
    # Sum projection (sumProj)
    sumProj = np.sum(stack, axis=0).astype('float32')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    opened = cv2.morphologyEx(sumProj, cv2.MORPH_OPEN, kernel)
    sumProj = cv2.subtract(sumProj, opened * bgCoeff, dst=sumProj)
    sumProj = gaussian(sumProj, sigma)
    
    # Local max detection
    coords = peak_local_max(
        stdProj, min_distance=minDist, threshold_abs=minProm
        ).astype(int)
       
    # Local max measurments
    dot_data = []
    for coord in coords:
        y = coord[0]; x = coord[1]
        if (y - roiRadius >= 1 and y + roiRadius <= sumProj.shape[0] -1 and 
            x - roiRadius >= 1 and x + roiRadius <= sumProj.shape[1] -1):       
    
            ROI = sumProj[
                y - roiRadius:y + roiRadius + 1,
                x - roiRadius:x + roiRadius + 1
                ] * disk(roiRadius)
                    
            dot_data.append((stack_name, prot, tp, x, y, np.mean(ROI)))
    
    # Background measurment
    bgROI = np.zeros_like(sumProj)
    bgROI[coords[:,0], coords[:,1]] = 1
    bgROI = binary_dilation(bgROI, footprint=disk(roiRadius * 2))
    bgROI = np.invert(bgROI)
    bg = np.mean(sumProj * bgROI)    
            
    stack_data = (stack_name, prot, tp, stdProj, sumProj, dot_data, bg)
    
    return stack_data

start = time.time()
print('Process')

# Parallel processing
outputs = Parallel(n_jobs=-1)(
    delayed(process)(
        stack_name,
        ) 
    for stack_name in stack_names
    )

stack_data = [data for data in outputs]
dot_data = [dot_data for data in outputs for dot_data in data[5]]

end = time.time()
print(f'  {(end-start):5.3f} s') 

#%% Results -------------------------------------------------------------------

# Extract data for selected protein (showProt)
prot_data = [data for data in dot_data if data[1] == showProt]
tps = sorted(np.unique([data[2] for data in dot_data]))
fig, ax = plt.subplots(
    len(tps), 1, figsize=(6, 2*len(tps)))

# Find xlimits
intDen = [data[5] for data in dot_data]
# xLow = np.percentile(intDen, 0.5)
xLow = 0
xHigh = np.percentile(intDen, 99.5)

# Plot results
for i, tp in enumerate(tps):
    bg = np.mean([data[6] for data in stack_data if data[1] == showProt and data[2] == tp])
    intDen = [data[5] for data in dot_data if data[1] == showProt and data[2] == tp]
    ax[i].hist(intDen, bins=200)
    ax[i].axvline(x=bg, color='r')
    ax[i].set_xlim((xLow, xHigh))
    ax[i].text(0.02, 0.94, f'{showProt} - {tp}h', 
                fontsize=12, ha='left', va='top', transform=ax[i].transAxes)
                
#%% Display -------------------------------------------------------------------

# Extract processed images
stdProj = np.stack([data[3] for data in stack_data if data[1] == showProt])
sumProj = np.stack([data[4] for data in stack_data if data[1] == showProt])

# Extract coordinates
x = [dot_data[3] for data in stack_data if data[1] == showProt for dot_data in data[5]]
y = [dot_data[4] for data in stack_data if data[1] == showProt for dot_data in data[5]]
i = [i for i, data in enumerate(stack_data) if data[1] == showProt for dot_data in data[5]]
mapping = {value: i for i, value in enumerate(sorted(set(i)))}
i = [mapping[i] for i in i]
coords = nd_array = np.column_stack((i, y, x))

# Display results in napari
import napari
viewer = napari.Viewer()
viewer.add_image(stdProj, visible=False)
viewer.add_image(sumProj)
points_layer = viewer.add_points(
    coords, 
    size=12,
    edge_width=0.1,
    edge_color='red',
    face_color='transparent',
    opacity = 0.5,
    )
#%% Imports -------------------------------------------------------------------

import nd2
import cv2
import csv
import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Skimage
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.morphology import disk, binary_dilation

#%% Parameters ----------------------------------------------------------------

sigma = 1
minProm = 6
minDist = 4
roiRadius = 4 

#%% Initialize ----------------------------------------------------------------

data_path = Path('D:\local_Zsok\data')  
conds = ["eGFP", "Mlp1", "Mlp2", "Nup96", "Nup133", "Pml39", "dpml39"]
nProt = [     1,     16,      8,      32,       16,       4,        8]

# Get stack names
stack_names = []
for stack_path in data_path.iterdir():
    if stack_path.is_file():
        stack_names.append(stack_path.name)
        
#%% Process -------------------------------------------------------------------

def process(stack_name):

    # Open stack & info
    stack = nd2.imread(Path(data_path) / stack_name) 
    date, cond, numb = stack_name.replace(".nd2", "").split("_")    
       
    # Standard projection (stdProj)
    stdProj = np.std(stack, axis=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    opened = cv2.morphologyEx(stdProj, cv2.MORPH_OPEN, kernel)
    stdProj = cv2.subtract(stdProj, opened, dst=stdProj)
    stdProj = gaussian(stdProj, sigma)
    
    # Sum projection (sumProj)
    sumProj = np.sum(stack, axis=0).astype('float32')
    
    # Local max detection
    coords = peak_local_max(
        stdProj, min_distance=minDist, threshold_abs=minProm
        ).astype(int)
    
    # Background measurment
    bgROI = np.zeros_like(sumProj, dtype=bool)
    bgROI[coords[:,0], coords[:,1]] = True
    bgROI = binary_dilation(bgROI, footprint=disk(roiRadius * 1.25)) 
    bgROI ^= binary_dilation(bgROI, footprint=disk(roiRadius * 1))
    bg = sumProj.copy()
    bg[bgROI!=True] = np.nan
    bg = np.nanmean(bg)
    
    # Local max measurments
    dot_data = []
    strel = disk(roiRadius, dtype=float)
    strel[strel==0] = np.nan
    for coord in coords:
        y = coord[0]; x = coord[1]
        if (y - roiRadius >= 20 and y + roiRadius <= sumProj.shape[0] -20 and 
            x - roiRadius >= 20 and x + roiRadius <= sumProj.shape[1] -20):       
    
            ROI = sumProj[
                y - roiRadius:y + roiRadius + 1,
                x - roiRadius:x + roiRadius + 1
                ] * strel
                    
            dot_data.append((stack_name, date, cond, numb, x, y, np.nanmean(ROI) - bg))
            
    stack_data = (stack_name, date, cond, numb, stdProj, sumProj, dot_data)
    
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
dot_data = [dot_data for data in outputs for dot_data in data[6]]

end = time.time()
print(f'  {(end-start):5.3f} s')

#%% Results -------------------------------------------------------------------

# Plot results
fig, ax = plt.subplots(len(conds), 1, figsize=(6, 2*len(conds)))
for i, cond in enumerate(conds):
    intDen = [data[6] for data in dot_data if data[2] == cond]
    ax[i].hist(intDen, bins=100)
    ax[i].set_xlim((-250, 1500))
    ax[i].text(0.98, 0.94, f'{conds[i]} ({nProt[i]})\n{np.mean(intDen):.2f}', 
                fontsize=12, ha='right', va='top', transform=ax[i].transAxes)
    
# 
dot_dataframe = pd.DataFrame(
    dot_data, columns=["stack_name", "date", "cond", "numb", "x", "y", "int"]
    )
dot_dataframe.to_csv('results.csv', index=False)
    
#%% Displays ------------------------------------------------------------------

showCond = "Mlp1"

# Extract processed images & info
names = [data[0] for data in stack_data if data[2] == showCond]
stdProj = np.stack([data[4] for data in stack_data if data[2] == showCond])
sumProj = np.stack([data[5] for data in stack_data if data[2] == showCond])
x = [dot_data[4] for data in stack_data if data[2] == showCond for dot_data in data[6]]
y = [dot_data[5] for data in stack_data if data[2] == showCond for dot_data in data[6]]
i = [i for i, data in enumerate(stack_data) if data[2] == showCond for dot_data in data[6]]
mapping = {value: i for i, value in enumerate(sorted(set(i)))}
i = [mapping[i] for i in i]
coords = nd_array = np.column_stack((i, y, x))

# Display results in napari
import napari
viewer = napari.Viewer()
viewer.add_image(sumProj, visible=True, contrast_limits=[3000, 9000])
viewer.add_image(stdProj, visible=False, contrast_limits=[0, 30])
points_layer = viewer.add_points(
    coords, 
    size=roiRadius*2,
    edge_width=0.1,
    edge_color='red',
    face_color='transparent',
    opacity = 0.5,
    )

# Show current image info
current_slice = viewer.dims.current_step[0]
text_labels = [f'{name}' for name in names]
viewer.text_overlay.text = text_labels[current_slice]
viewer.text_overlay.visible = True
def update_text_label(event):
    current_slice = viewer.dims.current_step[0]
    viewer.text_overlay.text = text_labels[current_slice]
viewer.dims.events.current_step.connect(update_text_label)
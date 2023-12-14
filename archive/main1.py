#%% Imports -------------------------------------------------------------------

import nd2
import cv2
import csv
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from skimage.morphology import disk
from skimage.filters import gaussian
from skimage.feature import peak_local_max

#%% Parameters ----------------------------------------------------------------

sigma = 1
minProm = 10
minDist = 3
roiRadius = 6 

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

#%% Measure -------------------------------------------------------------------

def measure(stack_name, conds, sigma=1, minProm=10, minDist=3, roiRadius=6):

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
    
    # Local max detection
    coords = peak_local_max(
        stdProj, min_distance=minDist, threshold_abs=minProm
        ).astype(int)
    
    # Sum projection (sumProj)
    sumProj = np.sum(stack, axis=0).astype('float32')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    opened = cv2.morphologyEx(sumProj, cv2.MORPH_OPEN, kernel)
    sumProj = cv2.subtract(sumProj, opened, dst=sumProj)
    sumProj = gaussian(sumProj, sigma)

    imgData = (stack_name, prot, tp, stdProj, sumProj)

    # Measurments
    dotData = []
    for coord in coords:
        
        y = coord[0]; x = coord[1]
        
        if (y - roiRadius >= 1 and y + roiRadius <= sumProj.shape[0] -1 and 
            x - roiRadius >= 1 and x + roiRadius <= sumProj.shape[1] -1):       
            
            ROI = sumProj[
                y - roiRadius:y + roiRadius + 1,
                x - roiRadius:x + roiRadius + 1
                ] * disk(roiRadius)
                    
            dotData.append((stack_name, prot, tp, x, y, np.sum(ROI)))
            
    return imgData#, dotData

# -----------------------------------------------------------------------------

start = time.time()
print('Measure')

# Parallel processing
outputs = Parallel(n_jobs=-1)(
    delayed(measure)(
        stack_name,
        conds,
        sigma=sigma,
        minProm=minProm,
        minDist=minDist,
        roiRadius=roiRadius
        ) 
    for stack_name in stack_names
    )

# merged_imgData = [data[0] for data in outputs]
# merged_dotData = [data for dotData in outputs for data in dotData[1]]

end = time.time()
print(f'  {(end-start):5.3f} s') 

#%% Results -------------------------------------------------------------------

# prot_name = 'Mlp1'

# prot_data = [data for data in merged_dotData if data[1] == prot_name]
# tp_points = sorted(np.unique([data[2] for data in merged_dotData]))
# fig, ax = plt.subplots(
#     len(tp_points), 1, figsize=(6, 2*len(tp_points)))

# for i, tp_point in enumerate(tp_points):
    
#     intDen = [data[5] for data in merged_dotData if data[1] == prot_name and data[2] == tp_point]
#     ax[i].hist(intDen, bins=200)
#     ax[i].set_xlim((1e+04, 1e+05))
#     ax[i].text(0.02, 0.94, f'{prot_name} - {tp_point}h', 
#                fontsize=12, ha='left', va='top', transform=ax[i].transAxes)

#%% Display -------------------------------------------------------------------

# prot_name = 'Mlp1'
# tp_point = 0



# stdProj = np.stack([
#     data[3] for data in merged_imgData
#     if data[1] == prot_name and data[2] == tp_point
#     ])

# sumProj = np.stack([
#     data[4] for data in merged_imgData
#     if data[1] == prot_name and data[2] == tp_point
#     ])


#%% Display -------------------------------------------------------------------

# import napari
# viewer = napari.Viewer()
# viewer.add_image(stdProj)
# viewer.add_image(sumProj)

# points_layer = viewer.add_points(
#     coords, 
#     size=12,
#     edge_width=0.1,
#     edge_color='red',
#     face_color='transparent',
#     opacity = 0.5,
#     )
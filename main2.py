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
    sumProj = cv2.subtract(sumProj, opened, dst=sumProj)
    sumProj = gaussian(sumProj, sigma)
    
    # Local max detection
    coords = peak_local_max(
        stdProj, min_distance=minDist, threshold_abs=minProm
        ).astype(int)
    
    # Measurments
    dot_data = []
    for coord in coords:
        y = coord[0]; x = coord[1]
        if (y - roiRadius >= 1 and y + roiRadius <= sumProj.shape[0] -1 and 
            x - roiRadius >= 1 and x + roiRadius <= sumProj.shape[1] -1):       
    
            ROI = sumProj[
                y - roiRadius:y + roiRadius + 1,
                x - roiRadius:x + roiRadius + 1
                ] * disk(roiRadius)
                    
            dot_data.append((stack_name, prot, tp, x, y, np.sum(ROI)))
            
    stack_data = (stack_name, prot, tp, stdProj, sumProj, dot_data)
    
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
                
#%% Display -------------------------------------------------------------------

prot_name = 'Mlp1'

stdProj = np.stack([data[3] for data in stack_data if data[1] == prot_name])
sumProj = np.stack([data[4] for data in stack_data if data[1] == prot_name])
ycoords = ???

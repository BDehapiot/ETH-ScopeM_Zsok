#%% Imports -------------------------------------------------------------------

import nd2
import cv2
import numpy as np
from skimage import io 
from pathlib import Path
from joblib import Parallel, delayed 

#%% Parameters ----------------------------------------------------------------

sigma = 1
minProm = 10
minDist = 3
roiRadius = 6 

#%% Initialize ----------------------------------------------------------------

data_path = 'D:\local_Zsok\data'
# stack_name = 'WellG01_PointG01_0000_ChannelSD GFP_Seq0000.nd2' # Mlp1 0hr
stack_name = 'WellO13_PointO13_0000_ChannelSD GFP_Seq0000.nd2' # Mlp1 7hr
# stack_name = 'WellF20_PointF20_0000_ChannelSD GFP_Seq0000.nd2' # Mlp1 14hr

stack = nd2.imread(Path(data_path) / stack_name) 
stack = stack[10:20,500:1500,500:1500]

#%% Process -------------------------------------------------------------------

from skimage.filters import gaussian
from skimage.restoration import rolling_ball
from skimage.feature import peak_local_max
from skimage.morphology import disk, white_tophat, binary_dilation, dilation, ball
from skimage.transform import downscale_local_mean, resize
from skimage.segmentation import expand_labels
from skimage.measure import label

# Sum projection (sumProj) for measurments
sumProj = np.sum(stack, axis=0)

# Standard projection (stdProj) for detection
stdProj = np.std(stack, axis=0)
stdProj = stdProj - rolling_ball(stdProj, radius=10)
stdProj = gaussian(stdProj, sigma)

# Local max detection (stdProj)
coords = peak_local_max(
    stdProj, min_distance=minDist, threshold_abs=minProm
    ).astype(int)

ROIs = []
for coord in coords:
    
    y = coord[0]
    x = coord[1]
    
    if (y - roiRadius >= 0 and y + roiRadius <= sumProj.shape[0] and 
        x - roiRadius >= 0 and x + roiRadius <= sumProj.shape[1]):
        
        ROI = sumProj[y - roiRadius:y + roiRadius, x - roiRadius:x + roiRadius]
        ROIs.append((x, y, ROI, np.sum(ROI)))
    
    

#%% ctrd -------------------------------------------------------------------

import napari
viewer = napari.Viewer()
viewer.add_image(stdProj)
viewer.add_image(sumProj)

points_layer = viewer.add_points(
    coords, 
    size=12,
    edge_width=0.1,
    edge_color='red',
    face_color='transparent',
    opacity = 0.5,
    )
#%% Imports -------------------------------------------------------------------

import nd2
import cv2
import numpy as np
from skimage import io 
from pathlib import Path
from joblib import Parallel, delayed 

#%% Parameters ----------------------------------------------------------------

sigma = 1
radius = 5
rsize_factor = 0.1
minProm = 20
minDist = 3

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

def process(
        stack, 
        sigma=1, 
        radius=5, 
        rsize_factor=0.1, 
        minProm=10,
        minDist=3, 
        ):

    # Scale parameters acc. to rsize_factor
    radius = round(radius * rsize_factor)
    
    def _process(img):
        
        # preprocess image
        img = gaussian(img, sigma, preserve_range=True)
        rscale_img = resize(
            img, (img.shape[0]*rsize_factor, img.shape[1]*rsize_factor))
        background = rolling_ball(rscale_img, radius=radius)
        background = resize(background, (img.shape[0], img.shape[1]))
        img = img - background
        
        # Local max detection
        coords = peak_local_max(
            img, min_distance=minDist, threshold_abs=minProm
            ).astype(int)
        
        # Local max display
        display = np.zeros(img.shape, dtype=bool)
        display[coords[:,0], coords[:,1]] = True
        display = label(display)
        display = expand_labels(display, distance=5)
        
        return img, coords, display
    
    # Parallel processing
    outputs = Parallel(n_jobs=-1)(
        delayed(_process)(img) 
        for img in stack
        )
    
    # Extract outputs
    pStack = np.stack([data[0] for data in outputs])
    display = np.stack([data[2] for data in outputs])
    coords = [data[1] for data in outputs]
    
    return pStack, coords, display

# Process raw stack image
pStack, coords, display = process(
    stack, 
    sigma=sigma, 
    radius=radius, 
    rsize_factor=rsize_factor, 
    minDist=minDist, 
    minProm=minProm,
    )

#%% ctrd -------------------------------------------------------------------

# import napari
# viewer = napari.Viewer()
# viewer.add_image(pStack)
# viewer.add_labels(display)

# viewer.add_image(img)
# points_layer = viewer.add_points(
#     coords, 
#     size=20,
#     edge_width=0.1,
#     edge_color='red',
#     face_color='transparent',
#     opacity = 0.5,
#     )
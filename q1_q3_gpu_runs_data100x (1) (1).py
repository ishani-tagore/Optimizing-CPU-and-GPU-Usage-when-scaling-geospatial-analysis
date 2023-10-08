import rasterio
import numpy as np
import time
from numba import vectorize, cuda

# Import bands as separate images;
band4 = rasterio.open(
      '/home/imajumd0/week3/landsat8/LC08_B4.tif') #red
band5 = rasterio.open(
      '/home/imajumd0/week3/landsat8/LC08_B5.tif') #nir

# Convert nir and red objects to float64 arrays
red = band4.read(1).astype('float64')
nir = band5.read(1).astype('float64')

red100 = np.tile(red, 100)
nir100 = np.tile(nir, 100)
del band4, band5

@vectorize(['f8(f8, f8)'], target='cuda')
def ndvi_calc(x, y):
  #NDVI calculation
  ndvi = (y - x) / (y + x)
  return ndvi

t0 = time.time()
final_ndvi = ndvi_calc(red100, nir100)
gpu_runtime = time.time() - t0

with open("/home/imajumd0/week3/GPU_Data100x.out", "w") as f:
    f.write(str(gpu_runtime))

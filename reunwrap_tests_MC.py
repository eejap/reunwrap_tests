#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues 13 June 2023

@author: eejap
"""
#%%
import matplotlib as plt
# this package is for general plotting
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
import imageio
import datetime
import warnings
import cmcrameri.cm as cmc
warnings.filterwarnings('ignore')


#%% Import vel tifs
ruw_list = ['default', 'gaussian', 'lowpass', 'goldstein_cascade'] 
tif_end = '.vel.geo.tif'
file_dir = '/nfs/a285/homes/eejap/reunwrap_tests/078A_07049_131313/reunw_vels/'

#  Create list of tif names
vel_tifs = []
for i in range(len(ruw_list)):
    tif = file_dir + ruw_list[i] + tif_end
    vel_tifs.append(tif)
    
# Import loop err tifs and save them in list called loop_err_list
vel_list =[]
for tif in vel_tifs:
    imp = gdal.Open(tif)
    vel_list.append(imp) 
    
# Import vel tiffs as arrays and save them in list called vel_arr_list
vel_arr_list =[]
for tif in vel_tifs:
    imp = imageio.imread(tif)
    vel_arr_list.append(imp)

#%% Calc fastest pixel and mean vel
vel_arr_list =[]
for tif in vel_tifs:
    imp = imageio.imread(tif)
    vel_arr_list.append(imp)

vel_min = np.zeros(len(vel_arr_list))
for i, vel in enumerate(vel_arr_list):
    vel_min[i] = np.nanmin(vel)
    
vel_means = np.zeros(len(vel_arr_list))
for i, vel in enumerate(vel_arr_list):
    vel_means[i] = np.nanmean(vel)
    
#%% Plot velocities and profile trace
fig, axs = plt.subplots(2,2, figsize=(9, 10))
titles_vels = ["No reunwrapping", "Gaussian", "Lowpass", "Goldstein + Cascade"]

vmax = 150
vmin = -150

# Define start and end points of profile
#start_point = (-99.2, 19.4) # W-E
#end_point = (-98.87, 19.47) # W-E
start_point = (-99.18, 19.836) # N-S
end_point = (-98.98, 19.127) # N-S

plot = vel_list[0]
# Get the geotransform
geotransform = plot.GetGeoTransform()
arr = plot.ReadAsArray()
# Define the x and y coordinates of each pixel
x = geotransform[0] + geotransform[1] * np.arange(arr.shape[1])
y = geotransform[3] + geotransform[5] * np.arange(arr.shape[0])
# Create a meshgrid of the x and y coordinates
xx, yy = np.meshgrid(x, y)

for i in range(2):
    for j in range(2):
        k = i*2 + j
        if k < len(vel_arr_list):
            im = axs[j,i].imshow(vel_arr_list[k], cmap='cmc.vik', interpolation='none', vmax=vmax, vmin=vmin, extent=[x.min(), x.max(), y.min(), y.max()])
            axs[j,i].set_title(titles_vels[k])
            #axs[j,i].text(52.8, 36.9, 'Mean vel: {:.0f} mm/yr\nMax vel: {:.0f} mm/yr'.format(vel_means[k], vel_min[k]), ha='right')
            # Plot trace - comment out to remove
            axs[0,0].plot([end_point[0], start_point[0]], [end_point[1], start_point[1]], color='red')
        if j < 1:
            axs[j,i].set_xticklabels([])
        if i > 0:
            axs[j,i].set_yticklabels([])

axs[0,0].plot(-99.068,19.432, 'rX', markersize=12) #MMX1
axs[0,0].plot(-99.016,19.256, 'rX', markersize=12) #UTUL

font = {"size": 14}

axs[0,0].text(-99.068-0.18, 19.432-0.07, 'MMX1', fontdict=font)
axs[0,0].text(-99.016-0.13, 19.256-0.06, 'UTUL', fontdict=font)

# Set a label at the base of the colorbar
cbar = fig.colorbar(im, ax=axs, location='right', pad=0.05, aspect=30)
cbar.ax.set_xlabel('mm/yr', fontsize=12, labelpad=10)

# Adjust colorbar width and height
cbar.ax.set_position([0.78, 0.2, 0.9, 0.6]) # left, bottom, width, height

plt.savefig('/nfs/see-fs-02_users/eejap/public_html/sarwatch_paper/vels_line.png', dpi=200)
#plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/078A_07049_131313/results/vels.png', dpi=200)

#%% Calculate vel differences

# titles_vels = ['No reunwrapping', 'Goldstein + Cascade']
# vel_dict = dict(zip(titles_vels, vel_arr_list))

# vel_diffs = {}
# for i, arr1_name in enumerate(vel_dict):
#     for j, arr2_name in enumerate(vel_dict):
#         if i < j:
#             diff_name = f"({arr1_name})-({arr2_name})"  # name of difference
#             arr1 = vel_dict[arr1_name]  # get array 1
#             arr2 = vel_dict[arr2_name]  # get array 2
#             diff = arr1 - arr2  # calculate the difference
#             vel_diffs[diff_name] = diff  # store the difference in the dictionar

#%% Separate out subsidence pixels and compute their differences
vel_fast_list = []

for i, vel in enumerate(vel_arr_list):
    vel_mask = vel <= -10
    arr_fast = np.empty_like(vel, dtype=float)
    arr_fast.fill(np.nan)
    arr_fast[vel_mask] = vel[vel_mask]
    vel_fast_list.append(arr_fast)
    
#%% find the mean, max, std of each fast velocity output

vel_fast_means = np.zeros(len(vel_fast_list))
for i, vel in enumerate(vel_fast_list):
    vel_fast_means[i] = np.nanmean(vel)

vel_fast_min = np.zeros(len(vel_fast_list))
for i, vel in enumerate(vel_fast_list):
    vel_fast_min[i] = np.nanmin(vel)
    
vel_fast_std = np.zeros(len(vel_fast_list))
for i, vel in enumerate(vel_fast_list):
    vel_fast_std[i] = np.nanstd(vel)
    
#%% Plot these arrays (fast pixels only!)
fig, axs = plt.subplots(2,2, figsize=(12,12))
vmin = -190
vmax = -10

plot = vel_list[0]
# Get the geotransform
geotransform = plot.GetGeoTransform()
arr = plot.ReadAsArray()
# Define the x and y coordinates of each pixel
x = geotransform[0] + geotransform[1] * np.arange(arr.shape[1])
y = geotransform[3] + geotransform[5] * np.arange(arr.shape[0])
# Create a meshgrid of the x and y coordinates
xx, yy = np.meshgrid(x, y)

for i in range(2):
    for j in range(2):
        k = i*2 + j
        if k < len(vel_fast_list):
            im = axs[j,i].imshow(vel_fast_list[k], cmap='viridis', interpolation='none', vmax=vmax, vmin=vmin, extent=[x.min(), x.max(), y.min(), y.max()])
            axs[j,i].set_title(titles_vels[k])
            #axs[j,i].text(52.8, 36.9, 'Mean vel: {:.0f} mm/yr\nMax vel: {:.0f} mm/yr'.format(vel_fast_means[k], vel_fast_min[k]), ha='right')
            # Plot trace - comment out to remove
            #axs[j,i].plot([end_point[0], start_point[0]], [end_point[1], start_point[1]], color='red')
        if j < 2:
            axs[j,i].set_xticklabels([])
        if i > 0:
            axs[j,i].set_yticklabels([])

# Set a label at the base of the colorbar
cbar = fig.colorbar(im, ax=axs, location='right', pad=0.05, aspect=30)
cbar.ax.set_xlabel('mm/yr', fontsize=12, labelpad=10)
# Adjust colorbar width and height
cbar.ax.set_position([0.78, 0.2, 0.9, 0.6]) # left, bottom, width, height
    
plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/078A_07049_131313/results/vel_fast.png', dpi=200)

#%% Difference the fast vel arrays
vel_fast_dict = dict(zip(titles_vels, vel_fast_list))

vel_fast_diffs = {}
for i, arr1_name in enumerate(vel_fast_dict):
    for j, arr2_name in enumerate(vel_fast_dict):
        if i < j:
            diff_name = f"({arr1_name})-({arr2_name})"  # name of difference
            arr1 = vel_fast_dict[arr1_name]  # get array 1
            arrtest = arr1[0:900,0:650] # resize array 1 to diff from array 2
            arr2 = vel_fast_dict[arr2_name]  # get array 2
            diff = arrtest - arr2  # calculate the difference
            vel_fast_diffs[diff_name] = diff  # store the difference in the dictionary
            
#%% Compute means, max, min differences
vel_fast_diffs_keys = list(vel_fast_diffs.keys())
vel_fast_diffs_values = list(vel_fast_diffs.values())

vel_fast_diffs_means = np.zeros(len(vel_fast_diffs_keys))
for i, vel in enumerate(vel_fast_diffs_values):
    vel_fast_diffs_means[i] = np.nanmean(vel)

vel_fast_diffs_max = np.zeros(len(vel_fast_diffs_keys))
for i, vel in enumerate(vel_fast_diffs_values):
    vel_fast_diffs_max[i] = np.nanmax(vel)
    
vel_fast_diffs_min = np.zeros(len(vel_fast_diffs_keys))
for i, vel in enumerate(vel_fast_diffs_values):
    vel_fast_diffs_min[i] = np.nanmin(vel)
    
vel_fast_diffs_abs_max = np.maximum(np.abs(vel_fast_diffs_min), np.abs(vel_fast_diffs_max))

for i in range(len(vel_fast_diffs_keys)):
    print('Velocity diff', vel_fast_diffs_keys[i])
    print('has mean: {:.0f} mm/yr, has min diff {:.0f} mm/yr, and max diff {:.0f} mm/yr'.format(vel_fast_diffs_means[i], vel_fast_diffs_min[i],vel_fast_diffs_max[i]))

#%% Plot fast vel differences
fig, axs = plt.subplots(3, 2, figsize=(10,14))

vmax = 100
vmin = -150
    
for i in range(2):
    for j in range(3):
        k = i*3 + j
        if k < len(vel_fast_diffs_keys):
            im = axs[j,i].imshow(vel_fast_diffs_values[k], cmap='viridis', interpolation='none', vmax=vmax, vmin=vmin, extent=[x.min(), x.max(), y.min(), y.max()])
            axs[j,i].set_title(vel_fast_diffs_keys[k])
            #axs[j,i].text(52.95, 37.0, 'Mean diff: {:.0f} mm/yr\nMax absolute diff: {:.0f} mm/yr'.format(vel_fast_diffs_means[k], vel_fast_diffs_abs_max[k]), ha='right')
        if j < 2:
            axs[j,i].set_xticklabels([])
        if i > 0:
            axs[j,i].set_yticklabels([])
            
# Set a label at the base of the colorbar
cbar = fig.colorbar(im, ax=axs, location='right', pad=0.05, aspect=30)
cbar.ax.set_xlabel('mm/yr', fontsize=12, labelpad=10)
# Adjust colorbar width and height
cbar.ax.set_position([0.78, 0.3, 0.015, 0.4]) # left, bottom, width, bottom

plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/078A_07049_131313/results/vel_fast_diffs.png', dpi=200)

#%% Draw profiles through velocities

# Store arrays in a dictionary
vel_dict = dict(zip(titles_vels, vel_list))

# Define start and end points of profile
#start_point = (-99.2, 19.4) # W-E
#end_point = (-98.87, 19.47) # W-E
start_point = (-99.18, 19.836) # N-S
end_point = (-98.98, 19.127) # N-S

# Create an array of indices for the profile
y, x = np.linspace(start_point[0], end_point[0], 1000), np.linspace(start_point[1], end_point[1], 100)
plt.figure(figsize=(10,6))

# Loop through arrays in the dictionary and plot a profile for each
for name, ds in vel_dict.items():
    # Get the array values along the profile
    geotransform = ds.GetGeoTransform()
    arr = ds.ReadAsArray()
    # Create an array of indices for the profile
    x = np.linspace(start_point[0], end_point[0], 150)
    y = np.linspace(start_point[1], end_point[1], 150)
    x_px = (x - geotransform[0]) / geotransform[1]  # convert x coordinates to pixels
    y_px = (y - geotransform[3]) / geotransform[5]  # convert y coordinates to pixels
    
    # Get the array values along the profile
    values = [arr[int(round(y)), int(round(x))] for x, y in zip(x_px, y_px)]
    # Plot the profile
    plt.plot(x, values,  label=name)
    
# Add legend and axis labels
plt.legend(loc='lower left', fontsize=12)
plt.xlabel('Longitude (deg)', fontsize=13)
plt.ylabel('LoS Velocity (mm/yr)', fontsize=13)
plt.tick_params(axis='both', which='major', labelsize=14)

# Set y-axis properties to plot on the right side
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/078A_07049_131313/results/vels_profile_poster.png', dpi=200)
#plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/078A_07049_131313/results/vels_profile.png', dpi=200)

#%% import loop error tifs
ruw_list = ['default', 'gaussian', 'lowpass', 'goldstein', 'goldstein_cascade']
tif_end = '.n_loop_err.geo.tif'
file_dir = '/nfs/a285/homes/eejap/reunwrap_tests/078A_07049_131313/reunw_vels/'

#  Create list of tif names
vel_tifs = []
for i in range(len(ruw_list)):
    tif = file_dir + ruw_list[i] + tif_end
    vel_tifs.append(tif)
    
# Import loop err tifs and save them in list called loop_err_list
vel_list =[]
for tif in vel_tifs:
    imp = gdal.Open(tif)
    vel_list.append(imp) 
    
# Import vel tiffs as arrays and save them in list called vel_arr_list
vel_arr_list =[]
for tif in vel_tifs:
    imp = imageio.imread(tif)
    vel_arr_list.append(imp)

#%%
fig, axs = plt.subplots(3,2, figsize=(9, 10))
titles_vels = ["No reunwrapping", "Gaussian", "Lowpass", "Goldstein", "Goldstein + Cascade"]

vmax = 150
vmin = -150

# Define start and end points of profile
#start_point = (-99.2, 19.4) # W-E
#end_point = (-98.87, 19.47) # W-E
start_point = (-99.18, 19.836) # N-S
end_point = (-98.98, 19.127) # N-S

plot = vel_list[0]
# Get the geotransform
geotransform = plot.GetGeoTransform()
arr = plot.ReadAsArray()
# Define the x and y coordinates of each pixel
x = geotransform[0] + geotransform[1] * np.arange(arr.shape[1])
y = geotransform[3] + geotransform[5] * np.arange(arr.shape[0])
# Create a meshgrid of the x and y coordinates
xx, yy = np.meshgrid(x, y)

for i in range(2):
    for j in range(3):
        k = i*3 + j
        if k < len(vel_arr_list):
            im = axs[j,i].imshow(vel_arr_list[k], interpolation='none', extent=[x.min(), x.max(), y.min(), y.max()])
            axs[j,i].set_title(titles_vels[k])
            #axs[j,i].text(52.8, 36.9, 'Mean vel: {:.0f} mm/yr\nMax vel: {:.0f} mm/yr'.format(vel_means[k], vel_min[k]), ha='right')
            # Plot trace - comment out to remove
            axs[0,0].plot([end_point[0], start_point[0]], [end_point[1], start_point[1]], color='red')
        if j < 1:
            axs[j,i].set_xticklabels([])
        if i > 0:
            axs[j,i].set_yticklabels([])

# Set a label at the base of the colorbar
cbar = fig.colorbar(im, ax=axs, location='right', pad=0.05, aspect=30)
cbar.ax.set_xlabel('mm/yr', fontsize=12, labelpad=10)

# Adjust colorbar width and height
cbar.ax.set_position([0.78, 0.2, 0.9, 0.6]) # left, bottom, width, height

#plt.savefig('/nfs/see-fs-02_users/eejap/public_html/sarwatch_paper/vels_line.png', dpi=200)
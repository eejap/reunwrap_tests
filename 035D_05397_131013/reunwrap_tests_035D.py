#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:24:18 2023

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

#%% Import vstd tifs
ruw_list = ['default', 'cascade', 'lowpass', 'goldstein', 'gauss', 'cascade_gauss', 'lowpass_gauss', 'goldstein_cascade']
tif_end = '.vstd.geo.tif'
file_dir = '/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/reunw_vels/'

#  Create list of tif names
vstd_tifs = []
for i in range(len(ruw_list)):
    tif = file_dir + ruw_list[i] + tif_end
    vstd_tifs.append(tif)
    
# Import loop err tifs and save them in list called loop_err_list
vstd_list =[]
for tif in vstd_tifs:
    imp = gdal.Open(tif)
    vstd_list.append(imp) 
    
# Import vstd tiffs as arrays and save them in list called vstd_arr_list
vstd_arr_list =[]
for tif in vstd_tifs:
    imp = imageio.imread(tif)
    vstd_arr_list.append(imp)

#%% calc max and mean vstds
vstd_max = np.zeros(len(vstd_arr_list))
for i, vstd in enumerate(vstd_arr_list):
    vstd_max[i] = np.nanmax(vstd)   

#%% Plot vstds
fig, axs = plt.subplots(4,2, figsize=(10,10))
titles_vstd = ["Default", "Cascade", "Lowpass", "Goldstein", "Gaussian", "Gaussian + Cascade", "Gaussian + Lowpass", "Goldstein + Cascade"]
vmax = 2
vmin = 0

plot = vstd_list[0]
# Get the geotransform
geotransform = plot.GetGeoTransform()
arr = plot.ReadAsArray()
# Define the x and y coordinates of each pixel
x = geotransform[0] + geotransform[1] * np.arange(arr.shape[1])
y = geotransform[3] + geotransform[5] * np.arange(arr.shape[0])
# Create a meshgrid of the x and y coordinates
xx, yy = np.meshgrid(x, y)    

for i in range(2):
    for j in range(4):
        k = i*4 + j
        if k < len(vstd_arr_list):
            im = axs[j,i].imshow(vstd_arr_list[k], cmap='viridis', interpolation='none', vmax=vmax, vmin=vmin, extent=[x.min(), x.max(), y.min(), y.max()])
            axs[j,i].set_title(titles_vstd[k])
            axs[j,i].text(52.95, 37.1, 'Max vstd: {:.1f} mm/yr'.format(vstd_max[k]), ha='right')
        if j < 3:
            axs[j,i].set_xticklabels([])
        if i > 0:
            axs[j,i].set_yticklabels([])
            
# Set a label at the base of the colorbar
cbar = fig.colorbar(im, ax=axs, location='right', pad=0.05, aspect=30)
cbar.ax.set_xlabel('mm/yr', fontsize=12, labelpad=10)
# Adjust colorbar width and height
cbar.ax.set_position([0.78, 0.2, 0.9, 0.6]) # left, bottom, width, height

plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/vstd.png', dpi=200)

#%% Calculate vstd differences

titles = ["Default", "Cascade", "Lowpass", "Goldstein", "Gaussian", "Gaussian + Cascade", "Gaussian + Lowpass", "Goldstein + Cascade"]
vstd_dict = dict(zip(titles, vstd_arr_list))

vstd_diffs = {}
for i, arr1_name in enumerate(vstd_dict):
    for j, arr2_name in enumerate(vstd_dict):
        if i < j:
            diff_name = f"({arr1_name})-({arr2_name})"  # name of difference
            arr1 = vstd_dict[arr1_name]  # get array 1
            arr2 = vstd_dict[arr2_name]  # get array 2
            diff = arr1 - arr2  # calculate the difference
            vstd_diffs[diff_name] = diff  # store the difference in the dictionary

#%% Compute means, max, min differences
keys = list(vstd_diffs.keys())
vstd_diff_arr = list(vstd_diffs.values())

vstd_diffs_means = np.zeros(len(keys))
for i, vel in enumerate(vstd_diff_arr):
    vstd_diffs_means[i] = np.nanmean(vel)

vstd_diffs_max = np.zeros(len(keys))
for i, vel in enumerate(vstd_diff_arr):
    vstd_diffs_max[i] = np.nanmax(vel)
    
vstd_diffs_min = np.zeros(len(keys))
for i, vel in enumerate(vstd_diff_arr):
    vstd_diffs_min[i] = np.nanmin(vel)
    
vstd_diffs_abs_max = np.maximum(np.abs(vstd_diffs_min), np.abs(vstd_diffs_max))

#%% Plot vstd differences
fig, axs = plt.subplots(5, 3, figsize=(16,18))

vmax = 1
vmin = -1

plot = vstd_list[0]
# Get the geotransform
geotransform = plot.GetGeoTransform()
arr = plot.ReadAsArray()
# Define the x and y coordinates of each pixel
x = geotransform[0] + geotransform[1] * np.arange(arr.shape[1])
y = geotransform[3] + geotransform[5] * np.arange(arr.shape[0])
# Create a meshgrid of the x and y coordinates
xx, yy = np.meshgrid(x, y)

for i in range(3):
    for j in range(5):
        k = i*5 + j
        if k < len(vstd_diff_arr):
            im = axs[j,i].imshow(vstd_diff_arr[k], cmap='viridis', interpolation='none', vmax=vmax, vmin=vmin, extent=[x.min(), x.max(), y.min(), y.max()])
            #plt.colorbar(im, ax=axs[j,i], label="mm/yr")
            axs[j,i].set_title(keys[k])
            axs[j,i].text(52.95, 36.95, 'Mean diff: {:.0f} mm/yr\nMax abs diff: {:.0f} mm/yr'.format(vstd_diffs_means[k], vstd_diffs_abs_max[k]), ha='right')
        if j < 4:
            axs[j,i].set_xticklabels([])
        if i > 0:
            axs[j,i].set_yticklabels([])

# Set a label at the base of the colorbar
cbar = fig.colorbar(im, ax=axs, location='right', pad=0.05, aspect=30)
cbar.ax.set_xlabel('mm/yr', fontsize=12, labelpad=10)
# Adjust colorbar width and height
cbar.ax.set_position([0.78, 0.3, 0.015, 0.4]) # left, bottom, width, bottom

plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/vstd_diffs.png', dpi=200)

#%% Import vel tifs
ruw_list = ['default', 'cascade', 'lowpass', 'goldstein', 'gauss', 'cascade_gauss', 'lowpass_gauss', 'goldstein_cascade']
tif_end = '.vel.geo.tif'
file_dir = '/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/reunw_vels/'

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
fig, axs = plt.subplots(4,2, figsize=(10, 12))
titles_vels = ["Default", "Cascade", "Lowpass", "Goldstein", "Gaussian", "Gaussian + Cascade", "Gaussian + Lowpass", "Goldstein + Cascade"]

vmax = 150
vmin = -150

# Define start and end points of profile
start_point = (50.2, 36.1)
end_point = (52, 35.25)

plot = vstd_list[0]
# Get the geotransform
geotransform = plot.GetGeoTransform()
arr = plot.ReadAsArray()
# Define the x and y coordinates of each pixel
x = geotransform[0] + geotransform[1] * np.arange(arr.shape[1])
y = geotransform[3] + geotransform[5] * np.arange(arr.shape[0])
# Create a meshgrid of the x and y coordinates
xx, yy = np.meshgrid(x, y)

for i in range(2):
    for j in range(4):
        k = i*4 + j
        if k < len(vel_arr_list):
            im = axs[j,i].imshow(vel_arr_list[k], cmap='cmc.vik', interpolation='none', vmax=vmax, vmin=vmin, extent=[x.min(), x.max(), y.min(), y.max()])
            axs[j,i].set_title(titles_vels[k])
            axs[j,i].text(52.8, 36.9, 'Mean vel: {:.0f} mm/yr\nMax vel: {:.0f} mm/yr'.format(vel_means[k], vel_min[k]), ha='right')
            # Plot trace - comment out to remove
            #axs[j,i].plot([end_point[0], start_point[0]], [end_point[1], start_point[1]], color='red')
        if j < 3:
            axs[j,i].set_xticklabels([])
        if i > 0:
            axs[j,i].set_yticklabels([])


# Set a label at the base of the colorbar
cbar = fig.colorbar(im, ax=axs, location='right', pad=0.05, aspect=30)
cbar.ax.set_xlabel('mm/yr', fontsize=12, labelpad=10)

# Adjust colorbar width and height
cbar.ax.set_position([0.78, 0.2, 0.9, 0.6]) # left, bottom, width, height

#plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/vels_line.png', dpi=200)
plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/vels.png', dpi=200)

#%% Plot default velocity with line
# Import vel tifs
pre = ['default']
tif_end = '.vel.geo.tif'
file_dir = '/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/reunw_vels/'

#  Create list of tif names
vel_tif = []
for i in range(len(pre)):
    tif = file_dir + pre[i] + tif_end
    vel_tif.append(tif)
    
# Import loop err tifs and save them in list called loop_err_list
vel_list =[]
for tif in vel_tif:
    imp = gdal.Open(tif)
    vel_list.append(imp) 

# Calc fastest pixel and mean vel
vel_arr =[]
for tif in vel_tif:
    imp = imageio.imread(tif)
    vel_arr.append(imp)

fig, axs = plt.subplots(figsize=(9, 9))

# Define start and end points of profile
start_point = (50.2, 36.1)
end_point = (52, 35.25)

plot = vstd_list[0]
# Get the geotransform
geotransform = plot.GetGeoTransform()
arr = plot.ReadAsArray()
# Define the x and y coordinates of each pixel
x = geotransform[0] + geotransform[1] * np.arange(arr.shape[1])
y = geotransform[3] + geotransform[5] * np.arange(arr.shape[0])
# Create a meshgrid of the x and y coordinates
xx, yy = np.meshgrid(x, y)

im = axs.imshow(vel_arr[0], cmap='cmc.vik', interpolation='none', vmax=vmax, vmin=vmin, extent=[x.min(), x.max(), y.min(), y.max()])

# Plot trace - comment out to remove
axs.plot([end_point[0], start_point[0]], [end_point[1], start_point[1]], color='red')

# Set a label at the base of the colorbar
cbar = fig.colorbar(im, ax=axs, location='right', pad=0.05, aspect=30)
cbar.ax.set_xlabel('mm/yr', fontsize=12, labelpad=10)

# Adjust colorbar width and height
cbar.ax.set_position([0.79, 0.25, 0.9, 0.5]) # left, bottom, width, height

plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/default_vel_line.png', dpi=200)

#%% Calculate vel differences

titles = ["Default", "Cascade", "Lowpass", "Goldstein", "Gaussian", "Gaussian + Cascade", "Gaussian + Lowpass", "Goldstein + Cascade"]
vel_dict = dict(zip(titles, vel_arr_list))

vel_diffs = {}
for i, arr1_name in enumerate(vel_dict):
    for j, arr2_name in enumerate(vel_dict):
        if i < j:
            diff_name = f"({arr1_name})-({arr2_name})"  # name of difference
            arr1 = vel_dict[arr1_name]  # get array 1
            arr2 = vel_dict[arr2_name]  # get array 2
            diff = arr1 - arr2  # calculate the difference
            vel_diffs[diff_name] = diff  # store the difference in the dictionar

#%% Compute means, max, min differences of vels
keys = list(vel_diffs.keys())
vel_diffs_values = list(vel_diffs.values())

vel_diffs_means = np.zeros(len(keys))
for i, vel in enumerate(vel_diffs_values):
    vel_diffs_means[i] = np.nanmean(vel)

vel_diffs_max = np.zeros(len(keys))
for i, vel in enumerate(vel_diffs_values):
    vel_diffs_max[i] = np.nanmax(vel)
    
vel_diffs_min = np.zeros(len(keys))
for i, vel in enumerate(vel_diffs_values):
    vel_diffs_min[i] = np.nanmin(vel)
    
vel_diffs_abs_max = np.maximum(np.abs(vel_diffs_min), np.abs(vel_diffs_max))

#%% Plot vel differences
im, axs = plt.subplots(5, 3, figsize=(16,18))
vmax=100
vmin=-100

for i in range(3):
    for j in range(5):
        k = i*5 + j
        if k < len(vel_diffs_values):
            im = axs[j,i].imshow(vel_diffs_values[k], cmap='viridis', interpolation='none', vmax=vmax, vmin=vmin, extent=[x.min(), x.max(), y.min(), y.max()])
            axs[j,i].set_title(keys[k])
            axs[j,i].text(52.95, 37.0, 'Mean diff: {:.0f} mm/yr\nMax absolute diff: {:.0f} mm/yr'.format(vel_diffs_means[k], vel_diffs_abs_max[k]), ha='right')
        if j < 4:
            axs[j,i].set_xticklabels([])
        if i > 0:
            axs[j,i].set_yticklabels([])
            
# Set a label at the base of the colorbar
cbar = fig.colorbar(im, ax=axs, location='right', pad=0.05, aspect=30)
cbar.ax.set_xlabel('mm/yr', fontsize=12, labelpad=10)
# Adjust colorbar width and height
cbar.ax.set_position([0.78, 0.3, 0.015, 0.4]) # left, bottom, width, bottom

plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/vel_diffs.png', dpi=200)

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
fig, axs = plt.subplots(4,2, figsize=(12,12))
titles = ["Default", "Cascade", "Lowpass", "Goldstein", "Gaussian", "Gaussian + Cascade", "Gaussian + Lowpass", "Goldstein + Cascade"]
vmin = -190
vmax = -10

plot = vstd_list[0]
# Get the geotransform
geotransform = plot.GetGeoTransform()
arr = plot.ReadAsArray()
# Define the x and y coordinates of each pixel
x = geotransform[0] + geotransform[1] * np.arange(arr.shape[1])
y = geotransform[3] + geotransform[5] * np.arange(arr.shape[0])
# Create a meshgrid of the x and y coordinates
xx, yy = np.meshgrid(x, y)

for i in range(2):
    for j in range(4):
        k = i*4 + j
        if k < len(vel_fast_list):
            im = axs[j,i].imshow(vel_fast_list[k], cmap='viridis', interpolation='none', vmax=vmax, vmin=vmin, extent=[x.min(), x.max(), y.min(), y.max()])
            axs[j,i].set_title(titles[k])
            axs[j,i].text(52.8, 36.9, 'Mean vel: {:.0f} mm/yr\nMax vel: {:.0f} mm/yr'.format(vel_fast_means[k], vel_fast_min[k]), ha='right')
            # Plot trace - comment out to remove
            #axs[j,i].plot([end_point[0], start_point[0]], [end_point[1], start_point[1]], color='red')
        if j < 3:
            axs[j,i].set_xticklabels([])
        if i > 0:
            axs[j,i].set_yticklabels([])

# Set a label at the base of the colorbar
cbar = fig.colorbar(im, ax=axs, location='right', pad=0.05, aspect=30)
cbar.ax.set_xlabel('mm/yr', fontsize=12, labelpad=10)
# Adjust colorbar width and height
cbar.ax.set_position([0.78, 0.2, 0.9, 0.6]) # left, bottom, width, height
    
plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/vel_fast.png', dpi=200)
#%% Difference the fast vel arrays
titles = ["Default", "Cascade", "Lowpass", "Goldstein", "Gaussian", "Gaussian + Cascade", "Gaussian + Lowpass", "Goldstein + Cascade"]
vel_fast_dict = dict(zip(titles, vel_fast_list))

vel_fast_diffs = {}
for i, arr1_name in enumerate(vel_fast_dict):
    for j, arr2_name in enumerate(vel_fast_dict):
        if i < j:
            diff_name = f"({arr1_name})-({arr2_name})"  # name of difference
            arr1 = vel_fast_dict[arr1_name]  # get array 1
            arr2 = vel_fast_dict[arr2_name]  # get array 2
            diff = arr1 - arr2  # calculate the difference
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

for i in range(len(keys)):
    print('Velocity diff', keys[i])
    print('has mean: {:.0f} mm/yr, has min diff {:.0f} mm/yr, and max diff {:.0f} mm/yr'.format(vel_fast_diffs_means[i], vel_fast_diffs_min[i],vel_fast_diffs_max[i]))
#%% Plot fast vel differences
fig, axs = plt.subplots(7, 4, figsize=(26,27))

vmax = 100
vmin = -150
    
for i in range(4):
    for j in range(7):
        k = i*7 + j
        if k < len(vel_fast_diffs_keys):
            im = axs[j,i].imshow(vel_fast_diffs_values[k], cmap='viridis', interpolation='none', vmax=vmax, vmin=vmin, extent=[x.min(), x.max(), y.min(), y.max()])
            axs[j,i].set_title(vel_fast_diffs_keys[k], fontsize=16)
            axs[j,i].text(52.95, 36.9, 'Mean diff: {:.0f} mm/yr\nMax absolute diff: {:.0f} mm/yr'.format(vel_fast_diffs_means[k], vel_fast_diffs_abs_max[k]), ha='right', fontsize = 14)
        if j < 6:
            axs[j,i].set_xticklabels([])
        if i > 0:
            axs[j,i].set_yticklabels([])
            
# Set a label at the base of the colorbar
cbar = fig.colorbar(im, ax=axs, location='right', pad=0.05, aspect=30)
cbar.ax.set_xlabel('mm/yr', fontsize=18, labelpad=10)
# Adjust colorbar width and height
cbar.ax.set_position([0.78, 0.3, 0.015, 0.4]) # left, bottom, width, bottom

plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/vel_fast_diffs.png', dpi=200)

#%%

ruw_list = ['default','cascade', 'lowpass', 'goldstein', 'gauss', 'cascade_gauss', 'lowpass_gauss', 'goldstein_cascade']
tif_end = '.vel.geo.tif'
file_dir = '/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/reunw_vels/'

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
    
vel_arr_list =[]
for tif in vel_tifs:
    imp = imageio.imread(tif)
    vel_arr_list.append(imp)

#%% Draw profiles through velocities
titles_vels = ["Default", "Cascade", "Lowpass", "Goldstein", "Gaussian", "Gaussian + Cascade", "Gaussian + Lowpass","Goldstein + Cascade"]

# Store arrays in a dictionary
vel_dict = dict(zip(titles_vels, vel_list))

# Define start and end points of profile
start_point = (50.2, 36.1)
end_point = (52, 35.25)

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
plt.legend(loc='lower right', fontsize=12)
plt.xlabel('Longitude (deg)', fontsize=13)
plt.ylabel('LoS Velocity (mm/yr)', fontsize=13)
plt.tick_params(axis='both', which='major', labelsize=14)

# Set y-axis properties to plot on the right side
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/vels_profile_poster.png', dpi=200)
#plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/vels_profile.png', dpi=200)

#%% Import n_loop_err tifs
ruw_list = ['default','cascade', 'lowpass', 'goldstein', 'gauss', 'cascade_gauss', 'lowpass_gauss', 'goldstein_cascade']
tif_end = '.n_loop_err.geo.tif'
file_dir = '/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/reunw_vels/'

#  Create list of tif names
loop_err_tifs = []
for i in range(len(ruw_list)):
    tif = file_dir + ruw_list[i] + tif_end
    loop_err_tifs.append(tif)
    
# Import loop err tifs and save them in list called loop_err_list
loop_err_list =[]
for tif in loop_err_tifs:
    imp = gdal.Open(tif)
    loop_err_list.append(imp) 

# Import loop err tiffs as arrays and save them in list called vstd_arr_list
loop_err_arr_list =[]
for tif in loop_err_tifs:
    imp = imageio.imread(tif)
    loop_err_arr_list.append(imp)

#%% Plot n_loop_err with wgs 4326 coordinates
fig, axs = plt.subplots(4,2, figsize=(8,10))
titles = ["Default","Cascade", "Lowpass", "Goldstein", "Gaussian", "Gaussian + Cascade", "Gaussian + Lowpass", "Goldstein + Cascade"]

vmax=800
vmin=0
    
plot = vstd_list[0]
# Get the geotransform
geotransform = plot.GetGeoTransform()
arr = plot.ReadAsArray()
# Define the x and y coordinates of each pixel
x = geotransform[0] + geotransform[1] * np.arange(arr.shape[1])
y = geotransform[3] + geotransform[5] * np.arange(arr.shape[0])
# Create a meshgrid of the x and y coordinates
xx, yy = np.meshgrid(x, y)

for i in range(2):
    for j in range(4):
        k = i*4 + j
        if k < len(loop_err_arr_list):
            im = axs[j,i].imshow(loop_err_arr_list[k], cmap='viridis', interpolation='none', vmax=vmax, vmin=vmin, extent=[x.min(), x.max(), y.min(), y.max()])
            axs[j,i].set_title(titles[k])
        if j < 3:
            axs[j,i].set_xticklabels([])
        if i > 0:
            axs[j,i].set_yticklabels([])

# Set a label at the base of the colorbar
cbar = fig.colorbar(im, ax=axs, location='right', pad=0.05, aspect=30)
cbar.ax.set_xlabel('n loop err', fontsize=12, labelpad=10)
# Adjust colorbar width and height
cbar.ax.set_position([0.8, 0.2, 0.9, 0.6]) # left, bottom, width, height

plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/loop_err.png', dpi=200)

#%% Plot n_loop_err; zoom
fig, axs = plt.subplots(4,2, figsize=(8, 10))
titles = ["Default","Cascade", "Lowpass", "Goldstein", "Gaussian", "Gaussian + Cascade", "Gaussian + Lowpass", "Goldstein + Cascade"]

vmax=300
vmin=0
    
plot = vstd_list[0]
# Get the geotransform
geotransform = plot.GetGeoTransform()
arr = plot.ReadAsArray()
# Define the x and y coordinates of each pixel
x = geotransform[0] + geotransform[1] * np.arange(arr.shape[1])
y = geotransform[3] + geotransform[5] * np.arange(arr.shape[0])
# Create a meshgrid of the x and y coordinates
xx, yy = np.meshgrid(x, y)

for i in range(2):
    for j in range(4):
        k = i*4 + j
        if k < len(loop_err_arr_list):
            im = axs[j,i].imshow(loop_err_arr_list[k], cmap='viridis', interpolation='none', vmax=vmax, vmin=vmin, extent=[x.min(), x.max(), y.min(), y.max()])
            axs[j,i].set_title(titles[k])
            axs[j,i].set_xlim(50.35,52)
            axs[j,i].set_ylim(35,36.1)
        if j < 3:
            axs[j,i].set_xticklabels([])
        if i > 0:
            axs[j,i].set_yticklabels([])

# Set a label at the base of the colorbar
cbar = fig.colorbar(im, ax=axs, location='right', pad=0.05, aspect=30)
cbar.ax.set_xlabel('n loop err', fontsize=12, labelpad=10)
# Adjust colorbar width and height
cbar.ax.set_position([0.8, 0.2, 0.9, 0.6]) # left, bottom, width, height


plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/loop_err_zoom.png', dpi=200)

#%% Compare processing time

time_list = ['cascade', 'lowpass', 'goldstein', 'gauss', 'cascade_gauss', 'lowpass_gauss', 'goldstein_cascade']
titles = ["Cascade", "Lowpass", "Goldstein", "Gaussian", "Gaussian + Cascade", "Gaussian + Lowpass", "Goldstein + Cascade"]

start_time_cascade = datetime.datetime(2023, 3, 28, 13, 26, 0)
start_time_lowpass = datetime.datetime(2023, 3, 28, 14, 29, 0)
start_time_goldstein = datetime.datetime(2023, 3, 28, 13, 27, 0)
start_time_cascade_gauss = datetime.datetime(2023, 3, 28, 13, 35, 0)
start_time_lowpass_gauss = datetime.datetime(2023, 3, 28, 13, 31, 0)
start_time_gauss = datetime.datetime(2023, 3, 28, 13, 32, 0)
start_time_goldstein_cascade = datetime.datetime(2023, 7, 13, 13, 50, 0)

end_time_cascade = datetime.datetime(2023, 3, 28, 19, 3, 0)
end_time_lowpass = datetime.datetime(2023, 3, 28, 20, 48, 0)
end_time_goldstein = datetime.datetime(2023, 3, 28, 18, 36, 0)
end_time_cascade_gauss = datetime.datetime(2023, 3, 28, 22, 24, 0)
end_time_lowpass_gauss = datetime.datetime(2023, 3, 28, 19, 33, 0)
end_time_gauss = datetime.datetime(2023, 3, 28, 21, 1, 0)
end_time_goldstein_cascade = datetime.datetime(2023, 7, 14, 11, 55, 0)

end_times = []
start_times = []
diff_times = []
for i in range(len(time_list)):
    end_time = globals()['end_time_' + time_list[i]]
    start_time = globals()['start_time_' + time_list[i]]
    end_times.append(end_time)
    start_times.append(start_time)
    diff_time = end_time - start_time
    diff_times.append(diff_time)

time_dict = dict(zip(titles, diff_times))
values = [value.total_seconds() / 3600 for value in time_dict.values()]

plt.figure(figsize=(16, 8))
plt.bar(titles, values)
plt.xlabel("Reunwrapping method")
plt.ylabel("Processing time (hr)")
plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/proc_time.png', dpi=200)

#%% Compare the clipped plots - double check WLS v LS
# Import clipped data
ruw_list = ['WLS.vel.geo.tif', 'WLS_n_loop_err.geo.tif', 'WLS.vstd.geo.tif', 'WLS.vel.mskd.geo.tif', 'WLS.vel_filt.geo.tif', 'WLS.vel_filt.mskd.geo.tif', 'LS.vel.geo.tif', 'LS_n_loop_err.geo.tif', 'LS.vstd.geo.tif', 'LS.vel.mskd.geo.tif', 'LS.vel_filt.geo.tif', 'LS.vel_filt.mskd.geo.tif']
file_dir = '/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/clipped/'

#  Create list of tif names
clip_tifs = []
for i in range(len(ruw_list)):
    tif = file_dir + ruw_list[i]
    clip_tifs.append(tif)
    
# Import clip tifs and save them in a list of georef datasets called clip_list
clip_list =[]
for tif in clip_tifs:
    imp = gdal.Open(tif)
    clip_list.append(imp) 

# Import clip tiffs as arrays and save them in list or arrays called vstd_arr_list
clip_arr_list =[]
for tif in clip_tifs:
    imp = imageio.imread(tif)
    clip_arr_list.append(imp)
    
#%% #%% Plot clipped results with wgs 4326 coordinates
# add in masked and filt vels too!!
fig, axs = plt.subplots(6,2, figsize=(7, 11.5))
y_titles = ["Velocities", "Loop Errors", "Vstd", "Vel Masked", "Vel Filt", "Vel Filt Masked"]
x_titles = ["Goldstein + Cascade + WLS", "Goldstein + Cascade + LS"]
cb_labels = ["mm/yr", "n loop errors", "mm/yr", "mm/yr", "mm/yr", "mm/yr", "mm/yr", "n loop errors", "mm/yr", "mm/yr", "mm/yr", "mm/yr"]

    
plot = clip_list[0]
# Get the geotransform
geotransform = plot.GetGeoTransform()
arr = plot.ReadAsArray()
# Define the x and y coordinates of each pixel
x = geotransform[0] + geotransform[1] * np.arange(arr.shape[1])
y = geotransform[3] + geotransform[5] * np.arange(arr.shape[0])
# Create a meshgrid of the x and y coordinates
xx, yy = np.meshgrid(x, y)

for i in range(2):
    for j in range(6):
        k = i*6 + j
        if k < len(clip_arr_list):
            im = axs[j,i].imshow(clip_arr_list[k], cmap='viridis', interpolation='none', extent=[x.min(), x.max(), y.min(), y.max()])
            plt.colorbar(im, ax=axs[j,i], label=cb_labels[k])
           #axs[j,i].set_title(titles[k])
        if j < 5:
            axs[j,i].set_xticklabels([])
        if i > 0:
            axs[j,i].set_yticklabels([])
        if i == 0:
            axs[j,i].set_ylabel(y_titles[j], rotation=90, va="center", labelpad=10)
        if j == 0:
            axs[j,i].set_title(x_titles[i], rotation=0, va="center", pad=20)

# Plot trace - comment out to remove
#axs[0,0].plot([end_point[0], start_point[0]], [end_point[1], start_point[1]], color='red')
#axs[0,1].plot([end_point[0], start_point[0]], [end_point[1], start_point[1]], color='red')

plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/clipped.png', dpi=200)
#plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/clipped_profile_trace.png', dpi=200)

#%% Profile across the vels and other
# Store names of vel tifs
ruw_list = ['WLS.vel.geo.tif', 'WLS.vel.mskd.geo.tif', 'WLS.vel_filt.geo.tif', 'WLS.vel_filt.mskd.geo.tif', 'LS.vel.geo.tif', 'LS.vel.mskd.geo.tif', 'LS.vel_filt.geo.tif', 'LS.vel_filt.mskd.geo.tif', 'default.vel.geo.tif', 'default.vel.mskd.geo.tif', 'default.vel.filt.geo.tif', 'default.vel.filt.mskd.geo.tif']
file_dir = '/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/clipped/'

#  Create list of tif names
clip_vel_tifs = []
for i in range(len(ruw_list)):
    tif = file_dir + ruw_list[i]
    clip_vel_tifs.append(tif)
    
# Import clip tifs and save them in a list of georef datasets called clip_list
clip_vel_list =[]
for tif in clip_vel_tifs:
    imp = gdal.Open(tif)
    clip_vel_list.append(imp) 

# Import clip tiffs as arrays and save them in list or arrays called vstd_arr_list
clip_vel_arr_list =[]
for tif in clip_vel_tifs:
    imp = imageio.imread(tif)
    clip_vel_arr_list.append(imp)
    
# Store arrays in a dictionary
clip_vel_titles = ["WLS Velocities", "WLS Vel Masked", "WLS Vel Filt", "WLS Vel Filt Masked", "LS Velocities", "LS Vel Masked", "LS Vel Filt", "LS Vel Filt Masked", "Def Velocities", "Def Vel Masked", "Def Vel Filt", "Def Vel Filt Masked"]
clip_vel_dict = dict(zip(clip_vel_titles, clip_vel_list))

# Define start and end points of profile
start_point = (50.2, 36.1)
end_point = (52, 35.25)

# Define colors for the lines
vel_greens = ['yellowgreen', 'darkolivegreen', 'lawngreen', 'darkgreen']
vel_blues = ['darkorchid', 'midnightblue', 'deepskyblue', 'mediumslateblue']
vel_pink = ['hotpink', 'mediumvioletred', 'lightpink', 'crimson']

# Create an array of indices for the profile
y, x = np.linspace(start_point[0], end_point[0], 1000), np.linspace(start_point[1], end_point[1], 100)
plt.figure(figsize=(10,6))

vel_values =[]

# Loop through arrays in the dictionary and plot a profile for each
for i, (name, ds) in enumerate(clip_vel_dict.items()):
    # Get the array values along the profile
    geotransform = ds.GetGeoTransform()
    arr = ds.ReadAsArray()
    # Create an array of indices for the profile
    x = np.linspace(start_point[0], end_point[0], 150)
    y = np.linspace(start_point[1], end_point[1], 150)
    x_px = (x - geotransform[0]) / geotransform[1]  # convert x coordinates to pixels
    y_px = (y - geotransform[3]) / geotransform[5]  # convert y coordinates to pixels
    
 #Set the linestyle based on the index
    if i < 4:
         color = vel_greens[i]
         linestyle = '--'
    elif i in range(4,8):
         color = vel_blues[i-4]
         linestyle = '-'
    else:
         color = vel_pink[i-8]
         linestyle = ':'
    
    # Get the array values along the profile
    vel_value = [arr[int(round(y)), int(round(x))] for x, y in zip(x_px, y_px)]
    vel_values.append(vel_value)
    # Plot the profile
    plt.plot(x, vel_value, linestyle=linestyle, color=color, label=name)
    
# Add legend and axis labels
plt.legend(loc='lower right')
plt.xlabel('Longitude (deg)')
plt.ylabel('LoS Velocity (mm/yr)')
plt.title('Profiles through InSAR Velocities Reunwrapped using Goldstein Filter and Cascade Method')

plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/vel_clipped_profiles.png', dpi=200)


#%% Same for n loop err and vstd

# Store names of loop err tifs
ruw_list = ['WLS_n_loop_err.geo.tif', 'LS_n_loop_err.geo.tif', 'default.n_loop_err.geo.tif']

#  Create list of tif names
clip_err_tifs = []
for i in range(len(ruw_list)):
    tif = file_dir + ruw_list[i]
    clip_err_tifs.append(tif)
    
# Import clip tifs and save them in a list of georef datasets called clip_list
clip_err_list =[]
for tif in clip_err_tifs:
    imp = gdal.Open(tif)
    clip_err_list.append(imp) 

# Import clip tiffs as arrays and save them in list or arrays called vstd_arr_list
clip_err_arr_list =[]
for tif in clip_err_tifs:
    imp = imageio.imread(tif)
    clip_err_arr_list.append(imp)
    
# Store arrays in a dictionary
clip_err_titles = ["WLS N Loop Errors", "LS N Loop Errors", 'Def N Loop Errors']
clip_err_dict = dict(zip(clip_err_titles, clip_err_list))

# Define start and end points of profile
start_point = (50.2, 36.1)
end_point = (52, 35.25)

# Define colors for the lines
greens = ['darkgreen']
blues = ['midnightblue']
pinks = ['crimson']

# Create an array of indices for the profile
y, x = np.linspace(start_point[0], end_point[0], 1000), np.linspace(start_point[1], end_point[1], 100)
plt.figure(figsize=(10,6))

loop_err_values = []

# Loop through arrays in the dictionary and plot a profile for each
for i, (name, ds) in enumerate(clip_err_dict.items()):
    # Get the array values along the profile
    geotransform = ds.GetGeoTransform()
    arr = ds.ReadAsArray()
    # Create an array of indices for the profile
    x = np.linspace(start_point[0], end_point[0], 150)
    y = np.linspace(start_point[1], end_point[1], 150)
    x_px = (x - geotransform[0]) / geotransform[1]  # convert x coordinates to pixels
    y_px = (y - geotransform[3]) / geotransform[5]  # convert y coordinates to pixels
    
# Set the linestyle based on the index
    if i < 1:
        color = greens[i]
        linestyle = '--'
    elif i in range(1,2):
        color = blues[i-1]
        linestyle = '-'
    else:
         color = pinks[i-2]
         linestyle = ':'
         
    # Get the array values along the profile
    loop_err_value = [arr[int(round(y)), int(round(x))] for x, y in zip(x_px, y_px)]
    loop_err_values.append(loop_err_value)
    # Plot the profile
    plt.plot(x, loop_err_value, linestyle=linestyle, color=color, label=name)
    
    
# Add legend and axis labels
plt.legend(loc='lower right')
plt.xlabel('Longitude (deg)')
plt.ylabel('Number of Loop Errors')
plt.title('Profiles through InSAR Velocities Loop Errors Reunwrapped using Goldstein Filter and Cascade Method')
plt.gca().invert_yaxis()

plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/n_loop_err_clipped_profiles.png', dpi=200)

#%% vstd

# Store names of vstd tifs
ruw_list = ['WLS.vstd.geo.tif', 'LS.vstd.geo.tif', 'default.vstd.geo.tif']

#  Create list of tif names
clip_vstd_tifs = []
for i in range(len(ruw_list)):
    tif = file_dir + ruw_list[i]
    clip_vstd_tifs.append(tif)
    
# Import clip tifs and save them in a list of georef datasets called clip_list
clip_vstd_list =[]
for tif in clip_vstd_tifs:
    imp = gdal.Open(tif)
    clip_vstd_list.append(imp) 

# Import clip tiffs as arrays and save them in list or arrays called vstd_arr_list
clip_vstd_arr_list =[]
for tif in clip_vstd_tifs:
    imp = imageio.imread(tif)
    clip_vstd_arr_list.append(imp)
    
# Store arrays in a dictionary
clip_vstd_titles = ["WLS Vstd", "LS Vstd", "Def Vstd"]
clip_vstd_dict = dict(zip(clip_vstd_titles, clip_vstd_list))

# Define start and end points of profile
start_point = (50.2, 36.1)
end_point = (52, 35.25)

# Define colors for the lines
greens = ['darkgreen']
blues = ['midnightblue']
pinks = ['crimson']

# Create an array of indices for the profile
y, x = np.linspace(start_point[0], end_point[0], 1000), np.linspace(start_point[1], end_point[1], 100)
plt.figure(figsize=(10,6))

vstd_values = []

# Loop through arrays in the dictionary and plot a profile for each
for i, (name, ds) in enumerate(clip_vstd_dict.items()):
    # Get the array values along the profile
    geotransform = ds.GetGeoTransform()
    arr = ds.ReadAsArray()
    # Create an array of indices for the profile
    x = np.linspace(start_point[0], end_point[0], 150)
    y = np.linspace(start_point[1], end_point[1], 150)
    x_px = (x - geotransform[0]) / geotransform[1]  # convert x coordinates to pixels
    y_px = (y - geotransform[3]) / geotransform[5]  # convert y coordinates to pixels
    
# Set the linestyle based on the index
    if i < 1:
        color = greens[i]
        linestyle = '--'
    elif i in range(1,2):
        color = blues[i-1]
        linestyle = '-'
    else:
         color = pinks[i-2]
         linestyle = ':'
    
    # Get the array values along the profile
    vstd_value = [arr[int(round(y)), int(round(x))] for x, y in zip(x_px, y_px)]
    vstd_values.append(vstd_value)
    # Plot the profile
    plt.plot(x, vstd_value, linestyle=linestyle, color=color, label=name)
    
# Add legend and axis labels
plt.legend(loc='lower right')
plt.xlabel('Longitude (deg)')
plt.ylabel('mm/yr vstd')
plt.title('Profiles through InSAR Velocities Vstd Reunwrapped using Goldstein Filter and Cascade Method')
plt.gca().invert_yaxis()

plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/vstd_clipped_profiles.png', dpi=200)

#%% All together
# Create a new figure with three subplots
fig, axs = plt.subplots(3, 1, figsize=(12, 13))

for i in range(len(vel_values)):
    if i < 4:
        color = vel_greens[i]
        linestyle = '--'
    elif i in range(4,8):
         color = vel_blues[i-4]
         linestyle = '-'
    else:
         color = vel_pink[i-8]
         linestyle = ':'
    axs[0].plot(x, vel_values[i], label=clip_vel_titles[i], color=color, linestyle=linestyle)
    axs[0].legend(loc='lower right')
    axs[0].set_xticklabels([])
    axs[0].set_ylabel('LoS Velocity (mm/yr)')
    axs[0].set_title('Profiles through InSAR Velocities Reunwrapped using Goldstein Filter and Cascade Method')
    
for i in range(len(loop_err_values)):
    if i < 1:
        color = greens[i]
        linestyle = '--'
    elif i in range(1,2):
        color = blues[i-1]
        linestyle = '-'
    else:
         color = pinks[i-2]
         linestyle = ':'
    axs[1].plot(x, loop_err_values[i], label=clip_err_titles[i], color=color, linestyle=linestyle)
    axs[1].legend(loc='lower right')
    axs[1].set_xticklabels([])
    axs[1].set_ylabel('Number of Loop Errors')
    axs[1].set_title('Profiles through InSAR Loop Errors Reunwrapped using Goldstein Filter and Cascade Method')

for i in range(len(vstd_values)):
    if i < 1:
        color = greens[i]
        linestyle = '--'
    elif i in range(1,2):
        color = blues[i-1]
        linestyle = '-'
    else:
         color = pinks[i-2]
         linestyle = ':'
    axs[2].plot(x, vstd_values[i], label=clip_vstd_titles[i], color=color, linestyle=linestyle)  
    axs[2].legend(loc='lower right')
    axs[2].set_xlabel('Longitude (deg)')
    axs[2].set_ylabel('LoS Velocity Vstd (mm/yr)')
    axs[2].set_title('Profiles through InSAR Velocities Vstd Reunwrapped using Goldstein Filter and Cascade Method')
    
axs[1].invert_yaxis()
axs[2].invert_yaxis()

plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/035D_05397_131013/results/all_clipped_profiles.png', dpi=200)
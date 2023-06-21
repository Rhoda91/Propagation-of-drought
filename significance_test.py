# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:48:31 2022

@author: roo290
"""

import pandas as pd
import pickle
import os
import math
import matplotlib.pyplot as plt
import sys
import numpy as np
import netCDF4 as nc 
from netCDF4 import Dataset
from osgeo import gdal
import rasterio
import seaborn as sns
import geopandas as gpd
import earthpy as et
import earthpy.plot as ep
from shapely.geometry import mapping
import rioxarray as rxr
import xarray as xr
import functools
from scipy.stats import percentileofscore
from statistics import NormalDist
import scipy.stats as stats
from numpy import savetxt
import itertools
import regionmask
import csv 
import matplotlib.ticker as ticker
from shapely.ops import unary_union
from shapely.ops import cascaded_union
# from mpl_toolkits.Basemap import Basemap 
gdal.UseExceptions()
 
#%%
# Path
fp_ATLAS = r'C:\Users\roo290\surfdrive (backup)\Data\SHEDs\BasinATLAS_lev6.shp' 
fp_SHEDs = r'C:\Users\roo290\surfdrive (backup)\Data\SHEDs\hydroSHEDS_lev6.shp'

data_ATLAS = gpd.read_file(fp_ATLAS,crs="epsg:4326")                            #BasinATLAS

data_SHEDs = gpd.read_file(fp_SHEDs,crs="epsg:4326")        
#%%
def loading_excel (path):
    file= pd.read_excel (path)
    #file.set_index('Unnamed: 0',inplace=True) # make the first column into the index
    #file.index.rename('Index',inplace=True) #rename the new index
    return file
#%%

geology_file = r'C:\Users\roo290\surfdrive (backup)\Data\SHEDs\geology_catch_intersect.shp'
geo_excel = loading_excel(r'C:\Users\roo290\surfdrive (backup)\Data\MSWEP_data\Past\Results\Plots\geology_catchment_intersect.xlsx')
landcover_excel = loading_excel(r'C:\Users\roo290\surfdrive (backup)\Data\MSWEP_data\Past\Results\Plots\Landcover_catchment_intersect.xlsx')
data_geo = gpd.read_file(geology_file,crs="epsg:4326")   

geo_excel['geo_maxtype'] = (geo_excel[['HISTO_1', 'HISTO_2', 'HISTO_3', 'HISTO_4', 'HISTO_5']].idxmax(axis=1)).str[6:].astype(int)
landcover_excel['land_maxtype'] = (landcover_excel[['HISTO_1', 'HISTO_2', 'HISTO_3', 'HISTO_4', 'HISTO_5','HISTO_6', 'HISTO_7', 'HISTO_8','HISTO_9']].idxmax(axis=1)).str[6:].astype(int)
#%%
#significance test

from scipy.stats import ttest_ind

v1 = geo_2['HISTO_2']
v2 = geo_1['sp-sm']

res = ttest_ind(v1, v2)

print(res)
#%%
xteristics = ['UP_AREA', 'Terrain_sl', 'Avg_GW_tab', 'UPstream_A', 'Climate_zo',
              'Global_avg', 'Sand_fract', 'Clay_fract', 'Silt_fract', 'Avg_soilwa',
              'Avg_Pop_de', 'annual_pre']
accumul=[1,2,3,4,5,6,7,8,9]
dict_clusters = {}
clus = []
signi_clus = pd.DataFrame(columns=accumul)


for x in accumul:
    clus = []
    for i in range(1,10):
        v1 = (landcover_excel.loc[landcover_excel['land_maxtype'] == i, 'sp-sm']).dropna()
        v2 = (landcover_excel.loc[landcover_excel['land_maxtype'] == x, 'sp-sm']).dropna()
      
        res = ttest_ind(v1, v2)
        print(res[1])
        clus.append(res[1])
    signi_clus[x] = clus
        
#%%   
    signi_clus[i]= res[1]

#significance test for the accumulation clusters
from scipy.stats import ttest_ind
from scipy.stats import f_oneway

def significance_test (data1, data2):
    
    # compare samples
    stat, p = ttest_ind(data1, data2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
    	print('Same distributions (fail to reject H0)')
    else:
    	print('Different distributions (reject H0)')

v1 = geo_excel.loc[geo_excel['sp-sm'] == 8, 'UP_AREA']
v2 = geo_excel.loc[geo_excel['sp-sm'] == 8, 'sp-sm']

significance_test(v1, v2)
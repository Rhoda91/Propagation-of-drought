# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:40:05 2022

@author: roo290
"""

# In[1]

#Loading the packages needed for the script

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
 

# In[2]

#Setting the work folder

os.chdir(r'U:\Rhoda\Surfdrive\Data\MSWEP_data\MSWEP_V280\Past\Daily')
working_directory=os.getcwd()
print(working_directory)

# In[3]

#Loading the datasets
#setting file paths

fp_ATLAS = r'C:\Users\roo290\surfdrive (backup)\Data\SHEDs\BasinATLAS_lev6.shp' 
fp_SHEDs = r'C:\Users\roo290\surfdrive (backup)\Data\SHEDs\hydroSHEDS_lev6.shp'

# In[4]

# Read file using gpd.read_file()

data_ATLAS = gpd.read_file(fp_ATLAS,crs="epsg:4326")                            #BasinATLAS

data_SHEDs = gpd.read_file(fp_SHEDs,crs="epsg:4326")                            #hydroSHEDs

# In[5]

# dropping the columns with sub areas less than 150 km2
#This removes catchments that are smaller than the grid size of the precipitation data 

data_SHEDs.drop(data_SHEDs.loc[data_SHEDs['SUB_AREA']<=150].index, inplace=True)
data_SHEDs['Index']=np.arange(320);
data_SHEDs.set_index('Index',inplace=True)

#%%
data_SHEDs['diff_area'] = data_SHEDs['UP_AREA'] - data_SHEDs['SUB_AREA']   
       
# In[8]

# PRECIPITATION
    #loading precipitation dataset in sets of 10 to reduce to time and space allocated for the data

data = xr.open_mfdataset('*.nc',chunks={"time":10})

data =data.sel(lon =slice (30,53.5), lat = slice (17,-5))#slice the data to the study area

#data.to_netcdf(r'C:\Users\roo290\surfdrive (backup)\Data\precip_combined.nc')               #convert to netcdf to enable easier regridding
# data = xr.open_dataset ('precip_combined.nc')
print(data)

# In[11]

#AGGREGATION TO TIMESERIES FOR PRECIPITATION

# the function aggregates the gridded data to a timeseries per catchment 
    #shed_Id is the data_sheds dataframe (the catchments keys) 
    #Index_Id is the indexes of data_sheds dataframe which is the region Id and ranges from 0-320
    # mask_data is the mask created from the catchment regions
    #array_data is the variable to be converted to timeseries


'''aggregating the gridded data to timeseries and converting it to a dataframe for each catchment
then appending it to a single dataframe with the HYBAS ID as header'''
data_monthly_Sheds_new = pd.DataFrame(index =pd.date_range(start="1980-01-01",end="2020-12-31",freq='M'))

all_sheds = []
sheds_mask_poly = regionmask.Regions(name = 'sheds_mask', numbers = list(range(0,320)), names = list(data_SHEDs.HYBAS_ID), abbrevs = list(data_SHEDs.HYBAS_ID), outlines = list(data_SHEDs.geometry.values[i] for i in range(0,320)))

print(sheds_mask_poly)

index_Id = np.arange(37,45)

mask_prec = sheds_mask_poly.mask(data.isel(time = 0 ), lon_name = 'lon', lat_name = 'lat')

lat_prec = mask_prec.lat.values
lon_prec = mask_prec.lon.values


for idx in index_Id:
    print(idx)
    if data_SHEDs['diff_area'].iloc[idx] <= 1.4:
        #print(data_SHEDs.HYBAS_ID[ID_REGION])
        
       
        sel_mask = mask_prec.where(mask_prec == idx).values
        
        id_lon = lon_prec[np.where(~np.all(np.isnan(sel_mask), axis=0))]
        id_lat = lat_prec[np.where(~np.all(np.isnan(sel_mask), axis=1))]
        
        out_sel1 = data.sel(lat = slice(id_lat[0], id_lat[-1]), lon = slice(id_lon[0], id_lon[-1])).compute().where(mask_prec == idx)

        
         
    elif (data_SHEDs.HYBAS_ID[idx] == data_SHEDs.MAIN_BAS[idx]):
        df_new = (pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_SINK']==data_SHEDs.HYBAS_ID[idx])]).reset_index())
        df_comb = df_new['Index']
        df_geo=data_SHEDs.geometry[np.array(df_comb)]
        boundary = gpd.GeoSeries(df_geo.unary_union)
        
        sheds_mask_new = regionmask.Regions(name = 'sheds_mask',numbers = list(range(0,1)), outlines = boundary.values)                   
        
        
        mask_new = sheds_mask_new.mask(data.isel(time = 0 ), lon_name = 'lon', lat_name = 'lat')
        
        
        lat_new = mask_new.lat.values
        lon_new = mask_new.lon.values
        
        sel_mask_new = mask_new.where(mask_new == 0).values    
        
        id_lon_new = lon_new[np.where(~np.all(np.isnan(sel_mask_new), axis=0))]
        id_lat_new = lat_new[np.where(~np.all(np.isnan(sel_mask_new), axis=1))]
        
        out_sel1 = data.sel(lat = slice(id_lat_new[0], id_lat_new[-1]), lon = slice(id_lon_new[0], id_lon_new[-1])).compute().where(mask_new == 0)

    else:

        alle_df = []
        df_new = (pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_DOWN']==data_SHEDs.HYBAS_ID[idx])]).reset_index())
        alle_df.append(df_new)
        df_comb = []
        df_comb.append(idx)
        while df_new.shape:
            try:
                new_catch = []
                for i in df_new['Index']:
                    df_comb.append(i)
                    if data_SHEDs['diff_area'].iloc[i] <= 1.4:
                        alle_df.append(pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_DOWN']==data_SHEDs.HYBAS_ID[i])]).reset_index())
                    else:
                        alle_df.append(pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_DOWN']==data_SHEDs.HYBAS_ID[i])]).reset_index())
                        new_catch.append(pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_DOWN']==data_SHEDs.HYBAS_ID[i])]).reset_index())
                df_new = pd.concat(new_catch)        
            except ValueError:
                break
        print('Loop ended')
        #df_new = pd.concat(alle_df)

        df_geo=data_SHEDs.geometry[np.array(df_comb)]
        boundary = gpd.GeoSeries(df_geo.unary_union)
        
        sheds_mask_new = regionmask.Regions(name = 'sheds_mask',numbers = list(range(0,1)), outlines = boundary.values)                   
        
        
        mask_new = sheds_mask_new.mask(data.isel(time = 0 ), lon_name = 'lon', lat_name = 'lat')
        
        
        lat_new = mask_new.lat.values
        lon_new = mask_new.lon.values
        
        sel_mask_new = mask_new.where(mask_new == 0).values    
        
        id_lon_new = lon_new[np.where(~np.all(np.isnan(sel_mask_new), axis=0))]
        id_lat_new = lat_new[np.where(~np.all(np.isnan(sel_mask_new), axis=1))]
        
        out_sel1 = data.sel(lat = slice(id_lat_new[0], id_lat_new[-1]), lon = slice(id_lon_new[0], id_lon_new[-1])).compute().where(mask_new == 0)

    x = out_sel1.resample(time = '1M').sum()
    
    monthly_mean=x.precipitation.mean(dim=('lon','lat'))
    
    data_monthly_Sheds_new[data_SHEDs.HYBAS_ID[idx]] = monthly_mean.to_dataframe()        
            
    data_monthly_Sheds_new.to_excel(f'U:/Rhoda/Plots_csv/new/{data_SHEDs.HYBAS_ID[idx]}.xlsx',sheet_name=f'{data_SHEDs.HYBAS_ID[idx]}',index='False', 
                engine='xlsxwriter')
    
#     all_sheds.append(df)

# data_monthly_Sheds_new = pd.concat(all_sheds, axis = 1)
#%%
# #creating a geodataframe of upstream boundaries
# boundary_df = pd.DataFrame()

# all_sheds = []
# boun_sheds ={}
# sheds_mask_poly = regionmask.Regions(name = 'sheds_mask', numbers = list(range(0,320)), names = list(data_SHEDs.HYBAS_ID), abbrevs = list(data_SHEDs.HYBAS_ID), outlines = list(data_SHEDs.geometry.values[i] for i in range(0,320)))

# print(sheds_mask_poly)

# index_Id = np.arange(0,320)

# for idx in index_Id:
#     print(idx)
#     if data_SHEDs['diff_area'].iloc[idx] <= 1.4:
#         #print(data_SHEDs.HYBAS_ID[ID_REGION])
#         all_sheds.append(gpd.GeoSeries(data_SHEDs['geometry'].iloc[idx]))
             
         
#     elif (data_SHEDs.HYBAS_ID[idx] == data_SHEDs.MAIN_BAS[idx]):
#         df_new = (pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_SINK']==data_SHEDs.HYBAS_ID[idx])]).reset_index())
#         df_comb = df_new['Index']
#         df_geo=data_SHEDs.geometry[np.array(df_comb)]
#         boundary = gpd.GeoSeries(df_geo.unary_union)
        
#         all_sheds.append(boundary)
#     else:

#         alle_df = []
#         df_new = (pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_DOWN']==data_SHEDs.HYBAS_ID[idx])]).reset_index())
#         alle_df.append(df_new)
#         df_comb = []
#         df_comb.append(idx)
#         while df_new.shape:
#             try:
#                 new_catch = []
#                 for i in df_new['Index']:
#                     df_comb.append(i)
#                     if data_SHEDs['diff_area'].iloc[i] <= 1.4:
#                         alle_df.append(pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_DOWN']==data_SHEDs.HYBAS_ID[i])]).reset_index())
#                     else:
#                         alle_df.append(pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_DOWN']==data_SHEDs.HYBAS_ID[i])]).reset_index())
#                         new_catch.append(pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_DOWN']==data_SHEDs.HYBAS_ID[i])]).reset_index())
#                 df_new = pd.concat(new_catch)        
#             except ValueError:
#                 break
#         print('Loop ended')
#         #df_new = pd.concat(alle_df)

#         df_geo=data_SHEDs.geometry[np.array(df_comb)]
#         boundary = gpd.GeoSeries(df_geo.unary_union)
        
        
#         all_sheds.append(boundary)
# #%%
# #convert to geodataframe
# df = pd.DataFrame.from_dict(boun_sheds, orient='index').reset_index()

# df.columns = ['HYBAS_ID','geometry']

# boundary_sheds_gdf = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry=data_SHEDs['geometry'])
# boundary_sheds_gdf.to_file(driver = 'ESRI Shapefile', filename= 'C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/boundary_sheds_gdf.shp')

#%%

#saving the dataframe to an excel file

data_monthly_Sheds_new.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/data_monthly_prec_21_38.xlsx', index='False', engine='xlsxwriter') 

# In[16]

# Choose standardized indices used in analysis
      #Standardized Precipitation Index                    : 'SPI'
      #Standardized Precipitation Evapotranspiration Index : 'SPEI' 
      #Standardized Soil Moisture Index                    : 'SSMI'
      #Standardized Streamflow Index                       : 'SSFI'

indices = ['SPEI', 'SPI','SSFI','SSMI']
indicesfull = ['Standardized Precipitation Index','Standardized Precipitation Evapotranspiration Index','Standardized Soil Moisture Index','Standardized Streamflow Index']  


# Indicate the time for which data is available
start = 1980
end   = 2020
years = end - start 

# Choose reference time period (years from first year of data)
      # this can be either the historic records or the full data series 
      # (depending on how droughts are defined and the goal of the study)
        
refstart = 1981   # ! First year of data series cannot be used ! 
refend   = 2020  

# In[17]

# function to accumulate hydrological data
   # with a the input data and b the accumulation time
   # -> the accumulated value coincidences with the position of the last value 
   #     used in the accumulation process.

def moving_sum(a, b) :
    
    cummuldata = np.cumsum(a, dtype=float)                 
    cummuldata[b:] = cummuldata[b:] - cummuldata[:-b]         
    cummuldata[:b - 1] = np.nan                                           
    
    return cummuldata

def moving_mean(a, b):

    cummuldata = np.cumsum(a, dtype=float)                 
    cummuldata[b:] = cummuldata[b:] - cummuldata[:-b]         
    cummuldata[:b - 1] = np.nan                                           
    
    return cummuldata/b
# In[18]

# function to find the best fitting statistical distribution
    # with a the reference time series to test the distributions 
    # and b the standardized index that is analysed
    # (possible distributions differ per hydrological data source)

def get_best_distribution(a, b):
    
    if b == 'SPEI':                     # Suggestions by Stagge et al. (2015) 
        dist_names = ['norm','genextreme', 'genlogistic', 'pearson3']                  
    elif b == 'SSMI':                   # Suggestions in Ryu et al. (2005)
        dist_names = ['norm','beta',  'pearson3','fisk']                               
    elif b == 'SPI' :                   # Suggestions by Stagge et al. (2015) 
        dist_names = ['norm','gamma', 'exponweib', 'lognorm']
    elif b == 'SSFI':                   # Suggestions by Vincent_Serrano et al. (2012)
        dist_names = ['exponweib','lognorm', 'pearson3', 'genextreme'] 
    else:
        print('problem finding distribution')

    # find fit for each optional distribution
    dist_results = []
    params = {}
    for dist_name in dist_names:                                                # Find distribution parameters        
        dist = getattr(stats, dist_name)
        param = dist.fit(a)
        params[dist_name] = param
        
        # Assess goodness-of-fit using Kolmogorov–Smirnov test
        D, p = stats.kstest(a, dist_name, args=param)                      # Applying the Kolmogorov-Smirnov test
        dist_results.append((dist_name, p))
  
    # find best fitting statistical distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))           # Select the best fitted distribution

    return best_dist, best_p, params[best_dist]



# In[19]

# function to calculate Z values for a time series of one selected month
    # with a the data series over which the index is calculated
    # and b the standardized index that is analysed 
                            
def calculate_Zvalue(a, b):
        
    # extract reference time series
    referenceseries = a[refstart-start:refend-start]     
    
    # find fitting distribution for reference sereis
    best_dist, best_p, params = get_best_distribution(referenceseries, b)  
    
    # fit full time series over best distribution
    z = np.zeros(len(a))
    dist = getattr(stats, str(best_dist))                                   
    rv = dist(*params)         
        
    # Create suitable cummulative distribution function
    # Solve issue with zero values in Gamma distribution (cfr.Stagge et al. 2015)
    if dist == 'gamma':                                                     
        nyears_zero = len(a) - np.count_nonzero(a)
        p_zero = nyears_zero / len(a)
        p_zero_mean = (nyears_zero + 1) / (2 * (len(a) + 1))           

        ppd = (a * 0 ) + p_zero_mean
        ppd[np.nonzero(a)] = p_zero+((1-p_zero)*rv.cdf(a[np.nonzero(a)]))
       
    else:
        ppd = rv.cdf(a)
    
    # Standardize the fitted cummulative distribtuion distribution 
    z = stats.norm.ppf(ppd)                                   
    
    # limit extreme, unlikely values 
    z[z>3] = 3
    z[z<-3] = -3 
            
    return z
#%%
# dist_shed = pd.DataFrame(index=['best_dist', 'best_p', 'params'], columns = data_monthly_prec.columns)
# dist_shed = dist_shed.T
# for i,shed in enumerate(dist_shed.index):
#     best_dist, best_p, params = get_best_distribution(data_monthly_sm[shed].fillna(0), 'SSMI')
#     dist_shed.loc[shed,0]=best_dist
#     dist_shed.loc[shed,1]=best_p
    #dist_shed.loc[shed,2]=params
# In[20]

# function to calculate standardized indices per month
    # with a the full accumulated data series over which the index is calculated
    # and b the standardized index that is analysed
        
def calculate_Index(a, b):

    indexvalues = a * np.nan 
                                
    for m in range(12):  
     
        # Extract monthly values
        monthlyvalues = a * np.nan
        for yr in range(int(len(a)/12)): 
            
            monthlyvalues[yr] = a[(12*yr)+m]                                 
                                
        # Retrieve index per month
        Zval = calculate_Zvalue(monthlyvalues,b)
                            
        # Reconstruct time series
        for yr in range(int(len(a)/12)):
            indexvalues[(12*yr)+m] = Zval[yr]
            
    return indexvalues  
        
# In[21]

# function calculates the indicator values for each catchment in a loop 
    #data_variable is the data series over which the index is calculated 
    #accumulation is the accumulation values over which the data is accumulated 
    #index is the type of indicator being calculated
  
def calculate_indicators_spi (data_variable,accumulation, index):
    indicator=pd.DataFrame(index =pd.date_range(start="1981-01-01",end="2020-12-31",freq='M')) #Creating empty dataframes for the indices calculation
    for df in data_variable:
        print(df)
        accumulateddata=moving_sum(data_variable[df].values, accumulation)  
        indicator[df]=calculate_Index(accumulateddata[12:], index)
    return indicator

# In[22]

#Standardized Precipitation Index (SPI)
    #calling the function to calculation SPI with accumulation periods between 1-24
    #the distribution applied here is gamma distribution

spi_1=calculate_indicators_spi(data_monthly_Sheds_new, 1, 'SPI')
spi_3=calculate_indicators_spi(data_monthly_Sheds_new, 3, 'SPI')
spi_6=calculate_indicators_spi(data_monthly_Sheds_new, 6, 'SPI')
spi_12=calculate_indicators_spi(data_monthly_Sheds_new, 12,'SPI')
spi_24=calculate_indicators_spi(data_monthly_Sheds_new, 24,'SPI')
spi_2=calculate_indicators_spi(data_monthly_Sheds_new, 2, 'SPI')
spi_4=calculate_indicators_spi(data_monthly_Sheds_new, 4, 'SPI')
spi_5=calculate_indicators_spi(data_monthly_Sheds_new, 5, 'SPI')
spi_7=calculate_indicators_spi(data_monthly_Sheds_new, 7,'SPI')
spi_8=calculate_indicators_spi(data_monthly_Sheds_new, 8, 'SPI')
spi_9=calculate_indicators_spi(data_monthly_Sheds_new, 9, 'SPI')
spi_10=calculate_indicators_spi(data_monthly_Sheds_new, 10, 'SPI')
spi_11=calculate_indicators_spi(data_monthly_Sheds_new, 11,'SPI')
spi_13=calculate_indicators_spi(data_monthly_Sheds_new, 13, 'SPI')
spi_14=calculate_indicators_spi(data_monthly_Sheds_new, 14, 'SPI')
spi_15=calculate_indicators_spi(data_monthly_Sheds_new, 15, 'SPI')
spi_17=calculate_indicators_spi(data_monthly_Sheds_new, 17,'SPI')
spi_18=calculate_indicators_spi(data_monthly_Sheds_new, 18, 'SPI')
spi_19=calculate_indicators_spi(data_monthly_Sheds_new, 19, 'SPI')
spi_20=calculate_indicators_spi(data_monthly_Sheds_new, 20, 'SPI')
spi_16=calculate_indicators_spi(data_monthly_Sheds_new, 16,'SPI')
spi_21=calculate_indicators_spi(data_monthly_Sheds_new, 21, 'SPI')
spi_22=calculate_indicators_spi(data_monthly_Sheds_new, 22, 'SPI')
spi_23=calculate_indicators_spi(data_monthly_Sheds_new, 23, 'SPI')

# In[23]

#saving files the indicator files to excel files

spi_1.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_1.xlsx', index='False', engine='xlsxwriter') 
spi_3.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_3.xlsx', index='False', engine='xlsxwriter') 
spi_6.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_6.xlsx', index='False', engine='xlsxwriter') 
spi_12.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_12.xlsx', index='False', engine='xlsxwriter') 
spi_24.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_24.xlsx', index='False', engine='xlsxwriter') 
spi_2.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_2.xlsx', index='False', engine='xlsxwriter') 
spi_4.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_4.xlsx', index='False', engine='xlsxwriter') 
spi_5.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_5.xlsx', index='False', engine='xlsxwriter') 
spi_7.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_7.xlsx', index='False', engine='xlsxwriter') 
spi_8.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_8.xlsx', index='False', engine='xlsxwriter') 
spi_9.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_9.xlsx', index='False', engine='xlsxwriter') 
spi_10.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_10.xlsx', index='False', engine='xlsxwriter') 
spi_11.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_11.xlsx', index='False', engine='xlsxwriter') 
spi_13.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_13.xlsx', index='False', engine='xlsxwriter') 
spi_14.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_14.xlsx', index='False', engine='xlsxwriter') 
spi_15.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_15.xlsx', index='False', engine='xlsxwriter') 
spi_16.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_16.xlsx', index='False', engine='xlsxwriter') 
spi_17.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_17.xlsx', index='False', engine='xlsxwriter') 
spi_18.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_18.xlsx', index='False', engine='xlsxwriter') 
spi_19.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_19.xlsx', index='False', engine='xlsxwriter') 
spi_20.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_20.xlsx', index='False', engine='xlsxwriter') 
spi_21.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_21.xlsx', index='False', engine='xlsxwriter') 
spi_22.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_22.xlsx', index='False', engine='xlsxwriter') 
spi_23.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/spi_23.xlsx', index='False', engine='xlsxwriter') 

# In[38]

#SOIL MOISTURE->Standardized Soil Moisture Index

#Loading the regridded Soil moisture dataset

data_Sm =xr.open_dataset(r'C:\Users\roo290\surfdrive (backup)\Data\MSWEP_data\Past\regridded_file_Sm.nc')
                                      
print(data_Sm)

#%%

# the function aggregates the gridded data to a timeseries per catchment 
    #shed_Id is the data_sheds dataframe (the catchments keys) 
    #Index_Id is the indexes of data_sheds dataframe which is the region Id and ranges from 0-320
    # mask_data is the mask created from the catchment regions
    #array_data is the variable to be converted to timeseries


'''aggregating the gridded data to timeseries and converting it to a dataframe for each catchment
then appending it to a single dataframe with the HYBAS ID as header'''
    
data_monthly_Sm_new = pd.DataFrame(index =pd.date_range(start="1980-01-01",end="2020-12-31",freq='M'))

all_sheds = []
sheds_mask_poly = regionmask.Regions(name = 'sheds_mask', numbers = list(range(0,320)), names = list(data_SHEDs.HYBAS_ID), abbrevs = list(data_SHEDs.HYBAS_ID), outlines = list(data_SHEDs.geometry.values[i] for i in range(0,320)))

print(sheds_mask_poly)

index_Id = np.arange(37,45)

mask_sm = sheds_mask_poly.mask(data_Sm.isel(time = 0 ), lon_name = 'lon', lat_name = 'lat')

lat_sm = mask_sm.lat.values
lon_sm = mask_sm.lon.values


for idx in index_Id:
    print(idx)
    if data_SHEDs['diff_area'].iloc[idx] <= 1.4:
        #print(data_SHEDs.HYBAS_ID[ID_REGION])
        
       
        sel_mask = mask_sm.where(mask_sm == idx).values
        
        id_lon = lon_sm[np.where(~np.all(np.isnan(sel_mask), axis=0))]
        id_lat = lat_sm[np.where(~np.all(np.isnan(sel_mask), axis=1))]
        
        out_sel1 = data_Sm.sel(lat = slice(id_lat[0], id_lat[-1]), lon = slice(id_lon[0], id_lon[-1])).compute().where(mask_sm == idx)

        
         
    elif (data_SHEDs.HYBAS_ID[idx] == data_SHEDs.MAIN_BAS[idx]):
        df_new = (pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_SINK']==data_SHEDs.HYBAS_ID[idx])]).reset_index())
        df_comb = df_new['Index']
        df_geo=data_SHEDs.geometry[np.array(df_comb)]
        boundary = gpd.GeoSeries(df_geo.unary_union)
        
        sheds_mask_new = regionmask.Regions(name = 'sheds_mask',numbers = list(range(0,1)), outlines = boundary.values)                   
        
        
        mask_new = sheds_mask_new.mask(data_Sm.isel(time = 0 ), lon_name = 'lon', lat_name = 'lat')
        
        
        lat_new = mask_new.lat.values
        lon_new = mask_new.lon.values
        
        sel_mask_new = mask_new.where(mask_new == 0).values    
        
        id_lon_new = lon_new[np.where(~np.all(np.isnan(sel_mask_new), axis=0))]
        id_lat_new = lat_new[np.where(~np.all(np.isnan(sel_mask_new), axis=1))]
        
        out_sel1 = data_Sm.sel(lat = slice(id_lat_new[0], id_lat_new[-1]), lon = slice(id_lon_new[0], id_lon_new[-1])).compute().where(mask_new == 0)

    else:

        alle_df = []
        df_new = (pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_DOWN']==data_SHEDs.HYBAS_ID[idx])]).reset_index())
        alle_df.append(df_new)
        df_comb = []
        df_comb.append(idx)
        while df_new.shape:
            try:
                new_catch = []
                for i in df_new['Index']:
                    df_comb.append(i)
                    if data_SHEDs['diff_area'].iloc[i] <= 1.4:
                        alle_df.append(pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_DOWN']==data_SHEDs.HYBAS_ID[i])]).reset_index())
                    else:
                        alle_df.append(pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_DOWN']==data_SHEDs.HYBAS_ID[i])]).reset_index())
                        new_catch.append(pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_DOWN']==data_SHEDs.HYBAS_ID[i])]).reset_index())
                df_new = pd.concat(new_catch)        
            except ValueError:
                break
        print('Loop ended')
        #df_new = pd.concat(alle_df)

        df_geo=data_SHEDs.geometry[np.array(df_comb)]
        boundary = gpd.GeoSeries(df_geo.unary_union)
        
        sheds_mask_new = regionmask.Regions(name = 'sheds_mask',numbers = list(range(0,1)), outlines = boundary.values)                   
        
        
        mask_new = sheds_mask_new.mask(data_Sm.isel(time = 0 ), lon_name = 'lon', lat_name = 'lat')
        
        
        lat_new = mask_new.lat.values
        lon_new = mask_new.lon.values
        
        sel_mask_new = mask_new.where(mask_new == 0).values    
        
        id_lon_new = lon_new[np.where(~np.all(np.isnan(sel_mask_new), axis=0))]
        id_lat_new = lat_new[np.where(~np.all(np.isnan(sel_mask_new), axis=1))]
        
        out_sel1 = data_Sm.sel(lat = slice(id_lat_new[0], id_lat_new[-1]), lon = slice(id_lon_new[0], id_lon_new[-1])).compute().where(mask_new == 0)

    x = out_sel1.resample(time = '1M').mean()
    
    monthly_mean=x.SMroot.mean(dim=('lon','lat'))
    
    data_monthly_Sm_new[data_SHEDs.HYBAS_ID[idx]] = monthly_mean.to_dataframe()        
            
    # data_monthly_Sm_new.to_excel(f'U:/Rhoda/Plots_csv/new/{data_SHEDs.HYBAS_ID[idx]}.xlsx',sheet_name=f'{data_SHEDs.HYBAS_ID[idx]}',index='False', 
    #             engine='xlsxwriter')
    
            


# In[32]

#saving the dataframe of the P ET timeseries into excel

data_monthly_Sm_new.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/data_monthly_Sm_new.xlsx', index='False', engine='xlsxwriter') 

#%%
def calculate_indicators_ssmi_ssfi (data_variable,accumulation, index):
    indicator=pd.DataFrame(index =pd.date_range(start="1981-01-01",end="2020-12-31",freq='M')) #Creating empty dataframes for the indices calculation
    for df in data_variable:
        print(df)
        data_variable[df]= data_variable[df].fillna(0)
        accumulateddata=moving_mean(data_variable[df].values, accumulation)
        indicator[df]=calculate_Index(accumulateddata[12:], index)
    return indicator


# In[47]

#Standardized Soil Moisture Index (SSMI)
    #calling the function to calculate SSMI based on the accumulation periods per catchment
    #the distribution applied is 
    
ssmi_1=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 1, 'SSMI')
ssmi_3=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 3, 'SSMI')
ssmi_6=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 6, 'SSMI')
ssmi_12=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 12,'SSMI')
ssmi_24=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 24,'SSMI')
ssmi_2=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 2, 'SSMI')
ssmi_4=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 4, 'SSMI')
ssmi_5=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 5, 'SSMI')
ssmi_7=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 7,'SSMI')
ssmi_8=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 8, 'SSMI')
ssmi_9=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 9, 'SSMI')
ssmi_10=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 10, 'SSMI')
ssmi_11=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 11,'SSMI')
ssmi_13=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 13, 'SSMI')
ssmi_14=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 14, 'SSMI')
ssmi_15=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 15, 'SSMI')
ssmi_17=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 17,'SSMI')
ssmi_18=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 18, 'SSMI')
ssmi_19=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 19, 'SSMI')
ssmi_20=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 20, 'SSMI')
ssmi_16=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 16,'SSMI')
ssmi_21=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 21, 'SSMI')
ssmi_22=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 22, 'SSMI')
ssmi_23=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 23, 'SSMI')

# In[48]

#saving files of the SSMI index into excel files

ssmi_1.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_1.xlsx', index='False', engine='xlsxwriter') 
ssmi_3.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_3.xlsx', index='False', engine='xlsxwriter') 
ssmi_6.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_6.xlsx', index='False', engine='xlsxwriter') 
ssmi_12.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_12.xlsx', index='False', engine='xlsxwriter') 
ssmi_24.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_24.xlsx', index='False', engine='xlsxwriter') 
ssmi_2.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_2.xlsx', index='False', engine='xlsxwriter') 
ssmi_4.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_4.xlsx', index='False', engine='xlsxwriter') 
ssmi_5.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_5.xlsx', index='False', engine='xlsxwriter') 
ssmi_7.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_7.xlsx', index='False', engine='xlsxwriter') 
ssmi_8.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_8.xlsx', index='False', engine='xlsxwriter') 
ssmi_9.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_9.xlsx', index='False', engine='xlsxwriter') 
ssmi_10.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_10.xlsx', index='False', engine='xlsxwriter') 
ssmi_11.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_11.xlsx', index='False', engine='xlsxwriter') 
ssmi_13.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_13.xlsx', index='False', engine='xlsxwriter') 
ssmi_14.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_14.xlsx', index='False', engine='xlsxwriter') 
ssmi_15.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_15.xlsx', index='False', engine='xlsxwriter') 
ssmi_16.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_16.xlsx', index='False', engine='xlsxwriter') 
ssmi_17.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_17.xlsx', index='False', engine='xlsxwriter') 
ssmi_18.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_18.xlsx', index='False', engine='xlsxwriter') 
ssmi_19.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_19.xlsx', index='False', engine='xlsxwriter') 
ssmi_20.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_20.xlsx', index='False', engine='xlsxwriter') 
ssmi_21.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_21.xlsx', index='False', engine='xlsxwriter') 
ssmi_22.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_22.xlsx', index='False', engine='xlsxwriter') 
ssmi_23.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssmi_23.xlsx', index='False', engine='xlsxwriter') 

# In[50]

#DISCHARGE
#Loading discharge grib files 

data_dis= xr.open_mfdataset(r'U:\Rhoda\Surfdrive\Data\Glofas\*.grib',chunks={"time":10}, engine='cfgrib')

print(data_dis)
#%%


'''aggregating the gridded data to timeseries and converting it to a dataframe for each catchment
then appending it to a single dataframe with the HYBAS ID as header'''

data_monthly_dis_new = pd.DataFrame(index =pd.date_range(start="1980-01-01",end="2020-12-31",freq='M'))

all_sheds = []
sheds_mask_poly = regionmask.Regions(name = 'sheds_mask', numbers = list(range(0,320)), names = list(data_SHEDs.HYBAS_ID), abbrevs = list(data_SHEDs.HYBAS_ID), outlines = list(data_SHEDs.geometry.values[i] for i in range(0,320)))

print(sheds_mask_poly)

index_Id = np.arange(37,45)

mask_dis = sheds_mask_poly.mask(data_dis.isel(time = 0 ), lon_name = 'longitude', lat_name = 'latitude')

lat_dis = mask_dis.latitude.values
lon_dis = mask_dis.longitude.values


for idx in index_Id:
    print(idx)
    if data_SHEDs['diff_area'].iloc[idx] <= 1.4:
        #print(data_SHEDs.HYBAS_ID[ID_REGION])
        
       
        sel_mask = mask_dis.where(mask_dis == idx).values
        
        id_lon = lon_dis[np.where(~np.all(np.isnan(sel_mask), axis=0))]
        id_lat = lat_dis[np.where(~np.all(np.isnan(sel_mask), axis=1))]
        
        out_sel1 = data_dis.sel(latitude = slice(id_lat[0], id_lat[-1]), longitude = slice(id_lon[0], id_lon[-1])).compute().where(mask_dis == idx)

        
         
    elif (data_SHEDs.HYBAS_ID[idx] == data_SHEDs.MAIN_BAS[idx]):
        df_new = (pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_SINK']==data_SHEDs.HYBAS_ID[idx])]).reset_index())
        df_comb = df_new['Index']
        df_geo=data_SHEDs.geometry[np.array(df_comb)]
        boundary = gpd.GeoSeries(df_geo.unary_union)
        
        sheds_mask_new = regionmask.Regions(name = 'sheds_mask',numbers = list(range(0,1)), outlines = boundary.values)                   
        
        
        mask_new = sheds_mask_new.mask(data_dis.isel(time = 0 ), lon_name = 'longitude', lat_name = 'latitude')
        
        
        lat_new = mask_new.latitude.values
        lon_new = mask_new.longitude.values
        
        sel_mask_new = mask_new.where(mask_new == 0).values    
        
        id_lon_new = lon_new[np.where(~np.all(np.isnan(sel_mask_new), axis=0))]
        id_lat_new = lat_new[np.where(~np.all(np.isnan(sel_mask_new), axis=1))]
        
        out_sel1 = data_dis.sel(latitude = slice(id_lat_new[0], id_lat_new[-1]), longitude = slice(id_lon_new[0], id_lon_new[-1])).compute().where(mask_new == 0)

    else:

        alle_df = []
        df_new = (pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_DOWN']==data_SHEDs.HYBAS_ID[idx])]).reset_index())
        alle_df.append(df_new)
        df_comb = []
        df_comb.append(idx)
        while df_new.shape:
            try:
                new_catch = []
                for i in df_new['Index']:
                    df_comb.append(i)
                    if data_SHEDs['diff_area'].iloc[i] <= 1.4:
                        alle_df.append(pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_DOWN']==data_SHEDs.HYBAS_ID[i])]).reset_index())
                    else:
                        alle_df.append(pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_DOWN']==data_SHEDs.HYBAS_ID[i])]).reset_index())
                        new_catch.append(pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_DOWN']==data_SHEDs.HYBAS_ID[i])]).reset_index())
                df_new = pd.concat(new_catch)        
            except ValueError:
                break
        print('Loop ended')
        #df_new = pd.concat(alle_df)

        df_geo=data_SHEDs.geometry[np.array(df_comb)]
        boundary = gpd.GeoSeries(df_geo.unary_union)
        
        sheds_mask_new = regionmask.Regions(name = 'sheds_mask',numbers = list(range(0,1)), outlines = boundary.values)                   
        
        
        mask_new = sheds_mask_new.mask(data_dis.isel(time = 0 ), lon_name = 'longitude', lat_name = 'latitude')
        
        
        lat_new = mask_new.latitude.values
        lon_new = mask_new.longitude.values
        
        sel_mask_new = mask_new.where(mask_new == 0).values    
        
        id_lon_new = lon_new[np.where(~np.all(np.isnan(sel_mask_new), axis=0))]
        id_lat_new = lat_new[np.where(~np.all(np.isnan(sel_mask_new), axis=1))]
        
        out_sel1 = data_dis.sel(latitude = slice(id_lat_new[0], id_lat_new[-1]), longitude = slice(id_lon_new[0], id_lon_new[-1])).compute().where(mask_new == 0)

    x = out_sel1.resample(time = '1M').mean()
    
    monthly_mean= (x.dis24.max(dim=('longitude','latitude'))).to_dataframe()
    
    data_monthly_dis_new[data_SHEDs.HYBAS_ID[idx]] = monthly_mean['dis24']        
            
# In[56]

#saving the dataframe of the P ET timeseries into excel

data_monthly_dis_new.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/data_monthly_dis_new.xlsx', index='False', engine='xlsxwriter') 

# In[58]

#Standardized Streamflow Index
    #calling the function to calculate SSFI based on the accumulation periods per catchment
    #the distribution applied is 
    
ssfi_1=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 1, 'SSFI')
ssfi_3=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 3, 'SSFI')
ssfi_6=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 6, 'SSFI')
ssfi_12=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 12,'SSFI')
ssfi_24=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 24,'SSFI')
ssfi_2=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 2, 'SSFI')
ssfi_4=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 4, 'SSFI')
ssfi_5=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 5, 'SSFI')
ssfi_7=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 7,'SSFI')
ssfi_8=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 8, 'SSFI')
ssfi_9=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 9, 'SSFI')
ssfi_10=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 10, 'SSFI')
ssfi_11=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 11,'SSFI')
ssfi_13=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 13, 'SSFI')
ssfi_14=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 14, 'SSFI')
ssfi_15=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 15, 'SSFI')
ssfi_17=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 17,'SSFI')
ssfi_18=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 18, 'SSFI')
ssfi_19=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 19, 'SSFI')
ssfi_20=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 20, 'SSFI')
ssfi_16=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 16,'SSFI')
ssfi_21=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 21, 'SSFI')
ssfi_22=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 22, 'SSFI')
ssfi_23=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 23, 'SSFI')

# In[59]

#saving files of the SSMI index into excel files

ssfi_1.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_1.xlsx', index='False', engine='xlsxwriter') 
ssfi_3.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_3.xlsx', index='False', engine='xlsxwriter') 
ssfi_6.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_6.xlsx', index='False', engine='xlsxwriter') 
ssfi_12.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_12.xlsx', index='False', engine='xlsxwriter') 
ssfi_24.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_24.xlsx', index='False', engine='xlsxwriter') 
ssfi_2.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_2.xlsx', index='False', engine='xlsxwriter') 
ssfi_4.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_4.xlsx', index='False', engine='xlsxwriter') 
ssfi_5.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_5.xlsx', index='False', engine='xlsxwriter') 
ssfi_7.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_7.xlsx', index='False', engine='xlsxwriter') 
ssfi_8.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_8.xlsx', index='False', engine='xlsxwriter') 
ssfi_9.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_9.xlsx', index='False', engine='xlsxwriter') 
ssfi_10.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_10.xlsx', index='False', engine='xlsxwriter') 
ssfi_11.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_11.xlsx', index='False', engine='xlsxwriter') 
ssfi_13.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_13.xlsx', index='False', engine='xlsxwriter') 
ssfi_14.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_14.xlsx', index='False', engine='xlsxwriter') 
ssfi_15.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_15.xlsx', index='False', engine='xlsxwriter') 
ssfi_16.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_16.xlsx', index='False', engine='xlsxwriter') 
ssfi_17.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_17.xlsx', index='False', engine='xlsxwriter') 
ssfi_18.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_18.xlsx', index='False', engine='xlsxwriter') 
ssfi_19.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_19.xlsx', index='False', engine='xlsxwriter') 
ssfi_20.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_20.xlsx', index='False', engine='xlsxwriter') 
ssfi_21.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_21.xlsx', index='False', engine='xlsxwriter') 
ssfi_22.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_22.xlsx', index='False', engine='xlsxwriter') 
ssfi_23.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/ssfi_23.xlsx', index='False', engine='xlsxwriter') 

#%%

#the function loads excel files into dataframe
    #path is the location of the file to be loaded

def loading_excel (path):
    file= pd.read_excel (path)
    file.set_index('Unnamed: 0',inplace=True) # make the first column into the index
    file.index.rename('Index',inplace=True) #rename the new index
    return file

#%%
#loading the dataframes for the monthly hydrometeorological variables
data_monthly_prec = loading_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/data_monthly_prec_new.xlsx')
data_monthly_sm = loading_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/data_monthly_Sm_new.xlsx')
data_monthly_dis = loading_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/data_monthly_dis_new.xlsx')

#%%
fig = plt.figure(figsize=(30,10) ) # define size of plot
l1=plt.plot(super_dictionary['ssmi','12']['1060008100'].index,super_dictionary['ssmi','1']['1060008100'],"-", color='blue', label = 'new')
l2=plt.plot(super_dictionary_old['ssmi','12'][1060008100].index,super_dictionary_old['ssmi','1'][1060008100],"-", color='black', label = '0ld')

#plt.title('SPI_1',size=20) # give plots a title
# plt.xlabel('yield (t/ha)')
plt.ylabel('SSMI-1 values')
plt.legend() # create legend
plt.show()

#%%

#Loading all the indices into one dictionary

super_dictionary = {}

def create_super_dictionary(index, accumulation):
    opened_file = pd.read_excel(f"C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/{index}_{accumulation}.xlsx")
    frame = opened_file.set_index('Unnamed: 0')
    frame.index.rename('Index',inplace=True) 
    super_dictionary.update({(index, accumulation):frame})
    
index = ['spi','ssmi', 'ssfi']
accumulation = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                '18','19', '20', '21', '22', '23', '24']

#Loading the datasets for the indices
for ind, acc in itertools.product(index, accumulation):
    create_super_dictionary(index=ind, accumulation=acc)
    
#%%

'''CORRELATION ANALYSIS USING PEARSON'''

#using Pearson correlation to correlate SPI1-24 with SSMI-1 and SSFI-1

#create dataframes for the analysis data
    
sheds_spi_ssmi ={}

catchment_keys = ['1060005140', '1060645360', '1060005150', '1060005300', '1060005310', '1060005340',
                '1060005350', '1060005420', '1060005720', '1060005730', '1060005860', '1060005870', '1060005880', '1060005890',
                '1060006520', '1060006530', '1060006850', '1060006860', '1060007000', '1060007010', '1060007060', '1060007070',
                '1060007260', '1060881460', '1060881810', '1060867590', '1060867420', '1060861400', '1060861260', '1060845290',
                '1060844930', '1060007290', '1060007300', '1060007370', '1060007390', '1060007400', '1060007450', '1060007460',
                '1060008100', '1061144990', '1061145090', '1061139500', '1061139430', '1061131830', '1061131990', '1060645590',
                '1061126260', '1061126140', '1061125990', '1061126130', '1061132520', '1061132340', '1061115400', '1061115240',
                '1061102420', '1061102430', '1061106060', '1061105930', '1061137860', '1061137960', '1061094090', '1061093940',
                '1061101140', '1061101150', '1061087260', '1061087250', '1061118600', '1061118440', '1061081050', '1061081110',
                '1061113480', '1061110670', '1061110660', '1061060620', '1061060350', '1061060500', '1061056490', '1061056400',
                '1061102090', '1061101960', '1061045600', '1061045350', '1061046600', '1061046480', '1061047930', '1061048150', 
                '1061047230', '1061047240', '1061038680', '1061038740', '1061035420', '1061035300', '1061107890', '1061107680',
                '1061045270', '1061045390', '1061020080', '1061019780', '1061014970', '1061014470', '1061014460', '1060994190',
                '1060994030', '1060986860', '1060986690', '1061020800', '1061020700', '1060992050', '1060991900', '1060960400', 
                '1060960390', '1060982500', '1060982060', '1060979820', '1060980040', '1061022130', '1061022270', '1060973660',
                '1060973650', '1061013610', '1061013460', '1060914430', '1060957700', '1060957990', '1060949620', '1060949630',
                '1060929730', '1060929860', '1060918220', '1060918230', '1060901700', '1060901710', '1060008110', '1060008130',
                '1060008150', '1060008330', '1060008340', '1060008360', '1060008370', '1060008470', '1061198260', '1061198010',
                '1061190750', '1061190870', '1061178800', '1061178790', '1061170580', '1061170490', '1060008480', '1061213170',
                '1061213080', '1061209560', '1061209710', '1061192280', '1061192210', '1061184690', '1061184680', '1060008580',
                '1060008640', '1060008650', '1060008720', '1060008740', '1060008760', '1060008880', '1061223250', '1060564930',
                '1060648030', '1060664700', '1060664840', '1060703080', '1060703250', '1060695420', '1060665640', '1060665870',
                '1060673410', '1060675820', '1060675420', '1060692800', '1060692900', '1060716900', '1060716640', '1060762830',
                '1060762820', '1060809910', '1060823900', '1060745340', '1060745130', '1060779130', '1060778870', '1060791180',
                '1060791350', '1060840030', '1060819640', '1060819440', '1060883500', '1060883520', '1060883230', '1060805400',
                '1060805160', '1060886140', '1060003990', '1060005430', '1060008160', '1060040270', '1060040390', '1060040550',
                '1060040900', '1060040920', '1060041010', '1060041020', '1060041270', '1060041280', '1060041290', '1060041440',
                '1060041840', '1060042200', '1060042330', '1060043030', '1060043650', '1060044180', '1060044210', '1060044260',
                '1060044410', '1060044540', '1060044560', '1060044920', '1060046130', '1060046470', '1060046700', '1060046740',
                '1060047460', '1060047790', '1060048420', '1060050480', '1060052010', '1060054120', '1060054840', '1060071560',
                '1060074640', '1060076070', '1060744980', '1060745140', '1060757360', '1060757370', '1060765790', '1060766070',
                '1060769630', '1060769850', '1060769960', '1060774200', '1060774210', '1060801450', '1060801660', '1060810660',
                '1060810820', '1060816780', '1060817220', '1060820440', '1060820630', '1060822950', '1060823130', '1060850580',
                '1060850740', '1060879500', '1060880020', '1060883510', '1060885810', '1060890470', '1060890730', '1060891190',
                '1060891420', '1060909260', '1060914420', '1060926500', '1060926800', '1060927240', '1060927250', '1060938670',
                '1060938680', '1060963200', '1060963480', '1060965550', '1060965780', '1060971570', '1060971580', '1060971950',
                '1060972120', '1060989140', '1060989280', '1061000360', '1061000550', '1061001550', '1061016240', '1061017250',
                '1061022540', '1061022690', '1061028490', '1061028620', '1061029030', '1061029040', '1061031870', '1061031990',
                '1061033960', '1061038590', '1061038600', '1061043750', '1061043850', '1061059910', '1061060130', '1061065950',
                '1061066110', '1061067800', '1061067870', '1061069480', '1061069620', '1061081540', '1061081660', '1061135480',
                '1061143550', '1061159550']

for x, shed in enumerate (catchment_keys):
    
    df_final = pd.DataFrame(index =pd.date_range(start="1981-01-01",end="2020-12-31",freq='M'))

    for i, (index, accumulation) in enumerate(super_dictionary.keys()): 
        
        subdictcatchment = super_dictionary[(index, accumulation)]
        
        for j, (key) in enumerate(subdictcatchment):
            
            if shed == key:
                
                if index == 'spi' or (index, accumulation)==('ssmi','1'):

                    print (key)

                    df_final['%s_%s' % (index,accumulation)]=subdictcatchment[key]

                    sheds_spi_ssmi[key] = df_final     
                    
# In[68]

                
sheds_spi_ssfi ={}    

for x, shed in enumerate (catchment_keys):
    
    df_final = pd.DataFrame(index =pd.date_range(start="1981-01-01",end="2020-12-31",freq='M'))

    for i, (index, accumulation) in enumerate(super_dictionary.keys()): 
        
        subdictcatchment = super_dictionary[(index, accumulation)]
        
        for j, (key) in enumerate(subdictcatchment):
            
            if shed == key:
                
                if index == 'spi' or (index, accumulation)==('ssfi','1'):

                    print (key)

                    df_final['%s_%s' % (index,accumulation)]=subdictcatchment[key]

                    sheds_spi_ssfi[key] = df_final    
                    

# In[69]

sheds_ssmi_ssfi ={}    

for x, shed in enumerate (catchment_keys):
    
    df_final = pd.DataFrame(index =pd.date_range(start="1981-01-01",end="2020-12-31",freq='M'))

    for i, (index, accumulation) in enumerate(super_dictionary.keys()): 
        
        subdictcatchment = super_dictionary[(index, accumulation)]
        
        subdictcatchment.columns = subdictcatchment.columns.astype(str)
        
        for j, (key) in enumerate(subdictcatchment):
            
            if shed == key:
                
                if index == 'ssmi' or (index, accumulation)==('ssfi','1'):

                    print (key)

                    df_final['%s_%s' % (index,accumulation)]=subdictcatchment[key]

                    sheds_ssmi_ssfi[key] = df_final    
                    
#%%

#saving the dictionaries for the selected data 

# save dictionaries of dataframes using pkl

file = open("C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_spi_ssfi.pkl", "wb")
pickle.dump(sheds_spi_ssfi, file)
file.close()

file = open("C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_spi_ssmi.pkl", "wb")
pickle.dump(sheds_spi_ssmi, file)
file.close()

file = open("C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_ssmi_ssfi.pkl", "wb")
pickle.dump(sheds_ssmi_ssfi, file)
file.close()


# In[70]

#load dictionaries of dataframes using pkl

file = open("C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_spi_ssmi.pkl", "rb")
sheds_spi_ssmi = pickle.load(file)


file = open("C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_spi_ssfi.pkl", "rb")
sheds_spi_ssfi = pickle.load(file)


file = open("C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_ssmi_ssfi.pkl", "rb")
sheds_ssmi_ssfi = pickle.load(file)

#%%% 
#Pearson correlation matrix for pandas dataframe

#function to calculate the correlation matrix of each catchment indicators using pearson correlation for dataframes
# dict shed is the dictionary containing the selected data for each catchment to be correlated

def corr_matrix (dict_shed,sort_col):

    corr_matrix_dict ={}   
    
    for shed in dict_shed:
        #print(shed)
        matrix = dict_shed[shed].corr()
        matrix.sort_values(by = sort_col, ascending = False, inplace=True)
        matrix.reset_index(inplace=True)
        matrix = matrix.iloc[1: , :]

        corr_matrix_dict[shed] = matrix
        
    return corr_matrix_dict

# In[72]

#Function to select the highest, lowest and their corresponding indicator names
# corr_matrix_dict is the dictionary containing the correlation matrix for each catchment

def corr_values_shed (corr_matrix_dict, last_col,indicator):
    df_corr = pd.DataFrame(index=['best_indicator', 'highest_coefficient','duration'], columns= corr_matrix_dict.keys())
    
    for i, key in enumerate (df_corr):
        if indicator =='ssmi':
            corr_matrix_dict[key]['duration'] = corr_matrix_dict[key]['index'].str[5:].astype(int)

        else:
            corr_matrix_dict[key]['duration'] = corr_matrix_dict[key]['index'].str[4:].astype(int)
        if corr_matrix_dict[key][last_col].iloc[0] >= 0.5:

            df_corr.iloc[0][i] = (corr_matrix_dict[key]['index'].iloc[0])
            df_corr.iloc[1][i] = (corr_matrix_dict[key][last_col].iloc[0]).astype(float)
            df_corr.iloc[2][i] = (corr_matrix_dict[key]['duration'].iloc[0]).astype(int)
            
        
    return df_corr.T
    
# In[73]

#main code for calling the correlation matrix and correlation values above

sheds_spi_ssmi_corrmatrix = corr_matrix(sheds_spi_ssmi,'ssmi_1')
keys_to_remove = ['1060005870','1060074640']

for key in keys_to_remove:
    sheds_spi_ssmi_corrmatrix.pop(key)
for key in sheds_spi_ssmi_corrmatrix:
    sheds_spi_ssmi_corrmatrix[key]=sheds_spi_ssmi_corrmatrix[key].sort_values(by = 'ssmi_1', ascending = False)

sheds_spi_ssfi_corrmatrix = corr_matrix(sheds_spi_ssfi,'ssfi_1')
keys_to_remove = ['1060005870','1060074640']

for key in keys_to_remove:
    sheds_spi_ssfi_corrmatrix.pop(key)
    
for key in sheds_spi_ssmi_corrmatrix:    
    sheds_spi_ssfi_corrmatrix[key]=sheds_spi_ssfi_corrmatrix[key].sort_values(by = 'ssfi_1', ascending = False)

sheds_ssmi_ssfi_corrmatrix =corr_matrix(sheds_ssmi_ssfi,'ssfi_1')
keys_to_remove = ['1060005870','1060074640']

for key in keys_to_remove:
    sheds_ssmi_ssfi_corrmatrix.pop(key)
for key in sheds_spi_ssmi_corrmatrix:    
    sheds_ssmi_ssfi_corrmatrix[key]=sheds_ssmi_ssfi_corrmatrix[key].sort_values(by = 'ssfi_1', ascending = False)    
#%%
sheds_spi_ssmi_dfcorr = corr_values_shed(sheds_spi_ssmi_corrmatrix,'ssmi_1','spi')

sheds_spi_ssfi_dfcorr = corr_values_shed(sheds_spi_ssfi_corrmatrix,'ssfi_1','spi')

sheds_ssmi_ssfi_dfcorr =corr_values_shed(sheds_ssmi_ssfi_corrmatrix,'ssfi_1','ssmi')

#%%
sheds_ssmi_ssfi_dfcorr.index = sheds_ssmi_ssfi_dfcorr.index.astype(int)
sheds_ssmi_ssfi_dfcorr.duration = sheds_ssmi_ssfi_dfcorr.duration.fillna(0).astype(int)
sheds_ssmi_ssfi_dfcorr.best_indicator = sheds_ssmi_ssfi_dfcorr.best_indicator.fillna(0).astype(str)
sheds_ssmi_ssfi_dfcorr.highest_coefficient = sheds_ssmi_ssfi_dfcorr.highest_coefficient.fillna(0).astype(float)
sheds_spi_ssmi_dfcorr.index = sheds_spi_ssmi_dfcorr.index.astype(int)
sheds_spi_ssmi_dfcorr.duration = sheds_spi_ssmi_dfcorr.duration.fillna(0).astype(int)
sheds_spi_ssmi_dfcorr.best_indicator = sheds_spi_ssmi_dfcorr.best_indicator.fillna(0).astype(str)
sheds_spi_ssmi_dfcorr.highest_coefficient = sheds_spi_ssmi_dfcorr.highest_coefficient.fillna(0).astype(float)
sheds_spi_ssfi_dfcorr.index = sheds_spi_ssfi_dfcorr.index.astype(int)
sheds_spi_ssfi_dfcorr.duration = sheds_spi_ssfi_dfcorr.duration.fillna(0).astype(int)
sheds_spi_ssfi_dfcorr.best_indicator = sheds_spi_ssfi_dfcorr.best_indicator.fillna(0).astype(str)
sheds_spi_ssfi_dfcorr.highest_coefficient = sheds_spi_ssfi_dfcorr.highest_coefficient.fillna(0).astype(float)

#%%
cols = ['best_indicator', 'highest_coefficient', 'duration']
sheds_ssmi_ssfi_dfcorr[cols] = sheds_ssmi_ssfi_dfcorr[cols].replace({'0':np.nan, 0:np.nan})
sheds_spi_ssmi_dfcorr[cols] = sheds_spi_ssmi_dfcorr[cols].replace({'0':np.nan, 0:np.nan})
sheds_spi_ssfi_dfcorr[cols] = sheds_spi_ssfi_dfcorr[cols].replace({'0':np.nan, 0:np.nan})
#%%
#saving files
# save dictionaries of dataframes using pkl

file = open("C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_spi_ssmi_corrmatrix.pkl", "wb")
pickle.dump(sheds_spi_ssmi_corrmatrix, file)
file.close()

file = open("C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_spi_ssfi_corrmatrix.pkl", "wb")
pickle.dump(sheds_spi_ssfi_corrmatrix, file)
file.close()

file = open("C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_ssmi_ssfi_corrmatrix.pkl", "wb")
pickle.dump(sheds_ssmi_ssfi_corrmatrix, file)
file.close()

#saving dataframes to excel sheets for the correlation information of the indicators

sheds_spi_ssmi_dfcorr.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_spi_ssmi_dfcorr.xlsx', index='False', engine='xlsxwriter') 

sheds_spi_ssfi_dfcorr.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_spi_ssfi_dfcorr.xlsx', index='False', engine='xlsxwriter') 

sheds_ssmi_ssfi_dfcorr.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_ssmi_ssfi_dfcorr.xlsx', index='False', engine='xlsxwriter') 

# In[76]

#load dictionaries of dataframes using pkl

file = open("C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_spi_ssmi_corrmatrix.pkl", "rb")
sheds_spi_ssmi_corrmatrix = pickle.load(file)


file = open("C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_spi_ssfi_corrmatrix.pkl", "rb")
sheds_spi_ssfi_corrmatrix = pickle.load(file)


file = open("C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_ssmi_ssfi_corrmatrix.pkl", "rb")
sheds_ssmi_ssfi_corrmatrix = pickle.load(file)


#loading excel files

sheds_spi_ssmi_dfcorr = loading_excel ('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_spi_ssmi_dfcorr.xlsx') 

sheds_spi_ssfi_dfcorr = loading_excel ('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_spi_ssfi_dfcorr.xlsx') 

sheds_ssmi_ssfi_dfcorr = loading_excel ('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_ssmi_ssfi_dfcorr.xlsx') 

#%%
#creating geodataframes
sheds_ssmi_ssfi_dfcorr[['UP_AREA','SUB_AREA','geometry']]= data_SHEDs[['UP_AREA','SUB_AREA','geometry']]

sheds_ssmi_ssfi_dfcorr = gpd.GeoDataFrame(sheds_ssmi_ssfi_dfcorr, crs="EPSG:4326", geometry=data_SHEDs['geometry'])

sheds_ssmi_ssfi_dfcorr.to_file(driver = 'ESRI Shapefile', filename= 'C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_ssmi_ssfi_corrgdf.shp')
 
#%%
sheds_spi_ssmi_dfcorr[['UP_AREA','SUB_AREA','geometry']]= data_SHEDs[['UP_AREA','SUB_AREA','geometry']]

sheds_spi_ssmi_dfcorr = gpd.GeoDataFrame(sheds_spi_ssmi_dfcorr, crs="EPSG:4326", geometry=data_SHEDs['geometry'])

sheds_spi_ssmi_dfcorr.to_file(driver = 'ESRI Shapefile', filename= 'C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_spi_ssmi_corrgdf.shp')
    

# In[75]

sheds_spi_ssfi_dfcorr[['UP_AREA','SUB_AREA','geometry']]= data_SHEDs[['UP_AREA','SUB_AREA','geometry']]

sheds_spi_ssfi_dfcorr = gpd.GeoDataFrame(sheds_spi_ssfi_dfcorr, crs="EPSG:4326", geometry=data_SHEDs['geometry'])

sheds_spi_ssfi_dfcorr.to_file(driver = 'ESRI Shapefile', filename= 'C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/sheds_spi_ssfi_corrgdf.shp')
 
#%%
#selecting the catchment characteristics
Prop_Xteristics_gdfstd = pd.DataFrame(index = sheds_spi_ssmi_dfcorr.index)

Prop_Xteristics_gdfstd[['sp-sm', 'coef_sp-sm']] = sheds_spi_ssmi_dfcorr[['duration','highest_coefficient']]
Prop_Xteristics_gdfstd[['sp-sf', 'coef_sp-sf']] = sheds_spi_ssfi_dfcorr[['duration','highest_coefficient']]
Prop_Xteristics_gdfstd[['sm-sf', 'coef_sm-sf']] = sheds_ssmi_ssfi_dfcorr[['duration','highest_coefficient']]

Prop_Xteristics_gdfstd[['UP_AREA','Terrain_slope', 'Avg_GW_table(cm)','UPstream_Avg_Elevation(m)',
                    'Climate_zones','Global_avg_aridity_index(x100)','Sand_fraction%(top5cm)', 'Clay_fraction%(top5cm)',
                    'Silt_fraction%(top5cm)','Avg_soilwater%_uyr', 'Avg_Pop_density/km2','SUB_AREA']] = data_ATLAS[['UP_AREA','slp_dg_uav', 'gwt_cm_sav',
                                                                                                                    'ele_mt_uav','clz_cl_smj',
                                                                                                                    'snd_pc_uav','cly_pc_uav','slt_pc_uav', 
                                                                                                                    'swc_pc_uyr','ari_ix_uav','ppd_pk_uav',
                                                                                                                    'SUB_AREA']]

Prop_Xteristics_gdfstd['geometry']  = data_ATLAS['geometry'] 
Prop_Xteristics_gdfstd['annual_prec'] = Prop_Xteristics_std['annual_prec'] 
Prop_Xteristics_std = Prop_Xteristics_gdfstd.copy()
Prop_Xteristics_std=Prop_Xteristics_std.drop(labels=['coef_sp-sm','coef_sp-sf','coef_sm-sf', 'geometry'], axis=1)
#saving dataframes to excel sheets for the correlation information of the indicators

Prop_Xteristics_std.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/Prop_Xteristics_std.xlsx', index='False', engine='xlsxwriter') 
Prop_Xteristics_std= loading_excel ('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/Prop_Xteristics_std.xlsx') 

#%%

Prop_Xteristics_gdfstd['geometry']  = data_ATLAS['geometry']     

Prop_Xteristics_gdfstd = gpd.GeoDataFrame(Prop_Xteristics_gdfstd, crs="EPSG:4326", geometry=data_SHEDs['geometry'])

Prop_Xteristics_gdfstd.to_file(driver = 'ESRI Shapefile', filename= 'C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/Prop_Xteristics_std.shp')
                                       
#%%
'''statistical analysis for linking drought characteristics with catchment properties'''
#creating a heatmap to represent the pearson correlation matrix of the variables
# Increase the size of the heatmap.
def heatmap_corr(df,save_name):

    # define the mask to set the values in the upper triangle to True
    #mask = np.triu(np.ones_like(df.dropna(axis=1).corr(), dtype=np.bool))
    # Store heatmap object in a variable to easily access it when you want to include more features (such as title).
    # Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
    heatmap = sns.clustermap(df.corr(), figsize=(10, 10),metric='euclidean',vmin=-1, vmax=1, annot=False,cmap='RdBu_r',cbar_pos=(1,.4, .03, .4),linewidth = 0.5, linecolor = 'black')
    # Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
    #heatmap.set_title(title, fontdict={'fontsize':12}, pad=12);
    # save heatmap as .png file
    # dpi - sets the resolution of the saved image in dots/inches
    # bbox_inches - when set to 'tight' - does not allow the labels to be cropped
    plt.savefig(f'C:/Users/roo290/surfdrive (backup)/Data/MSWEP_data/Past/Results/Plots/{save_name}.pdf', dpi=300, bbox_inches='tight')

#%%   
spi_ssmi = pd.DataFrame() 
spi_ssfi = pd.DataFrame() 
ssmi_ssfi = pd.DataFrame() 
spi_ssmi[['SPI-to-SSMI','UP_AREA', 'Terrain_slope','Avg_GW_table(cm)', 'UPstream_Avg_Elevation(m)', 'Climate_zones','Global_avg_aridity_index(x100)',
          'Sand_fraction%(top5cm)','Clay_fraction%(top5cm)', 'Silt_fraction%(top5cm)','Avg_soilwater%_uyr', 'Avg_Pop_density/km2', 'SUB_AREA',
          'annual_prec']] = Prop_Xteristics_std[['sp-sm','UP_AREA', 'Terrain_slope','Avg_GW_table(cm)', 'UPstream_Avg_Elevation(m)', 'Climate_zones','Global_avg_aridity_index(x100)',
                                                 'Sand_fraction%(top5cm)','Clay_fraction%(top5cm)', 'Silt_fraction%(top5cm)','Avg_soilwater%_uyr', 'Avg_Pop_density/km2', 'SUB_AREA',
                                                 'annual_prec']]
spi_ssfi[['SPI-to-SSI','UP_AREA', 'Terrain_slope','Avg_GW_table(cm)', 'UPstream_Avg_Elevation(m)', 'Climate_zones','Global_avg_aridity_index(x100)',
          'Sand_fraction%(top5cm)','Clay_fraction%(top5cm)', 'Silt_fraction%(top5cm)','Avg_soilwater%_uyr', 'Avg_Pop_density/km2', 'SUB_AREA',
          'annual_prec']] = Prop_Xteristics_std[['sp-sf','UP_AREA', 'Terrain_slope','Avg_GW_table(cm)', 'UPstream_Avg_Elevation(m)', 'Climate_zones','Global_avg_aridity_index(x100)',
                                                 'Sand_fraction%(top5cm)','Clay_fraction%(top5cm)', 'Silt_fraction%(top5cm)','Avg_soilwater%_uyr', 'Avg_Pop_density/km2', 'SUB_AREA',
                                                 'annual_prec']]   
ssmi_ssfi[['SSMI-to-SSI','UP_AREA', 'Terrain_slope','Avg_GW_table(cm)', 'UPstream_Avg_Elevation(m)', 'Climate_zones','Global_avg_aridity_index(x100)',
          'Sand_fraction%(top5cm)','Clay_fraction%(top5cm)', 'Silt_fraction%(top5cm)','Avg_soilwater%_uyr', 'Avg_Pop_density/km2', 'SUB_AREA',
          'annual_prec']] = Prop_Xteristics_std[['sm-sf','UP_AREA', 'Terrain_slope','Avg_GW_table(cm)', 'UPstream_Avg_Elevation(m)', 'Climate_zones','Global_avg_aridity_index(x100)',
                                                 'Sand_fraction%(top5cm)','Clay_fraction%(top5cm)', 'Silt_fraction%(top5cm)','Avg_soilwater%_uyr', 'Avg_Pop_density/km2', 'SUB_AREA',
                                                 'annual_prec']]
heatmap_corr(Prop_Xteristics_std, 'Standardized_heatmap_new')
heatmap_corr(spi_ssmi, 'SPI_to_SSMI_heatmap')
heatmap_corr(spi_ssfi, 'SPI_to_SSI_heatmap')
heatmap_corr(ssmi_ssfi, 'SSMI_to_SSI_heatmap')

#%%
#creating correlation matrix heatmap for the vairous catchments
def corr_prop(matrix_dict,last_col):
    df_matrix_above = pd.DataFrame(index= range(1,25))
    df_matrix_below = pd.DataFrame(index= range(1,25))
    for key in matrix_dict:
        df = matrix_dict[key].set_index('duration')
        if matrix_dict[key][last_col].iloc[0] >= 0.5:
            df_matrix_above[key]=df.iloc[:,-1]
        else:
            df_matrix_below[key]=df.iloc[:,-1]
    #df_matrix = df_matrix.reset_index()
    #df_matrix = np.array(df_matrix)
    #heatmap = sns.heatmap(df_matrix,vmin=-1, vmax=1, annot=False,cmap='RdBu_r',linewidth = 0.5, linecolor = 'black')
   
    return df_matrix_above, df_matrix_below
#%%
spi_ssmi_corr_above, spi_ssmi_corr_below = corr_prop(sheds_spi_ssmi_corrmatrix,'ssmi_1')
spi_ssfi_corr_above, spi_ssfi_corr_below = corr_prop(sheds_spi_ssfi_corrmatrix,'ssfi_1')
ssmi_ssfi_corr_above, ssmi_ssfi_corr_below = corr_prop(sheds_ssmi_ssfi_corrmatrix,'ssfi_1')

#%%
sns.set(font_scale=0.6) 
grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
ax = sns.heatmap(spi_ssmi_corr_above.T, ax=ax,xticklabels=1, yticklabels=False,
                 cbar_ax=cbar_ax,
                 cbar_kws={"shrink":0.5,"label": "Correlation","orientation": "horizontal"},
                 linecolor='black',vmin=0, vmax=1)

ax.set_xlabel('SPI')
ax.set_ylabel('Catchments')
plt.savefig('C:/Users/roo290/surfdrive (backup)/Data/MSWEP_data/Past/Results/Plots/spi_ssmi_corrheatmap_above.jpeg', dpi=300, bbox_inches='tight')
#%%
from scipy.stats import spearmanr
# calculate spearman's correlation

def spearman_significance(df,title,save_name):
  
    coef, p = spearmanr(df)
    matrix = df.corr()
    signi_df = pd.DataFrame(data= p, index= [matrix.index], columns=[matrix.columns])
    
    signi_df = (signi_df < 0.05).astype(float)
    
    #plotting the significance boolean in a heatmap
    plt.figure(figsize=(16, 6))
    #mask = np.triu(np.ones_like(p, dtype=np.bool))
    heatmap = sns.heatmap(signi_df, vmin=0, vmax=1, annot=True,cmap='BrBG')
    # Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
    heatmap.set_title(title, fontdict={'fontsize':12}, pad=12);
    # save heatmap as .png file
    # dpi - sets the resolution of the saved image in dots/inches
    # bbox_inches - when set to 'tight' - does not allow the labels to be cropped
    plt.savefig(f'C:/Users/roo290/surfdrive (backup)/Data/MSWEP_data/Past/Results/Plots/{save_name}.pdf', dpi=300, bbox_inches='tight')

#%%
spearman_significance(Prop_Xteristics_std, 'Link between standardized drought indices and catchment properties','Standardized_heatmap_spearman')
spearman_significance(spi_ssmi, 'SPI_to_SSMI_spearman','SPI_to_SSMI_spearman' )
spearman_significance(spi_ssfi, 'SPI_to_SSI_spearman', 'SPI_to_SSI_spearman')
spearman_significance(ssmi_ssfi, 'SSMI_to_SSI_spearman','SSMI_to_SSI_spearman')
#%%

cols = ['sp-sm', 'sp-sf', 'UP_AREA','Avg_GW_table(cm)','River_area(ha)']

titles = ['SPI-to-SSMI duration(months)','SPI-to-SSI duration(months)', 'Upstream area(km2)',
          'Average groundwater table depth(cm)', 'River area(ha)']

for t, i in enumerate (cols):
        
        #sns.set(style="whitegrid")
        ax = sns.boxplot( x='sm-sf',y=i, data=Prop_Xteristics_std,width =0.7) 
        ax.grid(False)
    
        # Calculate number of obs per group & median to position labels
        
        medians = Prop_Xteristics_std.groupby(['sm-sf'])[i].median().values
        nobs = Prop_Xteristics_std['sm-sf'].value_counts().values
        nobs = [str(x) for x in nobs.tolist()]
        nobs = ["n: " + i for i in nobs] 
        
        # Add it to the plot
        pos = range(len(nobs))
        for tick,label in zip(pos,ax.get_xticklabels()):
            ax.text(pos[tick],
                    medians[tick]+0.1,
                    nobs[tick],
                    horizontalalignment='center',
                    size='x-small',
                    color='black',
                    weight='semibold')
        # for patch in ax.artists:
        #      r, g, b, a = patch.get_facecolor()
        #      patch.set_facecolor((r, g, b, .3))
        ax.set_yscale('log')
        #ax.set_xlim([0, 16])
        #ax.set_ylim([0, 1.75e11])
        ax.set_ylabel(titles[t])
        ax.set_xlabel('SSMI-to-SSI duration(months)')
        plt.show()
        
#%%

#Average pop density, above 0.5 and below 0.5 correlation catchments

Avg_pop_spsm_above = (spi_ssmi_corr_above.iloc[1,:]).T
Avg_pop_spsm_below = spi_ssmi_corr_below
Avg_pop_spsf_above = spi_ssfi_corr_above
Avg_pop_spsf_below = spi_ssfi_corr_below
Avg_pop_smsf_above = ssmi_ssfi_corr_above 
Avg_pop_smsf_below = ssmi_ssfi_corr_below

#%%
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

significance_test(indic_sl_spi_ssmi_9, indic_sl_spi_ssmi_6)
#%%
# Define path of Inputdata-folder
os.chdir(r'C:\Users\roo290\OneDrive - Vrije Universiteit Amsterdam\Data\New_results')
workingfolder = os.getcwd()

# Name of the input file
pre_file = os.path.join(workingfolder,'data_monthly_prec_new.xlsx')
sm_file = os.path.join(workingfolder,'data_monthly_Sm_new.xlsx')
dis_file = os.path.join(workingfolder,'data_monthly_dis_new.xlsx')

#%%
#converting all the hydrometeorological variables into the same unit (m3/month)
#precipitation: original units mm/day aggregated to mm/month through resampling then converted to m3/month

for i,shed in enumerate (data_monthly_prec):
    data_monthly_prec[shed]=(data_monthly_prec[shed]/1000)*data_SHEDs.iloc[i,6]*10e6

#soil moisture conversion to m3
for i, shed in enumerate (data_monthly_sm):
    data_monthly_sm[shed]= (data_monthly_sm[shed]*1*data_SHEDs.iloc[i,6]*10e6)
    
#discharge to m3/month
for i, shed in enumerate (data_monthly_dis):
    data_monthly_dis[shed]= (data_monthly_dis[shed]*24*3600*30)
#%%
def variable_THR (name_file, var_name):
    #nx=0
    #loading the variables
    
    def loading_excel (name_file):
        file= pd.read_excel (name_file)
        file.set_index('Unnamed: 0',inplace=True) # make the first column into the index
        file.index.rename('Index',inplace=True) #rename the new index
        return file
    
    df_var = loading_excel(name_file)
    #df_var.dropna(axis=1,inplace=True)
    # Threshold level method
    
    # Define a function that finds variable threshold for certain dataframe (df), variable (var), percentile and dates
    
    def VarThres(df, shed, percentile, dates):
        perc = 1-percentile
        Varthreshold = df[shed].groupby(dates.month).quantile(perc) # get percentile for every month
        VarThreshold = np.tile(Varthreshold.values, int(len(df[shed])/12)) # repeat the montly percentiles for every year, so it can be plotted
        return VarThreshold
    
    
    percentile = 0.70
    variable=var_name
    Var_Thresh_shed = {}
    start_end_drought_year = {}
    df_var.columns = df_var.columns.astype(int)
    Drought_conditions = pd.DataFrame(index = ['no. of drought months', 'no of droughts','average duration', 'average deficit','max deviation', 'max intensity'], columns=(df_var.columns))
    for shed in df_var:
        Var_Threshold = np.zeros(df_var[shed].shape)
        Var_Thresh_shed[shed]=(df_var[shed]).to_frame().rename(columns={shed:variable})
        Var_Threshold = VarThres(df_var,shed,percentile,df_var.index) # filling in VarThres function
        # Identify drought months according to variable threshold
        dPv = np.zeros(Var_Threshold.shape)
        dPv = np.where(df_var[shed]<=Var_Threshold, 1,0) #where 1 represents drought and 0 no drought
        Var_Thresh_shed[shed]['var_thr']=Var_Threshold
        Var_Thresh_shed[shed]['dry_mnths'] = dPv
        Drought_conditions[shed].iloc[0] = np.sum(dPv) #number of months under drought conditions
        
        #  Analyse the average duration of droughts under the variable threshold level method
        dPvpropagation = dPv * 0
        
        for t in range(len(dPv)):

            if dPv[t] == 1:
                dPvpropagation[t] = dPvpropagation[t-1] + 1
                dPvpropagation[t-1] = 0
            else:
                dPvpropagation[t] = 0
                
        Drought_conditions[shed].iloc[2] = np.mean(dPvpropagation[dPvpropagation>0])
        
        # Analyse the total number of drought events under the variable threshold level method
        value_dr=1 # count a drought only if the number of consecutive drought months is higher and equal to value_dr
        Drought_conditions[shed].iloc[1] = (dPvpropagation >=value_dr).sum()#number of drought events
        # Analyse the average deficit and max intensity of droughts under the variable threshold level method
        
        dPvdeficit = dPv * 0
        dPvmax_int = dPv * 0
                                 
        inter_var = df_var[shed] - Var_Threshold #difference in [units] between data and threshold 
        
        for t in range(len(dPv)):
            if dPv[t] == 1:
                dPvdeficit=dPvdeficit.astype(float)
                dPvmax_int=dPvmax_int.astype(float)
                dPvdeficit[t] = dPvdeficit[t-1] + inter_var[t] #deficit volume
                dPvdeficit[t-1] = 0
                dPvmax_int[t] = min(inter_var[t], dPvmax_int[t]) # maximum intensity
                dPvmax_int[t-1] = 0
            else:
                dPvdeficit[t] = 0
                dPvmax_int[t] = 0
                
        if var_name == 'Soil_moisture(m3)':
            
            Drought_conditions[shed].iloc[4] = np.min((dPvdeficit))
            Drought_conditions[shed].iloc[5] = np.min((dPvmax_int))
        else:

            Drought_conditions[shed].iloc[3] = np.mean((dPvdeficit))
            Drought_conditions[shed].iloc[5] = np.min((dPvmax_int))
        
        # Print the years in which a drought in the analysed area occurred
          # if in a year there are more than TOT consecutive drought months consider that year as a drought year
        
        value_dr_con=2
         
        Var_Thresh_shed[shed]['Varpropagation']=dPvpropagation      
        
        years_drought_var=Var_Thresh_shed[shed][(Var_Thresh_shed[shed]['Varpropagation']>=value_dr_con)].index.year
        
        # Identify drought start years 
        Var_Thresh_shed[shed].insert(0, 'ID', range(1, 1 + len(Var_Thresh_shed[shed][variable])))
                    
        Var_Thresh_shed[shed]['start_drought_ID_var']=Var_Thresh_shed[shed][(Var_Thresh_shed[shed]\
                                                                             ['Varpropagation']>=value_dr_con)].ID - Var_Thresh_shed[shed]\
            [(Var_Thresh_shed[shed]['Varpropagation']>=value_dr_con)].Varpropagation +1
                    
        

        Var_Thresh_shed[shed]['start_drought_year_var'] = np.nan
        
        for index, row in Var_Thresh_shed[shed].iterrows():
            if (~np.isnan(row['start_drought_ID_var'])):
                # print('Variable')
                # print(row['start_drought_ID_var'])
                # print(Var_Thresh_shed[shed][Var_Thresh_shed[shed]['ID']==row['start_drought_ID_var']].index.year.values)
                Var_Thresh_shed[shed].loc[index, 'start_drought_year_var']=\
                    Var_Thresh_shed[shed][Var_Thresh_shed[shed]['ID']==row['start_drought_ID_var']].index.year.values
        
        start_years_drought_var = pd.DataFrame(Var_Thresh_shed[shed][Var_Thresh_shed[shed]['start_drought_year_var']>0].start_drought_year_var.values)
        start_end_drought_year[shed]=  start_years_drought_var.rename(columns={0:'Start_year'})
        start_end_drought_year[shed]['End_Year']=  years_drought_var
        
    #Plot for the last watershed in the loop
    
    fig, ax = plt.subplots(figsize=(20,5))
    ax.plot(df_var.index,Var_Threshold, "--", color='g', mew=5, label = 'Variable Threshold')
    ax.plot(df_var.index,df_var[shed],"-", color='k', mew=5,label = var_name)
    ax.fill_between(df_var.index,df_var[shed],Var_Threshold, where=(df_var[shed].values< Var_Threshold), color='r') # fill graph red when drought
    ax.set_ylabel(var_name)
    ax.set_title('Threshold Analysis')
    ax.legend() # create legend

    return Var_Thresh_shed, Drought_conditions, start_end_drought_year
#%%
#Precipitation variable threshold
Pre_Thresh_shed, pre_drought_conditions, pre_start_end_drought_yrs = variable_THR(pre_file, 'Precipitation(m3)')

#Discharge variable threshold
Dis_Thresh_shed, dis_drought_conditions, dis_start_end_drought_yrs = variable_THR(dis_file, 'Discharge(m3)')

#Soil moisture variable threshold
Sm_Thresh_shed, sm_drought_conditions, sm_start_end_drought_yrs = variable_THR(sm_file, 'Soil_moisture(m3)')

#%%
pre_drought_conditions =pre_drought_conditions.T

dis_drought_conditions = dis_drought_conditions.T

sm_drought_conditions = sm_drought_conditions.T

#%%
Prop_Xteristics_new = pd.DataFrame()
Prop_Xteristics_new[['P_meandur', 'P_meandef']] = pre_drought_conditions[['average duration','average deficit']]
Prop_Xteristics_new[['Sm_meandur', 'Sm_meandef']] = sm_drought_conditions[['average duration','max deviation']]
Prop_Xteristics_new[['Q_meandur', 'Q_meandef']] = dis_drought_conditions[['average duration','average deficit']]
#%%
Prop_Xteristics_new['P/Q_def']=Prop_Xteristics_new['P_meandef']/Prop_Xteristics_new['Q_meandef']
Prop_Xteristics_new['SM/Q_def']=Prop_Xteristics_new['Sm_meandef']/Prop_Xteristics_new['Q_meandef']
Prop_Xteristics_new['P/SM_def']=Prop_Xteristics_new['P_meandef']/Prop_Xteristics_new['Sm_meandef']

Prop_Xteristics_new['P/Q_dur']=Prop_Xteristics_new['P_meandur']/Prop_Xteristics_new['Q_meandur']
Prop_Xteristics_new['SM/Q_dur']=Prop_Xteristics_new['Sm_meandur']/Prop_Xteristics_new['Q_meandur']
Prop_Xteristics_new['P/SM_dur']=Prop_Xteristics_new['P_meandur']/Prop_Xteristics_new['Sm_meandur']
#%%
#saving Prop_xteristics as a geodataframe
Prop_Xteristics_new_gdf = Prop_Xteristics_new.copy()
Prop_Xteristics_new_gdf=Prop_Xteristics_new_gdf.astype(float)
Prop_Xteristics_new_gdf['geometry']=data_SHEDs['geometry']
Prop_Xteristics_new_gdf = gpd.GeoDataFrame(Prop_Xteristics_new_gdf, crs="EPSG:4326", geometry=data_SHEDs['geometry'])
Prop_Xteristics_new_gdf.to_file(driver = 'ESRI Shapefile', filename= 'C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/Prop_Xteristics_new_gdf.shp')

#%%

P_Q_def = Prop_Xteristics_std[['UP_AREA', 'Terrain_slope','Avg_GW_table(cm)',
                               'UPstream_Avg_Elevation(m)', 'River_area(ha)','Climate_zones',
                               'Global_avg_aridity_index(x100)','Sand_fraction%(top5cm)', 'Clay_fraction%(top5cm)',
                               'Silt_fraction%(top5cm)', 'Avg_soilwater%_uyr', 'Avg_Pop_density/km2','SUB_AREA', 'annual_prec']]
P_Q_def['P/Q_meandef']=Prop_Xteristics_new_gdf['P/Q_def']

SM_Q_def = Prop_Xteristics_std[['UP_AREA', 'Terrain_slope','Avg_GW_table(cm)',
                               'UPstream_Avg_Elevation(m)', 'River_area(ha)','Climate_zones',
                               'Global_avg_aridity_index(x100)','Sand_fraction%(top5cm)', 'Clay_fraction%(top5cm)',
                               'Silt_fraction%(top5cm)', 'Avg_soilwater%_uyr', 'Avg_Pop_density/km2','SUB_AREA', 'annual_prec']]
SM_Q_def['SM/Q_meandef']=Prop_Xteristics_new_gdf['SM/Q_def']

P_SM_def = Prop_Xteristics_std[['UP_AREA', 'Terrain_slope','Avg_GW_table(cm)',
                               'UPstream_Avg_Elevation(m)', 'River_area(ha)','Climate_zones',
                               'Global_avg_aridity_index(x100)','Sand_fraction%(top5cm)', 'Clay_fraction%(top5cm)',
                               'Silt_fraction%(top5cm)', 'Avg_soilwater%_uyr', 'Avg_Pop_density/km2','SUB_AREA', 'annual_prec']]
P_SM_def['P/SM_meandef']=Prop_Xteristics_new_gdf['P/SM_def']

P_Q_dur = Prop_Xteristics_std[['UP_AREA', 'Terrain_slope','Avg_GW_table(cm)',
                               'UPstream_Avg_Elevation(m)', 'River_area(ha)','Climate_zones',
                               'Global_avg_aridity_index(x100)','Sand_fraction%(top5cm)', 'Clay_fraction%(top5cm)',
                               'Silt_fraction%(top5cm)', 'Avg_soilwater%_uyr', 'Avg_Pop_density/km2','SUB_AREA', 'annual_prec']]
P_Q_dur['P/Q_meandur']=Prop_Xteristics_new_gdf['P/Q_dur']

P_SM_dur = Prop_Xteristics_std[['UP_AREA', 'Terrain_slope','Avg_GW_table(cm)',
                               'UPstream_Avg_Elevation(m)', 'River_area(ha)','Climate_zones',
                               'Global_avg_aridity_index(x100)','Sand_fraction%(top5cm)', 'Clay_fraction%(top5cm)',
                               'Silt_fraction%(top5cm)', 'Avg_soilwater%_uyr', 'Avg_Pop_density/km2','SUB_AREA', 'annual_prec']]
P_SM_dur['P/SM_meandur']=Prop_Xteristics_new_gdf['P/SM_dur']

SM_Q_dur = Prop_Xteristics_std[['UP_AREA', 'Terrain_slope','Avg_GW_table(cm)',
                               'UPstream_Avg_Elevation(m)', 'River_area(ha)','Climate_zones',
                               'Global_avg_aridity_index(x100)','Sand_fraction%(top5cm)', 'Clay_fraction%(top5cm)',
                               'Silt_fraction%(top5cm)', 'Avg_soilwater%_uyr', 'Avg_Pop_density/km2','SUB_AREA', 'annual_prec']]
SM_Q_dur['SM/Q_meandur']=Prop_Xteristics_new_gdf['SM/Q_dur']

Thresh_indices = Prop_Xteristics_std[['UP_AREA', 'Terrain_slope','Avg_GW_table(cm)',
                               'UPstream_Avg_Elevation(m)', 'River_area(ha)','Climate_zones',
                               'Global_avg_aridity_index(x100)','Sand_fraction%(top5cm)', 'Clay_fraction%(top5cm)',
                               'Silt_fraction%(top5cm)', 'Avg_soilwater%_uyr', 'Avg_Pop_density/km2','SUB_AREA', 'annual_prec']]

Thresh_indices[['P/Q_meandef', 'SM/Q_meandef', 'P/SM_meandef',
                'P/Q_meandur','SM/Q_meandur', 'P/SM_meandur']] = Prop_Xteristics_new_gdf[['P/Q_def', 'SM/Q_def', 'P/SM_def', 'P/Q_dur','SM/Q_dur', 'P/SM_dur']]
#%%
heatmap_corr(Thresh_indices, 'Threshold_indices_heatmap')
heatmap_corr(P_Q_def, 'P_Q_def_heatmap')
heatmap_corr(SM_Q_def, 'SM_Q_def_heatmap')
heatmap_corr(P_SM_def, 'P_SM_def_heatmap')
heatmap_corr(P_Q_dur, 'P_Q_dur_heatmap')
heatmap_corr(P_SM_dur, 'P_SM_dur_heatmap')
heatmap_corr(SM_Q_dur, 'SM_Q_dur_heatmap')

#%%
spearman_significance(Thresh_indices, 'Threshold_indices_spearman','Threshold_indices_spearman' )
spearman_significance(P_Q_def, 'P_Q_def_spearman','P_Q_def_spearman' )
spearman_significance(SM_Q_def, 'SM_Q_def_spearman', 'SM_Q_def_spearman')
spearman_significance(P_SM_def, 'P_SM_def_spearman','P_SM_def_spearman')
spearman_significance(P_Q_dur, 'P_Q_dur_spearman','P_Q_def_spearman' )
spearman_significance(P_SM_dur, 'P_SM_dur_spearman', 'SM_Q_def_spearman')
spearman_significance(SM_Q_dur, 'SM_Q_dur_spearman','P_SM_def_spearman')
#%%
#saving to excel
P_Q_def.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/P_Q_def.xlsx', index='False', engine='xlsxwriter') 
Thresh_indices.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/Thresh_indices.xlsx', index='False', engine='xlsxwriter') 
SM_Q_def.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/SM_Q_def.xlsx', index='False', engine='xlsxwriter') 
P_SM_def.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/P_SM_def.xlsx', index='False', engine='xlsxwriter') 
P_Q_dur.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/P_Q_dur.xlsx', index='False', engine='xlsxwriter') 
P_SM_dur.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/P_SM_dur.xlsx', index='False', engine='xlsxwriter') 
SM_Q_dur.to_excel('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results/SM_Q_dur.xlsx', index='False', engine='xlsxwriter') 

#%%
#Scatter plots for correlation values against accumulation periods
#This helps to cluster the catchments into different groups
#a is the geodataframe data

cols = ['Avg_GW_table(cm)','UPstream_Avg_Elevation(m)', 'River_area(ha)',
        'Global_avg_aridity_index(x100)','Sand_fraction%(top5cm)',
        'Silt_fraction%(top5cm)','Avg_soilwater%_uyr', 'Avg_Pop_density/km2','annual_prec']
    
titles =['Average groundwater table depth(cm)','Average upstream elevation(m)', 'River area(ha)',
         'Global average aridity index(x100)','Percent sand fraction(%)',
         'Percent silt fraction(%)','Average annual soilwater content(%)', 'Average Population density/km2', 
          'Mean annual precipitation(mm/yr)']
for t, i in enumerate (cols):
    
    fig, ax = plt.subplots()
    ax.grid(True)
    # Be sure to only pick integer tick locations.
    for axis in [ax.xaxis, ax.yaxis]:
        
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    # y= np.array(Prop_Xteristics_All['P/SM_meandef'])
    # x= np.array(Prop_Xteristics_All[i])
    #ax.set(xscale="log", yscale="log")
    sns.set(font_scale=1) 
    ax=sns.regplot(x=i, y='P/SM_meandur', marker="+", data=P_SM_dur, ci=None) 

    #ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.set_xlim([0, 10e2])
    #ax.set_ylim([0, 20])
    ax.set_ylabel('P/SM_meandur', fontsize=10)
    ax.set_xlabel(titles[t], fontsize=10)
    #ax.set_title(i)
    
    fig.tight_layout()
    
    #plt.savefig(f'C:/Users/roo290/surfdrive (backup)/Data/MSWEP_data/Past/Results/Plots/P_SMdur_{i}.jpg')
    plt.show()
    
#%%
#significance test for the catchments 
#in the clusters which are based on the indicators for each of the key variables
#spi_ssfi -> annual_prec
#a is the dataframe containing all the data
#b is the index to consider
#col the variables 

from scipy.stats import ttest_ind
from scipy.stats import f_oneway

def significance_test (a,b):
    variables = a.columns
    signi_var = {}
    for col in variables:
        
      f_oneway((a[col].loc[a[b]==1]).values, (a[col].loc[a[b]==2]).values,
               (a[col].loc[a[b]==3]).values, (a[col].loc[a[b]==4]).values,
               (a[col].loc[a[b]==5]).values,(a[col].loc[a[b]==6]).values,
               (a[col].loc[a[b]==7]).values,(a[col].loc[a[b]==8]).values,
               (a[col].loc[a[b]==9]).values,(a[col].loc[a[b]==10]).values)
      signi_var[col]=stats,p 
      return signi_var

spi_ssfi_signi = significance_test(spi_ssfi, 'SPI-to-SSI')

#%%
(Prop_Xteristics_std.loc[Prop_Xteristics_std['Avg_soilwater%_uyr'] <= 15 , 'sp-sm']).mean()

(Prop_Xteristics_std.loc[Prop_Xteristics_std['Avg_soilwater%_uyr'] <= 15 , 'sp-sf']).mean()

(Prop_Xteristics_std.loc[(Prop_Xteristics_std['Avg_soilwater%_uyr'] > 15) & (Prop_Xteristics_std['Avg_soilwater%_uyr'] <= 25) , 'sp-sm']).mean()

(Prop_Xteristics_std.loc[(Prop_Xteristics_std['Avg_soilwater%_uyr'] > 15) & (Prop_Xteristics_std['Avg_soilwater%_uyr'] <= 25) , 'sp-sf']).mean()


(Prop_Xteristics_std.loc[(Prop_Xteristics_std['Avg_soilwater%_uyr'] > 25) & (Prop_Xteristics_std['Avg_soilwater%_uyr'] <= 37) , 'sp-sm']).mean()


(Prop_Xteristics_std.loc[(Prop_Xteristics_std['Avg_soilwater%_uyr'] > 25) & (Prop_Xteristics_std['Avg_soilwater%_uyr'] <= 37) , 'sp-sf']).mean()


(Prop_Xteristics_std.loc[(Prop_Xteristics_std['Avg_soilwater%_uyr'] > 37) & (Prop_Xteristics_std['Avg_soilwater%_uyr'] <= 57) , 'sp-sm']).mean()


(Prop_Xteristics_std.loc[(Prop_Xteristics_std['Avg_soilwater%_uyr'] > 37) & (Prop_Xteristics_std['Avg_soilwater%_uyr'] <= 57) , 'sp-sf']).mean()


(Prop_Xteristics_std.loc[(Prop_Xteristics_std['Avg_soilwater%_uyr'] > 57) & (Prop_Xteristics_std['Avg_soilwater%_uyr'] <= 112) , 'sp-sm']).mean()


(Prop_Xteristics_std.loc[(Prop_Xteristics_std['Avg_soilwater%_uyr'] > 57) & (Prop_Xteristics_std['Avg_soilwater%_uyr'] <= 112) , 'sp-sf']).mean()

#%%
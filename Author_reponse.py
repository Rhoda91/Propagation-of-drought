# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 18:03:06 2023

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
# from mpl_toolkits.Basemap import Basemap 
gdal.UseExceptions()

#%%
#Setting the work folder

os.chdir(r'C:\Users\roo290\OneDrive - Vrije Universiteit Amsterdam\Data\New_results_PPR1')
working_directory=os.getcwd()
print(working_directory)
 #%%
data = xr.open_dataset(r'C:\Users\roo290\OneDrive - Vrije Universiteit Amsterdam\Data\ERA5\wetransfer_era-5-precip-data_2023-02-08_1440\era5_tp_1959-2021_1_12_monthly_0.25deg.nc')
data = data.sel(time = slice("1980-01-01","2020-12-01"))
print(data)

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
       
#%%
'''aggregating the gridded data to timeseries and converting it to a dataframe for each catchment
then appending it to a single dataframe with the HYBAS ID as header'''
data_monthly_Sheds_ERA5 = pd.DataFrame(index =pd.date_range(start="1980-01-01",end="2020-12-31",freq='M'))

all_sheds = []
sheds_mask_poly = regionmask.Regions(name = 'sheds_mask', numbers = list(range(0,320)), names = list(data_SHEDs.HYBAS_ID), abbrevs = list(data_SHEDs.HYBAS_ID), outlines = list(data_SHEDs.geometry.values[i] for i in range(0,320)))

print(sheds_mask_poly)

index_Id = np.arange(35,40)

mask_prec = sheds_mask_poly.mask(data.isel(time = 0 ), lon_name = 'longitude', lat_name = 'latitude')

lat_prec = mask_prec.latitude.values
lon_prec = mask_prec.longitude.values


for idx in index_Id:
    print(idx)
    if data_SHEDs['diff_area'].iloc[idx] <= 1.4:
        #print(data_SHEDs.HYBAS_ID[ID_REGION])
        
       
        sel_mask = mask_prec.where(mask_prec == idx).values
        
        id_lon = lon_prec[np.where(~np.all(np.isnan(sel_mask), axis=0))]
        id_lat = lat_prec[np.where(~np.all(np.isnan(sel_mask), axis=1))]
        
        out_sel1 = data.sel(latitude = slice(id_lat[0], id_lat[-1]), longitude = slice(id_lon[0], id_lon[-1])).compute().where(mask_prec == idx)

        
         
    elif (data_SHEDs.HYBAS_ID[idx] == data_SHEDs.MAIN_BAS[idx]):
        df_new = (pd.DataFrame(data_SHEDs['HYBAS_ID'].iloc[np.where(data_SHEDs['NEXT_SINK']==data_SHEDs.HYBAS_ID[idx])]).reset_index())
        df_comb = df_new['Index']
        df_geo=data_SHEDs.geometry[np.array(df_comb)]
        boundary = gpd.GeoSeries(df_geo.unary_union)
        
        sheds_mask_new = regionmask.Regions(name = 'sheds_mask',numbers = list(range(0,1)), outlines = boundary.values)                   
        
        
        mask_new = sheds_mask_new.mask(data.isel(time = 0 ), lon_name = 'longitude', lat_name = 'latitude')
        
        
        lat_new = mask_new.latitude.values
        lon_new = mask_new.longitude.values
        
        sel_mask_new = mask_new.where(mask_new == 0).values    
        
        id_lon_new = lon_new[np.where(~np.all(np.isnan(sel_mask_new), axis=0))]
        id_lat_new = lat_new[np.where(~np.all(np.isnan(sel_mask_new), axis=1))]
        
        out_sel1 = data.sel(latitude = slice(id_lat_new[0], id_lat_new[-1]), longitude = slice(id_lon_new[0], id_lon_new[-1])).compute().where(mask_new == 0)

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
        
        
        mask_new = sheds_mask_new.mask(data.isel(time = 0 ), lon_name = 'longitude', lat_name = 'latitude')
        
        
        lat_new = mask_new.latitude.values
        lon_new = mask_new.longitude.values
        
        sel_mask_new = mask_new.where(mask_new == 0).values    
        
        id_lon_new = lon_new[np.where(~np.all(np.isnan(sel_mask_new), axis=0))]
        id_lat_new = lat_new[np.where(~np.all(np.isnan(sel_mask_new), axis=1))]
        
        out_sel1 = data.sel(latitude = slice(id_lat_new[0], id_lat_new[-1]), longitude = slice(id_lon_new[0], id_lon_new[-1])).compute().where(mask_new == 0)

    #x = out_sel1.resample(time = '1M').sum()
    
    monthly_mean=out_sel1.tp.mean(dim=('longitude','latitude'))
    
    data_monthly_Sheds_ERA5[data_SHEDs.HYBAS_ID[idx]] = monthly_mean.to_dataframe()        
            
#%%
MSWEP=pd.DataFrame((data_monthly_Sheds_new[1060007450]), columns=([1060007450]))
#%%
def loading_excel (path):
    file= pd.read_excel (path)
    file.set_index('Unnamed: 0',inplace=True) # make the first column into the index
    file.index.rename('Index',inplace=True) #rename the new index
    return file

#%%
#loading the dataframes for the monthly hydrometeorological variables

data_monthly_dis_new = loading_excel('data_monthly_dis_new.xlsx')
data_monthly_Sheds_new = loading_excel('data_monthly_prec_new.xlsx')
data_monthly_Sm_new = loading_excel('data_monthly_Sm_new.xlsx')

#%%
#plotting ERA5 against MSWEP precipitation

fig,ax = plt.subplots(figsize=(20,10))
    # make a plot
l1=plt.plot(MSWEP.index,
            MSWEP[1060007450],
            color="red", 
            marker="o", label = "MSWEP")
l2=plt.plot(data_monthly_Sheds_ERA5.index,
            data_monthly_Sheds_ERA5[1060007450],
            color="blue",marker="o", label="ERA5")
# set x-axis label
plt.xlabel("Year", fontsize = 14)

# set y-axis label
plt.ylabel("precipitation [m/month]", color="red",fontsize=14)

# twin object for two different y-axis on the sample plot
#ax2=ax.twinx()

# make a plot with different y-axis using second axis object
#ax2.plot(data_yield_Admin['Year'][31:].astype(str), data_yield_Admin[county][31:],color="blue",marker="o")
#ax2.set_ylabel("modelled_yield",color="blue",fontsize=14)
#plt.title(f'{county} ',size=20) # give plots a title
plt.legend() # create legend
#plt.setp(ax2.get_xticklabels(), rotation=45)
plt.show()
# save the plot as a file
# fig.savefig(f'C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/plots/yield_obs_mod/{county}.jpg',
#             format='jpeg',
#             dpi=100,
#             bbox_inches='tight')

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
dist_shed_sm = pd.DataFrame(index=['best_dist', 'best_p', 'params'], columns = catchments)
dist_shed_sm = dist_shed_sm.T
for i,shed in enumerate(dist_shed_sm.index):
    best_dist, best_p, params = get_best_distribution(data_monthly_Sm_new[shed].fillna(0), 'SSMI')
    dist_shed_sm.loc[shed,'best_dist']=best_dist
    dist_shed_sm.loc[shed,'best_p']=best_p
    dist_shed_sm.loc[shed,'params']=params
    
dist_shed_dis = pd.DataFrame(index=['best_dist', 'best_p', 'params'], columns = catchments)
dist_shed_dis = dist_shed_dis.T
for i,shed in enumerate(dist_shed_dis.index):
    best_dist, best_p, params = get_best_distribution(data_monthly_dis_new[shed].fillna(0), 'SSFI')
    dist_shed_dis.loc[shed,'best_dist']=best_dist
    dist_shed_dis.loc[shed,'best_p']=best_p
    dist_shed_dis.loc[shed,'params']=params

dist_shed_prec = pd.DataFrame(index=['best_dist', 'best_p', 'params'], columns = catchments)
dist_shed_prec = dist_shed_prec.T
for i,shed in enumerate(dist_shed_prec.index):
    best_dist, best_p, params = get_best_distribution(data_monthly_Sheds_new[shed].fillna(0), 'SPI')
    dist_shed_prec.loc[shed,'best_dist']=best_dist
    dist_shed_prec.loc[shed,'best_p']=best_p
    dist_shed_prec.loc[shed,'params']=params    
    
    
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
    for df in catchments:
        print(df)
        accumulateddata=moving_sum(data_variable[df].values, accumulation)  
        indicator[df]=calculate_Index(accumulateddata[12:], index)
    return indicator

#%%
def calculate_indicators_ssmi_ssfi (data_variable,accumulation, index):
    indicator=pd.DataFrame(index =pd.date_range(start="1981-01-01",end="2020-12-31",freq='M')) #Creating empty dataframes for the indices calculation
    for df in catchments:
        print(df)
        data_variable[df]= data_variable[df].fillna(0)
        accumulateddata=moving_mean(data_variable[df].values, accumulation)
        indicator[df]=calculate_Index(accumulateddata[12:], index)
    return indicator
#%%

 # find fitting distribution for reference sereis
best_dist, best_p, params = get_best_distribution(data_monthly_Sheds_new[1060716640], 'SPI')  

# fit full time series over best distribution
z = np.zeros(len(data_monthly_Sheds_new[1060716640]))
dist = getattr(stats, str(best_dist)   
#params = dist.fit(data_monthly_Sheds_new[1060716640])                                
rv = dist(*params)  
    
# Create suitable cummulative distribution function
# Solve issue with zero values in Gamma distribution (cfr.Stagge et al. 2015)
if dist == 'gamma':                                                     
    nyears_zero = len(data_monthly_Sheds_new[1060716640]) - np.count_nonzero(data_monthly_Sheds_new[1060716640])
    p_zero = nyears_zero / len(data_monthly_Sheds_new[1060716640])
    p_zero_mean = (nyears_zero + 1) / (2 * (len(data_monthly_Sheds_new[1060716640]) + 1))           

    ppd_gamma = (data_monthly_Sheds_new[1060716640] * 0 ) + p_zero_mean
    ppd_gamma[np.nonzero(data_monthly_Sheds_new[1060716640])] = p_zero+((1-p_zero)*rv.cdf(data_monthly_Sheds_new[1060716640][np.nonzero(data_monthly_Sheds_new[1060716640])]))
   
#else:
    #ppd_lognorm= rv.cdf(data_monthly_Sheds_new[1060716640])
     
#%%
param_dict = {}
best_df_m = pd.DataFrame(index=[['best_dist','best_p']], columns =data_monthly_Sheds_new.columns )
for df in data_monthly_Sheds_new:
    
    best_dist, best_p, params = get_best_distribution(data_monthly_Sheds_new[df], 'SPI')
    best_df_m[df].iloc[0]=best_dist
    best_df_m[df].iloc[1]=best_p

best_df_m=best_df_m.T

#%%
worst_data_SPI = pd.DataFrame(data_monthly_Sheds_new[1060716640])

worst_data_SPI['rank'] = worst_data_SPI[1060716640].rank(axis=0)
worst_data_SPI['freq']=(worst_data_SPI['rank']-0.5)/len(worst_data_SPI)
worst_data_SPI['ppd_norm']=ppd
worst_data_SPI['ppd_gamma']=ppd_gamma
worst_data_SPI['ppd_lognorm']=ppd_lognorm
worst_data_SPI['ppd_exponweib']=ppd_exponweib

#%%
fig = plt.figure(figsize=(10,10) )
l1=plt.plot(np.sort(ppd),np.sort(worst_data_SPI[1060716640]), label = 'norm') 
l2=plt.plot(np.sort(worst_data_SPI['freq']),np.sort(worst_data_SPI[1060716640]), label= 'empirical_prob') 
l3=plt.plot(np.sort(ppd_exponweib),np.sort(worst_data_SPI[1060716640]), label = 'exponweib') 
l4=plt.plot(np.sort(ppd_lognorm),np.sort(worst_data_SPI[1060716640]), label = 'lognorm') 
l5=plt.plot(np.sort(ppd_gamma),np.sort(worst_data_SPI[1060716640]), label = 'gamma')
plt.ylabel('cumulative probability',size=15)
plt.xlabel('accumulated precipitation (mm)',size=15)
plt.title('Cumulative distributions for SPI-1 Vs Empirical distribution',size=25)
plt.legend()
plt.show()
#%%
#calculating the spi, ssmi and ssi values under different distributions

catchments = [1061020080, 1060692900, 1060861400, 1060006850]

spi_6_gamma=calculate_indicators_spi(data_monthly_Sheds_new, 6, 'SPI')
spi_6_norm=calculate_indicators_spi(data_monthly_Sheds_new, 6, 'SPI')
spi_6_expo=calculate_indicators_spi(data_monthly_Sheds_new, 6, 'SPI')
spi_6_logn=calculate_indicators_spi(data_monthly_Sheds_new, 6, 'SPI')

ssmi_1_norm=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 1, 'SSMI')
ssmi_1_beta=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 1, 'SSMI')
ssmi_1_pear=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 1, 'SSMI')
ssmi_1_fisk=calculate_indicators_ssmi_ssfi(data_monthly_Sm_new, 1, 'SSMI')

ssfi_1_expo=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 1, 'SSFI')
ssfi_1_logn=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 1, 'SSFI')
ssfi_1_gen=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 1, 'SSFI')
ssfi_1_pear=calculate_indicators_ssmi_ssfi(data_monthly_dis_new, 1, 'SSFI')


#%%
#Plotting the indices values for each distribution

fig = plt.subplots(4,1,figsize=(40,40),sharex=True)

plt.subplot(4, 1, 1)
plt.title('Comparison of different distribution fittings',size=35)

plt.plot(spi_6_gamma.index,spi_6_gamma[1061020080],"-", color='k', mew=5, linewidth=2, label="gamma")
plt.plot(spi_6_norm.index,spi_6_norm[1061020080],"-", color='red', mew=5, linewidth=2, label='norm')
plt.plot(spi_6_expo.index,spi_6_expo[1061020080],"-", color='green', mew=5, linewidth=2, label="exponweib")
plt.plot(spi_6_logn.index,spi_6_logn[1061020080],"-", color='blue', mew=5, linewidth=2, label="lognorm")
plt.ylabel('SPI-6 values (Ethiopian highlands)',size =25)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.legend(fontsize=20).texts[2].set_fontweight('bold')# make specific label bold


plt.subplot(4, 1, 2)

plt.plot(spi_6_gamma.index,spi_6_gamma[1060692900],"-", color='k', mew=5, linewidth=2, label="gamma")
plt.plot(spi_6_norm.index,spi_6_norm[1060692900],"-", color='red', mew=5, linewidth=2, label='norm')
plt.plot(spi_6_expo.index,spi_6_expo[1060692900],"-", color='green', mew=5, linewidth=2, label="exponweib")
plt.plot(spi_6_logn.index,spi_6_logn[1060692900],"-", color='blue', mew=5, linewidth=2, label="lognorm")
plt.ylabel('SPI-6 values (semi-arid Ethiopia)',size =25)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.legend(fontsize=20).texts[3].set_fontweight('bold')# make specific label bold

plt.subplot(4, 1, 3)

plt.plot(spi_6_gamma.index,spi_6_gamma[1060861400],"-", color='k', mew=5, linewidth=2, label="gamma")
plt.plot(spi_6_norm.index,spi_6_norm[1060861400],"-", color='red', mew=5, linewidth=2, label='norm')
plt.plot(spi_6_expo.index,spi_6_expo[1060861400],"-", color='green', mew=5, linewidth=2, label="exponweib")
plt.plot(spi_6_logn.index,spi_6_logn[1060861400],"-", color='blue', mew=5, linewidth=2, label="lognorm")
plt.ylabel('SPI-6 values (semi-arid Kenya)',size =25)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.legend(fontsize=20).texts[3].set_fontweight('bold')# make specific label bold

plt.subplot(4, 1, 4)

plt.plot(spi_6_gamma.index,spi_6_gamma[1060006850],"-", color='k', mew=5, linewidth=2, label="gamma")
plt.plot(spi_6_norm.index,spi_6_norm[1060006850],"-", color='red', mew=5, linewidth=2, label='norm')
plt.plot(spi_6_expo.index,spi_6_expo[1060006850],"-", color='green', mew=5, linewidth=2, label="exponweib")
plt.plot(spi_6_logn.index,spi_6_logn[1060006850],"-", color='blue', mew=5, linewidth=2, label="lognorm")
plt.ylabel('SPI-6 values (semi-arid Somalia)',size =25)
plt.xlabel('Date',size =25)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.legend(fontsize=20).texts[2].set_fontweight('bold')# make specific label bold

# set axis tick size
plt.xticks(size=25)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.show()
#%%

fig = plt.subplots(4,1,figsize=(40,40),sharex=True)

plt.subplot(4, 1, 1)
plt.title('Comparison of different distribution fittings',size=35)

plt.plot(ssmi_1_beta.index,ssmi_1_beta[1061020080],"-", color='k', mew=5, linewidth=2, label="beta")
plt.plot(ssmi_1_norm.index,ssmi_1_norm[1061020080],"-", color='red', mew=5, linewidth=2, label='norm')
plt.plot(ssmi_1_pear.index,ssmi_1_pear[1061020080],"-", color='green', mew=5, linewidth=2, label="pearson3")
plt.plot(ssmi_1_fisk.index,ssmi_1_fisk[1061020080],"-", color='blue', mew=5, linewidth=2, label="fisk")
plt.ylabel('SSMI-1 values (Ethiopian highlands)',size =25)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.legend(fontsize=20).texts[3].set_fontweight('bold')# make specific label bold


plt.subplot(4, 1, 2)

plt.plot(ssmi_1_beta.index,ssmi_1_beta[1060692900],"-", color='k', mew=5, linewidth=2, label="beta")
plt.plot(ssmi_1_norm.index,ssmi_1_norm[1060692900],"-", color='red', mew=5, linewidth=2, label='norm')
plt.plot(ssmi_1_pear.index,ssmi_1_pear[1060692900],"-", color='green', mew=5, linewidth=2, label="pearson3")
plt.plot(ssmi_1_fisk.index,ssmi_1_fisk[1060692900],"-", color='blue', mew=5, linewidth=2, label="fisk")
plt.ylabel('SSMI-1 values (semi-arid Ethiopia)',size =25)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.legend(fontsize=20).texts[2].set_fontweight('bold')# make specific label bold

plt.subplot(4, 1, 3)

plt.plot(ssmi_1_beta.index,ssmi_1_beta[1060861400],"-", color='k', mew=5, linewidth=2, label="beta")
plt.plot(ssmi_1_norm.index,ssmi_1_norm[1060861400],"-", color='red', mew=5, linewidth=2, label='norm')
plt.plot(ssmi_1_pear.index,ssmi_1_pear[1060861400],"-", color='green', mew=5, linewidth=2, label="pearson3")
plt.plot(ssmi_1_fisk.index,ssmi_1_fisk[1060861400],"-", color='blue', mew=5, linewidth=2, label="fisk")
plt.ylabel('SSMI-1 values (semi-arid Kenya)',size =25)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.legend(fontsize=20).texts[3].set_fontweight('bold')# make specific label bold

plt.subplot(4, 1, 4)

plt.plot(ssmi_1_beta.index,ssmi_1_beta[1060006850],"-", color='k', mew=5, linewidth=2, label="beta")
plt.plot(ssmi_1_norm.index,ssmi_1_norm[1060006850],"-", color='red', mew=5, linewidth=2, label='norm')
plt.plot(ssmi_1_pear.index,ssmi_1_pear[1060006850],"-", color='green', mew=5, linewidth=2, label="pearson3")
plt.plot(ssmi_1_fisk.index,ssmi_1_fisk[1060006850],"-", color='blue', mew=5, linewidth=2, label="fisk")
plt.ylabel('SSMI-1 values (semi-arid Somalia)',size =25)
plt.ylabel('Date',size =25)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.legend(fontsize=20).texts[3].set_fontweight('bold')# make specific label bold

# set axis tick size
plt.xticks(size=25)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.show()

#%%
fig = plt.subplots(4,1,figsize=(40,40),sharex=True)

plt.subplot(4, 1, 1)
plt.title('Comparison of different distribution fittings',size=35)

plt.plot(ssfi_1_gen.index,ssfi_1_gen[1061020080],"-", color='k', mew=5, linewidth=2, label="genextreme")
plt.plot(ssfi_1_pear.index,ssfi_1_pear[1061020080],"-", color='red', mew=5, linewidth=2, label='pearson3')
plt.plot(ssfi_1_expo.index,ssfi_1_expo[1061020080],"-", color='green', mew=5, linewidth=2, label="exponweib")
plt.plot(ssfi_1_logn.index,ssfi_1_logn[1061020080],"-", color='blue', mew=5, linewidth=2, label="lognorm")
plt.ylabel('SSI-1  values (Ethiopian highlands)',size =25)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.legend(fontsize=20).texts[1].set_fontweight('bold')# make specific label bold


plt.subplot(4, 1, 2)

plt.plot(ssfi_1_gen.index,ssfi_1_gen[1060692900],"-", color='k', mew=5, linewidth=2, label="genextreme")
plt.plot(ssfi_1_pear.index,ssfi_1_pear[1060692900],"-", color='red', mew=5, linewidth=2, label='pearson3')
plt.plot(ssfi_1_expo.index,ssfi_1_expo[1060692900],"-", color='green', mew=5, linewidth=2, label="exponweib")
plt.plot(ssfi_1_logn.index,ssfi_1_logn[1060692900],"-", color='blue', mew=5, linewidth=2, label="lognorm")
plt.ylabel('SSI-1  values (semi-arid Ethiopia)',size =25)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.legend(fontsize=20).texts[2].set_fontweight('bold')# make specific label bold

plt.subplot(4, 1, 3)

plt.plot(ssfi_1_gen.index,ssfi_1_gen[1060861400],"-", color='k', mew=5, linewidth=2, label="genextreme")
plt.plot(ssfi_1_pear.index,ssfi_1_pear[1060861400],"-", color='red', mew=5, linewidth=2, label='pearson3')
plt.plot(ssfi_1_expo.index,ssfi_1_expo[1060861400],"-", color='green', mew=5, linewidth=2, label="exponweib")
plt.plot(ssfi_1_logn.index,ssfi_1_logn[1060861400],"-", color='blue', mew=5, linewidth=2, label="lognorm")
plt.ylabel('SSI-1  values (semi-arid Kenya)',size =25)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.legend(fontsize=20).texts[3].set_fontweight('bold')# make specific label bold

plt.subplot(4, 1, 4)

plt.plot(ssfi_1_gen.index,ssfi_1_gen[1060006850],"-", color='k', mew=5, linewidth=2, label="genextreme")
plt.plot(ssfi_1_pear.index,ssfi_1_pear[1060006850],"-", color='red', mew=5, linewidth=2, label='pearson3')
plt.plot(ssfi_1_expo.index,ssfi_1_expo[1060006850],"-", color='green', mew=5, linewidth=2, label="exponweib")
plt.plot(ssfi_1_logn.index,ssfi_1_logn[1060006850],"-", color='blue', mew=5, linewidth=2, label="lognorm")
plt.ylabel('SSI-1 values (semi-arid Somalia)',size =25)
plt.ylabel('Date',size =25)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.legend(fontsize=20).texts[3].set_fontweight('bold')# make specific label bold

# set axis tick size
plt.xticks(size=25)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.show()
#%%
#
df_tendaho = pd.read_excel(r'C:\Users\roo290\surfdrive (backup)\Data\Data_Validation.xlsx', sheet_name='Sheet1')
df_melka = pd.read_excel(r'C:\Users\roo290\surfdrive (backup)\Data\Data_Validation.xlsx',sheet_name='Sheet2')

df_tendaho.set_index('Unnamed: 0', inplace = True)
df_melka.set_index('Unnamed: 0', inplace =True)

df_melka['anomalies_obs'] = df_melka['Obs_melka']/(df_melka['Obs_melka'].mean())
df_melka['anomalies_glo'] = df_melka['Glofas_melka']/(df_melka['Glofas_melka'].mean())
df_tendaho['anomalies_obs'] = df_tendaho['Obs_Tendaho']/(df_tendaho['Obs_Tendaho'].mean())
df_tendaho['anomalies_glo'] =df_tendaho['Glofas_Tendaho']/(df_tendaho['Glofas_Tendaho'].mean())

#%%
fig,ax = plt.subplots(figsize=(30,15))
    # make a plot
l1=plt.plot(df_melka.index,
            df_melka['anomalies_obs'],
            color="k", 
            marker="o", label = "anomalies_observations")
l2=plt.plot(df_melka.index,
            df_melka['anomalies_glo'],
            color="b",marker="o", label="anomalies_GloFAS")
# set x-axis label
plt.xlabel("Year", fontsize = 30)

# set y-axis label
plt.ylabel("Anomalies", color="b",fontsize=30)

# twin object for two different y-axis on the sample plot
#ax2=ax.twinx()

# make a plot with different y-axis using second axis object
#ax2.plot(data_yield_Admin['Year'][31:].astype(str), data_yield_Admin[county][31:],color="blue",marker="o")
#ax2.set_ylabel("modelled_yield",color="blue",fontsize=14)
plt.title('Melka Kuntire',size=30) # give plots a title
plt.legend(fontsize=20) # create legend
plt.tick_params(axis='both', which='major', labelsize=30)
#plt.setp(ax2.get_xticklabels(), rotation=45)
plt.show()
# save the plot as a file
fig.savefig('C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/Author_responses/Melka_kuntire_obs_glo.jpg',
            format='jpeg',
            dpi=100,
            bbox_inches='tight')

#%%

dist_names = ['norm','gamma', 'exponweib', 'lognorm']
# find fit for each optional distribution
dist_results = []
params = {}
for dist_name in dist_names:                                                # Find distribution parameters        
    dist = getattr(stats, dist_name)
    param = dist.fit(data_monthly_Sheds_new[1060005720])
    params[dist_name] = param
    
    # Assess goodness-of-fit using Kolmogorov–Smirnov test
    D, p = stats.kstest(data_monthly_Sheds_new[1060005720], dist_name, args=param)                      # Applying the Kolmogorov-Smirnov test
    dist_results.append((dist_name, p))

#%%
#####################################################Propagation plots###############################################################################################
#Loading all the indices into one dictionary

super_dictionary = {}

def create_super_dictionary(index, accumulation):
    opened_file = pd.read_excel(f"C:/Users/roo290/OneDrive - Vrije Universiteit Amsterdam/Data/New_results_PPR1/{index}_{accumulation}.xlsx")
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
fig = plt.subplots(3,1,figsize=(30,15),sharex=True)

plt.subplot(3, 1, 1)
plt.title('Propagation of drought-(semi-)arid catchment in Somalia',size=20)

plt.plot(super_dictionary[('spi', '1')][1061101150].index,super_dictionary[('spi', '1')][1061101150],"-", color='k', mew=5, linewidth=2, label="SPI-1")
#plt.plot(super_dictionary[('spi', '1')][1061020080].index,Var_threshold_precipitation, "--", color='k') #
#plt.fill_between(dates[g_in:g_out],df["rainfall [mm]"][g_in:g_out],Var_threshold_precipitation, where=(df["rainfall [mm]"][g_in:g_out]< Var_threshold_precipitation), color='r')
plt.ylabel('Precipitation (mm)',size =15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=14)

plt.subplot(3, 1, 2)

plt.plot(super_dictionary[('spi', '5')][1061101150].index,super_dictionary[('spi', '5')][1061101150],"-", color='k', mew=5,linewidth=2, label="SPI-5")
plt.plot(super_dictionary[('ssmi', '1')][1061101150].index,super_dictionary[('ssmi', '1')][1061101150],"--", color='b', mew=5,linewidth=2, label="SSMI-1")
#plt.plot(super_dictionary[('spi', '1')][1061020080].index,Var_threshold_discharge, "--", color='k')
#plt.fill_between(dates[g_in:g_out],df['discharge [m3/s]'][g_in:g_out],Var_threshold_discharge, where=(df['discharge [m3/s]'][g_in:g_out]< Var_threshold_discharge), color='r')
plt.ylabel('Soil_moisture(mm/month)',size =15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=14)

plt.subplot(3, 1, 3)

plt.plot(super_dictionary[('spi', '7')][1061101150].index,super_dictionary[('spi', '7')][1061101150],"-", color='k', mew=5,linewidth=2, label="SPI-7")
plt.plot(super_dictionary[('ssfi', '1')][1061101150].index,super_dictionary[('ssfi', '1')][1061101150],"--", color='b', mew=5,linewidth=2, label="SSI-1")
#plt.plot(super_dictionary[('spi', '1')][1061020080].index,Var_threshold_groundwater, "--", color='k')
#plt.fill_between(dates[g_in:g_out],df["lwe_thickness [m]"][g_in:g_out],Var_threshold_groundwater, where=(df["lwe_thickness [m]"][g_in:g_out]< Var_threshold_groundwater), color='r')
plt.ylabel('Discharge (m3/sec)',size =15)
plt.legend(fontsize=14)

# set x-axis label and tick size
#fig.text(0.5, 0.04, 'Date', ha='center', size=15)
plt.xticks(size=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.show()


#%%
fig = plt.subplots(3,1,figsize=(30,15),sharex=True)

plt.subplot(3, 1, 1)
plt.title('Propagation of drought-humid catchment in Kenya',size=20)

plt.plot(super_dictionary[('spi', '1')][1061118600].index,super_dictionary[('spi', '1')][1061118600],"-", color='k', mew=5, linewidth=2, label="SPI-1")
plt.ylabel('Precipitation (mm)',size =15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=14)

plt.subplot(3, 1, 2)

plt.plot(super_dictionary[('spi', '4')][1061118600].index,super_dictionary[('spi', '4')][1061118600],"-", color='k', mew=5, linewidth=2, label="SPI-4")
plt.plot(super_dictionary[('ssmi', '1')][1061101150].index,super_dictionary[('ssmi', '1')][1061101150],"--", color='b', mew=5,linewidth=2, label="SSMI-1")
#plt.plot(super_dictionary[('spi', '1')][1061020080].index,Var_threshold_discharge, "--", color='k')
#plt.fill_between(dates[g_in:g_out],df['discharge [m3/s]'][g_in:g_out],Var_threshold_discharge, where=(df['discharge [m3/s]'][g_in:g_out]< Var_threshold_discharge), color='r')
plt.ylabel('Soil_moisture(mm/month)',size =15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=14)

plt.subplot(3, 1, 3)

plt.plot(super_dictionary[('spi', '2')][1061118600].index,super_dictionary[('spi', '2')][1061118600],"-", color='k', mew=5, linewidth=2, label="SPI-2")
plt.plot(super_dictionary[('ssfi', '1')][1061101150].index,super_dictionary[('ssfi', '1')][1061101150],"--", color='b', mew=5,linewidth=2, label="SSI-1")
#plt.plot(super_dictionary[('spi', '1')][1061020080].index,Var_threshold_groundwater, "--", color='k')
#plt.fill_between(dates[g_in:g_out],df["lwe_thickness [m]"][g_in:g_out],Var_threshold_groundwater, where=(df["lwe_thickness [m]"][g_in:g_out]< Var_threshold_groundwater), color='r')
plt.ylabel('Discharge (m3/sec)',size =15)
plt.legend(fontsize=14)

# set axis tick size
plt.xticks(size=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.show()
#%%
fig = plt.subplots(3,1,figsize=(30,15),sharex=True)

plt.subplot(3, 1, 1)
plt.title('Propagation of drought-humid catchment in Ethiopia',size=20)

plt.plot(super_dictionary[('spi', '1')][1061020080].index,super_dictionary[('spi', '1')][1061020080],"-", color='k', mew=5, linewidth=2, label="SPI-1")
#plt.plot(super_dictionary[('spi', '1')][1061020080].index,Var_threshold_precipitation, "--", color='k') #
#plt.fill_between(dates[g_in:g_out],df["rainfall [mm]"][g_in:g_out],Var_threshold_precipitation, where=(df["rainfall [mm]"][g_in:g_out]< Var_threshold_precipitation), color='r')
plt.ylabel('Precipitation (mm)',size =15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=14)

plt.subplot(3, 1, 2)

plt.plot(super_dictionary[('spi', '3')][1061020080].index,super_dictionary[('spi', '3')][1061020080],"-", color='k', mew=5, linewidth=2, label="SPI-3")
plt.plot(super_dictionary[('ssmi', '1')][1061101150].index,super_dictionary[('ssmi', '1')][1061101150],"--", color='b', mew=5,linewidth=2, label="SSMI-1")
#plt.plot(super_dictionary[('spi', '1')][1061020080].index,Var_threshold_discharge, "--", color='k')
#plt.fill_between(dates[g_in:g_out],df['discharge [m3/s]'][g_in:g_out],Var_threshold_discharge, where=(df['discharge [m3/s]'][g_in:g_out]< Var_threshold_discharge), color='r')
plt.ylabel('Soil_moisture(mm/month)',size =15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=14)

plt.subplot(3, 1, 3)

plt.plot(super_dictionary[('spi', '2')][1061020080].index,super_dictionary[('spi', '2')][1061020080],"-", color='k', mew=5, linewidth=2, label="SPI-2")
plt.plot(super_dictionary[('ssfi', '1')][1061101150].index,super_dictionary[('ssfi', '1')][1061101150],"--", color='b', mew=5,linewidth=2, label="SSI-1")
#plt.plot(super_dictionary[('spi', '1')][1061020080].index,Var_threshold_groundwater, "--", color='k')
#plt.fill_between(dates[g_in:g_out],df["lwe_thickness [m]"][g_in:g_out],Var_threshold_groundwater, where=(df["lwe_thickness [m]"][g_in:g_out]< Var_threshold_groundwater), color='r')
plt.ylabel('Discharge (m3/sec)',size =15)
plt.legend(fontsize=14)

# set axis tick size
plt.xticks(size=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.show()
#%%%
##########################################################Threshold testing###########################################################################################
# Name of the input file

    
def VarThres(df, shed, percentile, dates):
    perc = 1-percentile
    Varthreshold = df[shed].groupby(dates.month).quantile(perc) # get percentile for every month
    VarThreshold = np.tile(Varthreshold.values, int(len(df[shed])/12)) # repeat the montly percentiles for every year, so it can be plotted
    return VarThreshold 

percentile = 0.7
shed = 1061101150
df_var_pre = data_monthly_Sheds_new
df_var_sm = data_monthly_Sm_new
df_var_dis = data_monthly_dis_new


Var_Threshold = np.zeros(df_var_pre[shed].shape)


Var_Threshold_prec_70 = VarThres(df_var_pre,shed,percentile,df_var_pre[shed].index) # filling in VarThres function

Var_Threshold_sm70 = VarThres(df_var_sm,shed,percentile,df_var_sm[shed].index) # filling in VarThres function

Var_Threshold_dis70 = VarThres(df_var_dis,shed,percentile,df_var_dis[shed].index) # filling in VarThres function

#%%
percentile = 0.8
Var_Threshold_prec_80 = VarThres(df_var_pre,shed,percentile,df_var_pre[shed].index) # filling in VarThres function

Var_Threshold_sm80 = VarThres(df_var_sm,shed,percentile,df_var_sm[shed].index) # filling in VarThres function

Var_Threshold_dis80 = VarThres(df_var_dis,shed,percentile,df_var_dis[shed].index) # filling in VarThres function

#%%
percentile = 0.9
Var_Threshold_prec_90 = VarThres(df_var_pre,shed,percentile,df_var_pre[shed].index) # filling in VarThres function

Var_Threshold_sm90 = VarThres(df_var_sm,shed,percentile,df_var_sm[shed].index) # filling in VarThres function

Var_Threshold_dis90 = VarThres(df_var_dis,shed,percentile,df_var_dis[shed].index) # filling in VarThres function

#%%
#70% percentile
fig = plt.figure(figsize=(30,10))

plt.subplot(3, 1, 1)
plt.title('Threshold selection- 70th percentile',size=20)

plt.plot(data_monthly_Sheds_new.index,data_monthly_Sheds_new[1061101150],"-", color='k', mew=5)
plt.plot(data_monthly_Sheds_new.index,Var_Threshold_prec_70, "--", color='k') #
plt.fill_between(data_monthly_Sheds_new.index,data_monthly_Sheds_new[1061101150],Var_Threshold_prec_70, where=(data_monthly_Sheds_new[1061101150]< Var_Threshold_prec_70), color='r')
plt.ylabel('Precipitation (mm)',size =15)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(3, 1, 2)

plt.plot(df_var_sm.index,df_var_sm[shed],"-", color='k', mew=5)
plt.plot(df_var_sm.index,Var_Threshold_sm70, "--", color='k')
plt.fill_between(df_var_sm.index,df_var_sm[shed],Var_Threshold_sm70, where=(df_var_sm[shed]< Var_Threshold_sm70), color='r')
plt.ylabel('Soil_moisture(mm/month)',size =15)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(3, 1, 3)

plt.plot(df_var_dis.index,df_var_dis[shed],"-", color='k', mew=5)
plt.plot(df_var_dis.index,Var_Threshold_dis70, "--", color='k')
plt.fill_between(df_var_dis.index,df_var_dis[shed],Var_Threshold_dis70, where=(df_var_dis[shed]< Var_Threshold_dis70), color='r')
plt.ylabel('Discharge (m3/sec)',size =15)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.show()
#%%
#80% percentile

fig = plt.figure(figsize=(30,10))

plt.subplot(3, 1, 1)
plt.title('Threshold selection- 80th percentile',size=20)

plt.plot(data_monthly_Sheds_new.index,data_monthly_Sheds_new[1061101150],"-", color='k', mew=5)
plt.plot(data_monthly_Sheds_new.index,Var_Threshold_prec_80, "--", color='k') #
plt.fill_between(data_monthly_Sheds_new.index,data_monthly_Sheds_new[1061101150],Var_Threshold_prec_80, where=(data_monthly_Sheds_new[1061101150]< Var_Threshold_prec_80), color='r')
plt.ylabel('Precipitation (mm)',size =15)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(3, 1, 2)

plt.plot(df_var_sm.index,df_var_sm[shed],"-", color='k', mew=5)
plt.plot(df_var_sm.index,Var_Threshold_sm80, "--", color='k')
plt.fill_between(df_var_sm.index,df_var_sm[shed],Var_Threshold_sm80, where=(df_var_sm[shed]< Var_Threshold_sm80), color='r')
plt.ylabel('Soil_moisture(mm/month)',size =15)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(3, 1, 3)

plt.plot(df_var_dis.index,df_var_dis[shed],"-", color='k', mew=5)
plt.plot(df_var_dis.index,Var_Threshold_dis80, "--", color='k')
plt.fill_between(df_var_dis.index,df_var_dis[shed],Var_Threshold_dis80, where=(df_var_dis[shed]< Var_Threshold_dis80), color='r')
plt.ylabel('Discharge (m3/sec)',size =15)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.show()
#%%
#90% percentile
fig = plt.figure(figsize=(30,10))

plt.subplot(3, 1, 1)
plt.title('Threshold selection- 90th percentile',size=20)

plt.plot(data_monthly_Sheds_new.index,data_monthly_Sheds_new[1061101150],"-", color='k', mew=5)
plt.plot(data_monthly_Sheds_new.index,Var_Threshold_prec_90, "--", color='k') #
plt.fill_between(data_monthly_Sheds_new.index,data_monthly_Sheds_new[1061101150],Var_Threshold_prec_90, where=(data_monthly_Sheds_new[1061101150]< Var_Threshold_prec_90), color='r')
plt.ylabel('Precipitation (mm)',size =15)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(3, 1, 2)

plt.plot(df_var_sm.index,df_var_sm[shed],"-", color='k', mew=5)
plt.plot(df_var_sm.index,Var_Threshold_sm90, "--", color='k')
plt.fill_between(df_var_sm.index,df_var_sm[shed],Var_Threshold_sm90, where=(df_var_sm[shed]< Var_Threshold_sm90), color='r')
plt.ylabel('Soil_moisture(mm/month)',size =15)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplot(3, 1, 3)

plt.plot(df_var_dis.index,df_var_dis[shed],"-", color='k', mew=5)
plt.plot(df_var_dis.index,Var_Threshold_dis90, "--", color='k')
plt.fill_between(df_var_dis.index,df_var_dis[shed],Var_Threshold_dis90, where=(df_var_dis[shed]< Var_Threshold_dis90), color='r')
plt.ylabel('Discharge (m3/sec)',size =15)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.show()

#%%
#Loading the observed vs modelled data for 3 stations
path = r'C:\Users\roo290\surfdrive (backup)\Data\Data_Validation.xlsx'
dis_data = pd.read_excel(path, sheet_name=['Tendaho','Melka Kuntire','AB Ontulili'])

fig = plt.figure(figsize=(30,15))
plt.title('Tendaho, Ethiopia',size=30)

plt.plot(dis_data['Tendaho']['Date'],dis_data['Tendaho']['Observed'],"-", color='k', mew=5, linewidth=2, label="in-situ observations")
plt.plot(dis_data['Tendaho']['Date'],dis_data['Tendaho']['Modelled'],"-", color='b', mew=5, linewidth=2, label="modeled GloFAS")
#plt.fill_between(dates[g_in:g_out],df["rainfall [mm]"][g_in:g_out],Var_threshold_precipitation, where=(df["rainfall [mm]"][g_in:g_out]< Var_threshold_precipitation), color='r')
plt.ylabel('Discharge (m3/s)',size =30)
plt.tick_params(axis='both', which='major', labelsize=30)
plt.legend(fontsize=20)

fig = plt.figure(figsize=(30,15))
plt.title('Melka Kuntire, Ethiopia',size=30)

plt.plot(dis_data['Melka Kuntire']['Date'],dis_data['Melka Kuntire']['Observed'],"-", color='k', mew=5, linewidth=2, label="in-situ observations")
plt.plot(dis_data['Melka Kuntire']['Date'],dis_data['Melka Kuntire']['Modelled'],"-", color='b', mew=5, linewidth=2, label="modeled GloFAS")
#plt.plot(super_dictionary[('spi', '1')][1061020080].index,Var_threshold_discharge, "--", color='k')
#plt.fill_between(dates[g_in:g_out],df['discharge [m3/s]'][g_in:g_out],Var_threshold_discharge, where=(df['discharge [m3/s]'][g_in:g_out]< Var_threshold_discharge), color='r')
plt.ylabel('Discharge (m3/s)',size =30)
plt.tick_params(axis='both', which='major', labelsize=30)
plt.legend(fontsize=20)

fig = plt.figure(figsize=(30,15))
plt.title('AB Ontulili, Kenya',size=30)

plt.plot(dis_data['AB Ontulili']['Date'],dis_data['AB Ontulili']['Observed'],"-", color='k', mew=5, linewidth=2, label="in-situ observations")
plt.plot(dis_data['AB Ontulili']['Date'],dis_data['AB Ontulili']['Modelled'],"-", color='b', mew=5, linewidth=2, label="modeled GloFAS")
#plt.plot(super_dictionary[('spi', '1')][1061020080].index,Var_threshold_groundwater, "--", color='k')
#plt.fill_between(dates[g_in:g_out],df["lwe_thickness [m]"][g_in:g_out],Var_threshold_groundwater, where=(df["lwe_thickness [m]"][g_in:g_out]< Var_threshold_groundwater), color='r')
plt.ylabel('Discharge (m3/s)',size =30)
plt.tick_params(axis='both', which='major', labelsize=30)
plt.legend(fontsize=20)

#%%
import matplotlib.pyplot as plt

# Create a figure with 2 rows and 2 columns
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30,15))

# Remove the plot in the bottom right corner
axs[1, 1].remove()

# Plot the first figure in the top left corner
axs[0, 0].set_title('Tendaho, Ethiopia',size=30)
axs[0, 0].plot(dis_data['Tendaho']['Date'],dis_data['Tendaho']['Observed'],"-", color='k', mew=5, linewidth=2, label="in-situ observations")
axs[0, 0].plot(dis_data['Tendaho']['Date'],dis_data['Tendaho']['Modelled'],"-", color='b', mew=5, linewidth=2, label="modeled GloFAS")
axs[0, 0].set_ylabel('Discharge (m3/s)',size =30)
axs[0, 0].tick_params(axis='both', which='major', labelsize=30)
axs[0, 0].legend(fontsize=20)

# Plot the second figure in the top right corner
axs[0, 1].set_title('Melka Kuntire, Ethiopia',size=30)
axs[0, 1].plot(dis_data['Melka Kuntire']['Date'],dis_data['Melka Kuntire']['Observed'],"-", color='k', mew=5, linewidth=2, label="in-situ observations")
axs[0, 1].plot(dis_data['Melka Kuntire']['Date'],dis_data['Melka Kuntire']['Modelled'],"-", color='b', mew=5, linewidth=2, label="modeled GloFAS")
#axs[0, 1].set_ylabel('Discharge (m3/s)',size =30)
axs[0, 1].tick_params(axis='both', which='major', labelsize=30)
axs[0, 1].legend(fontsize=20)

# Plot the third figure in the center of the bottom row
axs[1, 0].set_title('AB Ontulili, Kenya',size=30)
axs[1, 0].plot(dis_data['AB Ontulili']['Date'],dis_data['AB Ontulili']['Observed'],"-", color='k', mew=5, linewidth=2, label="in-situ observations")
axs[1, 0].plot(dis_data['AB Ontulili']['Date'],dis_data['AB Ontulili']['Modelled'],"-", color='b', mew=5, linewidth=2, label="modeled GloFAS")
axs[1, 0].set_ylabel('Discharge (m3/s)',size =30)
axs[1, 0].tick_params(axis='both', which='major', labelsize=30)
axs[1, 0].legend(fontsize=20)

# Adjust the layout of the subplots
plt.tight_layout()

# Show the plot
plt.show()

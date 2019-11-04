import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats
from math import isnan
import re 

data=pd.read_csv('../Downloads/Arrest_Data_from_2010_to_Present.csv')


def spher_dist(lat1,long1,lat2,long2,R):
	lat1,lat2,long1,long2 = (np.pi*lat1)/180,(np.pi*lat2)/180,(np.pi*long1)/180,(np.pi*long2)/180
	del_lat = lat1-lat2
	del_long = long1-long2
	mean_lat = (lat1+lat2)/2
	x = del_lat**2 + (np.cos(mean_lat)*del_long)**2
	return R*np.sqrt(x)






m,n = data.shape
#num_2018 = np.count_nonzero(np.array([int(data['Arrest Date'][i][-4:])==2018 for i in range(m]))

age_ar = []
area_dic=defaultdict(int)
num_2018 = 0 
num_2018_lessthan_2km = 0 
pico_loc = []
age_by_group = defaultdict(list)
lat_bb, long_bb = 34.050536, -118.247861
R=  6371
for i in range(m):
    if int(data['Arrest Date'][i][-4:])==2018:
        num_2018 = num_2018+1
        area_id =  data['Area ID'][i]
        area_dic[area_id]=area_dic[area_id]+1        
        if data['Charge Group Description'][i] == 'Vehicle Theft' or data['Charge Group Description'][i] == 'Robbery' or data['Charge Group Description'][i] == 'Burglary' or data['Charge Group Description'][i] == 'Receive Stolen Property':
        	age_ar.append(data['Age'][i])
        #print(data['Charge Group Description'][i])
        if data['Charge Group Description'][i] != "Pre-Delinquency" and data['Charge Group Description'][i] != "Non-Criminal Detention" and (data['Charge Group Description'][i]):
        	age_by_group[data['Charge Group Description'][i]].append(data['Age'][i])

        #Calculating number of arrest incidents occurred within 2 km from the Bradbury Building in 2018
        lat_cur,long_cur = re.split(', ',data['Location'][i])
        lat_cur, long_cur = float(lat_cur[1:]), float(long_cur[:-1])
        dist = spher_dist(lat_cur,long_cur,lat_bb,long_bb,R)
        if dist <=2:
        	num_2018_lessthan_2km = num_2018_lessthan_2km +1

        #Calculating number of arrest incidents were made per kilometer on Pico Boulevard during 2018
        if "Pico" in data['Address'][i] or "PICO" in data['Address'][i]:
        	pico_loc.append([lat_cur,long_cur])

#Calculating the distance of the Pico Boulevard
pico_loc = np.array(pico_loc)
lat_ar = pico_loc[:,0]
long_ar = pico_loc[:,1]
pico_lat_mean, pico_lat_std = np.mean(lat_ar), np.std(lat_ar)
pico_long_mean, pico_long_std = np.mean(long_ar), np.std(long_ar)
new_pico_loc = []
for i,loc in enumerate(pico_loc):
	x,y = loc
	if abs(lat_ar[i]-pico_lat_mean) <= 2*pico_lat_std and abs(long_ar[i]-pico_long_mean) <= 2*pico_long_std:
		new_pico_loc.append((y,x))

new_pico_loc.sort()
pico_dist = spher_dist(new_pico_loc[0][1],new_pico_loc[0][0],new_pico_loc[-1][1],new_pico_loc[-1][0],R)
num_arrenst_per_km = np.ceil(len(new_pico_loc)/pico_dist)



avg_age_by_group = [np.mean(age_by_group[key]) for key in age_by_group]
#print(area_dic)
print("Number of Arrests made in 2018 is: ", num_2018)
print("95 percent quantile of the age of the arrestee in 2018: ", np.quantile(np.array(age_ar),0.95))
print("Bookings of arrestees were made in the area with the most arrests in 2018: ",max(area_dic.values()))
print("Z-score array", stats.zscore(avg_age_by_group))
print("Arrest incidents occurred within 2 km from the Bradbury Building in 2018: ",num_2018_lessthan_2km)
print("Arrest incidents were made per kilometer on Pico Boulevard during 2018: ",num_arrenst_per_km)
        
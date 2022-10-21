import numpy as np
import math
import random
import uuid 


class Node: 
    def __init__(self, id, latitude, longitude): 
        self.id = id
        self.latitude = latitude
        self.longitude = longitude

    def distance_to(self, second_node):
        '''Haversine formula for geodesic distance
        a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
        c = 2 ⋅ atan2( √a, √(1−a) )
        d = (Earth Radius) ⋅ c
        '''
        R = 6371e3 #m
        lat1 = self.latitude * np.pi/180; #rad
        lat2 = second_node.latitude * np.pi/180; #rad
        delta_lat = (second_node.latitude - self.latitude) * np.pi/180;
        delta_long = (self.longitude - second_node.longitude) * np.pi/180;
        a = (np.sin(delta_lat/2))**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_long/2) * np.sin(delta_long/2)
        c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
        d = R * c #m
        return d

class KNN: 
    def __init__(self, weather_stations, pm_stations, radius):
        self.weather_stations = weather_stations
        self.pm_stations = pm_stations
        self.radius = radius

    def get_neighbours(self, pm_station):
        nearest_weather_stations = list()
        for ws in self.weather_stations:
            dist = pm_station.distance_to(ws)
            if dist <= self.radius: 
                nearest_weather_stations.append({'id':ws.id, 'dist':dist})
        return nearest_weather_stations
    
    def find_corresponding_weather_stations(self, prune_val = None):
        pm_station_neighbours = {}
        for pms in self.pm_stations: 
            nearest_weather_stations = self.get_neighbours(pms)
            if prune_val != None: 
                nearest_weather_stations.sort(key=lambda dic: dic['dist'])
                nearest_weather_stations = nearest_weather_stations[:prune_val]
            pm_station_neighbours[pms.id] = nearest_weather_stations

            distances = [nws['dist'] for nws in nearest_weather_stations]
            distance_weights = [float(i)/sum(distances) for i in distances]
            for i,nws in enumerate(nearest_weather_stations):
                nws['weight'] = distance_weights[i]
        return pm_station_neighbours


            
#test with random latitude longitude
def randlatlon1():
    pi = math.pi
    cf = 180.0 / pi  # radians to degrees Correction Factor

    # get a random Gaussian 3D vector:
    gx = random.gauss(0.0, 1.0)
    gy = random.gauss(0.0, 1.0)
    gz = random.gauss(0.0, 1.0)

    # normalize to an equidistributed (x,y,z) point on the unit sphere:
    norm2 = gx*gx + gy*gy + gz*gz
    norm1 = 1.0 / math.sqrt(norm2)
    x = gx * norm1
    y = gy * norm1
    z = gz * norm1

    radLat = math.asin(z)      # latitude  in radians
    radLon = math.atan2(y,x)   # longitude in radians

    return (round(cf*radLat, 5), round(cf*radLon, 5))



weather_stations = []
for _ in range(100): 
    info = randlatlon1()
    id = uuid.uuid4().hex[:6].upper()
    node = Node(id = id, latitude = info[0], longitude = info[1])
    weather_stations.append(node)

pm_stations = []
for _ in range(10): 
    info = randlatlon1()
    id = uuid.uuid4().hex[:6].upper()
    node = Node(id = id, latitude = info[0], longitude = info[1])
    pm_stations.append(node)

    knn = KNN(weather_stations, pm_stations, radius = 10e6)
    pm_station_neighbours = knn.find_corresponding_weather_stations(prune_val = 5)


print('========== Sample Output with Random Lat, Lon, ID=================')
print(pm_station_neighbours)

    


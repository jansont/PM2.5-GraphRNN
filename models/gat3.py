import pandas as pd
from google.colab import drive
import numpy as np
import math
from tqdm import tqdm
from datetime import date, timedelta
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

from torch_geometric.data import Data

drive.mount('/content/drive')
path = '/content/drive/My Drive/Data/'

file1 = 'la_weather_with_pm_per_day.csv'
weather_data = pd.read_csv(path+file1)
file2 = 'metadata_with_station.csv'
metadata = pd.read_csv(path+file2)
metadata = metadata[metadata['location'] == 'Los Angeles (SoCAB)'].reset_index(drop = True)

def data_to_numpy(weather_data, edge_names, node_names, station_ids, date_range): 

    graph_node_features = np.empty((len(date_range), len(stations), len(node_cols)))
    graph_edge_features = np.empty((len(date_range), len(stations), len(edge_cols)))
    graph_labels = np.empty((len(date_range), len(stations)))

    for day_idx in range(len(date_range)): 
        for station_idx in range(len(stations)): 
                d = date_range[day_idx]             #get date from index
                station = stations[station_idx]     #get station number from index
                vals = weather_data[weather_data['DATE'] == d]  #get data date using date
                vals = vals[vals['STATION'] == station]         #get data of station on date
                pm = vals['pm25'].values
                edge = vals[edge_cols]
                edge_vals = np.array(edge.values.tolist()).flatten()  #edge feature as array
                node_vals = vals[node_cols]
                node_vals = np.array(node_vals.values.tolist()).flatten() #node features as array
                if len(node_vals) == 0:                           #if there is no weather data on date, set to all zeros
                    node_vals = np.zeros(len(node_cols))
                graph_node_features[day_idx, station_idx] = node_vals 
                if len(pm) == 0:     #if no pm label on date set to 0
                    pm = np.zeros(1)
                graph_labels[day_idx, station_idx] = pm
                if len(edge_vals) == 0:  #if no edge feature, set to 0
                    edge_vals = np.zeros(len(edge_cols))
                graph_edge_features[day_idx, station_idx] = edge_vals

    return graph_node_features, graph_edge_features, graph_labels

def generate_nodes(metadata):
    stations = metadata['STATION'].unique()
    stations.sort()
    nodes = OrderedDict()
    for id in stations:
        row = metadata[metadata['STATION'] == id]
        lat = row['Latitudes'].values[0]
        lon = row['Longitudes'].values[0]
        # elev = row['Elevetation']
        nodes.update({id:{'Latitude':lat,'Longitude':lon}})
    return nodes

def get_node_features(nodes):
    altitudes = [nodes[id]['altitude'] for id in nodes]
    node_features = np.array(altitudes)
    return node_features


def geo_distance(first_node, second_node):
        '''Haversine formula for geodesic distance'''
        lat1, long1 = first_node
        lat2, long2 = second_node
        R = 6371e3 #m
        lat1 = lat1 * np.pi/180; #rad
        lat2 = lat2 * np.pi/180; #rad
        delta_lat = (lat2 - lat1) * np.pi/180;
        delta_long = (long1 - long2) * np.pi/180;
        a = (np.sin(delta_lat/2))**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_long/2) * np.sin(delta_long/2)
        c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
        d = R * c #m
        return d

def generate_node_distances(coordinates):
    distance_matrix = np.zeros((len(coordinates), len(coordinates)))
    for i in range(len(coordinates)): 
        coord1 = coordinates[i]
        for j in range(len(coordinates)): 
            coord2 = coordinates[j]
            distance = geo_distance(coord1, coord2)
            distance_matrix[i][j] = distance
    return distance_matrix


def sparse_adjacency(adj): 
    """Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.
    """
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    index = adj.nonzero(as_tuple=True)
    edge_attr = adj[index]

    if len(index) == 3:
        batch = index[0] * adj.size(-1)
        index = (batch + index[1], batch + index[2])

    return torch.stack(index, dim=0), edge_attr


class Graph():
    def __init__(self,
                 graph_metadata,
                 edge_data, 
                 distance_threshold):
        self.graph_metadata = graph_metadata
        self.distance_threshold = distance_threshold
        self.nodes = generate_nodes(graph_metadata)
        self.size = len(self.nodes)
        self.edge_data = edge_data
        self.edge_index, self.edge_attr = self.generate_edges()
        self.edge_attr = self.edge_attr.transpose()
        self.adjacency = self.edge_list_sequence_to_adj()

    def generate_edges(self):
        nodes = self.nodes             
        node_list = list(self.nodes)        #get list of node ids
        coordinates = list(zip(self.graph_metadata['Latitudes'], self.graph_metadata['Longitudes']))  #zip latitude and longitude of all nodes
        distance_matrix = generate_node_distances(coordinates)  #square matrix of distance to all other nodes

        adj_matrix = np.identity(self.size)
        adj_matrix[distance_matrix < self.distance_threshold] = 1 #in adj matrix, set entry to 1 if distance between nodes below threshold

        distance_matrix = distance_matrix * adj_matrix

        edge_idx, edge_dist = sparse_adjacency(torch.tensor(distance_matrix)) #edge_idx : shape (2 * number of connected nodes (dis < threshold)) 
        edge_idx, edge_dist = edge_idx.numpy(), edge_dist.numpy()             #edge_dist: same shape as above, distance values between those node indices
        edge_vectors = []#other edge attributes?

        for i in range(edge_idx.shape[1]):
            #get index of non-zero edges
            source_idx = edge_idx[0, i]
            dest_idx = edge_idx[1, i]
            #get lat lon for the nodes at ends of non zero edge
            key0 = node_list[source_idx]
            lat0 = nodes[key0]['Latitude']
            long0 = nodes[key0]['Longitude']
            key1 = node_list[dest_idx]
            lat1 = nodes[key1]['Latitude']
            long1 = nodes[key1]['Longitude']
            distance_x = geo_distance((lat0, 0), (lat1, 0))
            distance_y = geo_distance((0, long1), (0, long1))
            edge_vect_x = distance_x 
            #get wind along x and y vectors from both source and edge
            wind_source_x = self.edge_data[:, source_idx, 0]
            wind_dest_x = self.edge_data[:, dest_idx, 0]
            wind_source_y = self.edge_data[:, source_idx, 1]
            wind_dest_y = self.edge_data[:, dest_idx, 1]
            #average source and edge wind components to get net wind 
            wind_x = (wind_source_x + wind_dest_x)/2
            wind_y = (wind_source_y + wind_dest_y)/2

            diffusion_x = distance_x * wind_x 
            diffusion_y = distance_y * wind_y

            dist_dir = np.tan(distance_y / (distance_x))
            wind_dir = np.tan(wind_y / (wind_x+1e-12))
            wind_str = wind_x**2 + wind_y**2
            distance = distance_x**2 + distance_y**2
            delta_angle = np.abs(wind_dir - dist_dir)
            edge_weight = (wind_str * np.cos(delta_angle) / distance**0.5)
            #normalize edge weights
            mean_edge_weight = np.mean(edge_weight)
            std_edge_weight = np.std(edge_weight)
            normalized_edge_weight = (edge_weight - mean_edge_weight) / std_edge_weight

            edge_vectors.append(edge_weight)

        edge_vectors = np.stack(edge_vectors)

        return edge_idx, edge_vectors

    def edge_list_to_adj(self):
        adj = np.identity(self.edge_index.size)
        for k,(i,j) in enumerate(zip(self.edge_index[0], self.edge_index[1])):
            adj[i,j] = self.edge_attr[j]
        return adj

    def edge_list_sequence_to_adj(self):
        adjacencies = []
        for i in range(self.edge_attr.shape[0]):
            adj = np.identity(self.size)
            for k,(i,j) in enumerate(zip(self.edge_index[0], self.edge_index[1])):
                adj[i,j] = self.edge_attr[i][k]
            adjacencies.append(adj)
        return np.array(adjacencies)

def get_sliding_timeframe(vect, sequence_len):
    vect_len = vect.shape[0]
    timeframes = []
    for i in range(sequence_len, vect_len):
        timeframe = vect[i - sequence_len : i]
        timeframes.append(timeframe)
    timeframes = np.stack(timeframes, axis = 0)
    return timeframes

def normalize(vect, mean, std):
    norm_v = (vect - mean) / std
    return norm_v



class WeatherData(Dataset):
    def __init__(self, 
                 labels,   
                 node_features,
                 edge_features, 
                 edge_index, 
                 historical_len, 
                 prediction_len, 
                 sequence = True):
        
        self.historical_len = historical_len
        self.prediction_len = prediction_len
        self.sequence_len = historical_len + prediction_len

        self.edge_features = edge_features
        self.edge_index = edge_index

        label_mean = labels.mean(axis = (0,1))
        label_sdev = labels.std(axis = (0,1))
        self.labels = normalize(labels, label_mean, label_sdev)

        feature_mean = node_features.mean()
        feature_sdev = node_features.std()
        self.features = normalize(node_features, feature_mean, feature_sdev)

        if sequence: 
            self.features = get_sliding_timeframe(self.features, self.sequence_len)
            self.edge_features = get_sliding_timeframe(self.edge_features, self.sequence_len)
            self.labels = get_sliding_timeframe(self.labels, self.sequence_len)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.features[idx], self.edge_index, self.edge_features[idx], self.labels[idx]
        # self.labels[idx], self.features[idx], self.edge_features[idx], self.edge_indedx
        return data

dataloader = DataLoader(dataset, batch_size = 64)
for data in dataloader:
    node_features, edge_index, edge_features, labels = data
    graph_data = Data(node_features, edge_index, edge_features, labels)
    
    break


# 
start = date(2018,2,1)
end = date(2021,1,1)
date_range = pd.date_range(start,end-timedelta(days=1))
date_range = [str(x)[:10] for x in date_range]
node_cols = ['ceiling', 'visibility', 'dew', 'precipitation_duration', 'precipitation_depth']
edge_cols = ['wind_x', 'wind_y']
stations = metadata['STATION'].unique()
stations.sort()


graph_node_features, graph_edge_features, graph_labels = data_to_numpy(weather_data, edge_cols, node_cols, stations, date_range)
graph_node_features = np.nan_to_num(graph_node_features)

graph = Graph(metadata, graph_edge_features, distance_threshold = 30e3)

split = int(labels.shape[0]*0.8)

train_dataset = WeatherData(edge_features = graph.edge_attr[:split], 
                      labels = graph_labels[:split],
                      node_features = graph_node_features[:split],
                      edge_index = graph.edge_index,
                      historical_len = 5,
                      prediction_len = 2, 
                      sequence = False)


val_dataset = WeatherData(edge_features = graph.edge_attr[split:], 
                      labels = graph_labels[split:],
                      node_features = graph_node_features[split:],
                      edge_index = graph.edge_index,
                      historical_len = 5,
                      prediction_len = 2, 
                      sequence = False)


train_dataloader = DataLoader(train_dataset, batch_size = 1)
val_dataloader = DataLoader(val_dataset, batch_size = 1)




"""## GAT implementation"""



!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""## Structure"""

in_features = 5
out_features = 2
nb_nodes = 3

W = nn.Parameter(torch.zeros(size=(in_features, out_features))) #xavier paramiter inizializator
nn.init.xavier_uniform_(W.data, gain=1.414)

input = torch.rand(nb_nodes,in_features) 


# linear transformation
h = torch.mm(input, W)
N = h.size()[0]

print(h.shape)

a = nn.Parameter(torch.zeros(size=(2*out_features, 1))) #xavier paramiter inizializator
nn.init.xavier_uniform_(a.data, gain=1.414)
print(a.shape)

leakyrelu = nn.LeakyReLU(0.2)  # LeakyReLU

a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * out_features)

e = leakyrelu(torch.matmul(a_input, a).squeeze(2))





"""# Use it"""

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import matplotlib.pyplot as plt

name_data = 'Cora'
dataset1 = Planetoid(root= '/tmp/' + name_data, name = name_data)
dataset.transform = T.NormalizeFeatures()

print(f"Number of Classes in {name_data}:", dataset1.num_classes)
print(f"Number of Node Features in {name_data}:", dataset1.num_node_features)

class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        
        self.conv1 = GATConv(5, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head, 1, concat=False,
                             heads=self.out_head, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Dropout before the GAT layer is used to avoid overfitting in small datasets like Cora.
        # One can skip them if the dataset is sufficiently large.

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GAT().to(device)

data = dataset1[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

criterion = nn.MSELoss()




train_loss = []
train_rms = []
val_loss = []
val_rms = []


for data in train_dataloader:
    node_features, edge_index, edge_features, labels = data
    node_features, edge_index, labels = node_features.squeeze().float(), edge_index.squeeze(), labels.squeeze().float()
    node_features, edge_index, labels = node_features.to(device), edge_index.to(device), labels.to(device)
    graph_data = Data(x = node_features, edge_index = edge_index, y = labels)


train_loss = []
train_rms = []
val_loss = []
val_rms = []

train_mask = [0,1,3,4,7,8,9,11,12]
val_mask = [2,5,6,10]

for epoch in range(5):

    model.train()

    optimizer.zero_grad()
    out = model(graph_data)
    out = out.squeeze()
    loss = criterion(out[train_mask], graph_data.y[train_mask])
    train_loss.append(loss.item())
    rms = mean_squared_error(graph_data.y.cpu().detach().numpy(), out.cpu().detach().numpy(), squared=False)
    train_rms.append(rms)

    loss.backward()
    optimizer.step()

    model.eval()
    out = model(graph_data)
    out = out.squeeze()
    loss = criterion(out[val_mask], graph_data.y[val_mask])
    val_loss.append(loss.item())
    rms = mean_squared_error(graph_data.y.cpu().detach().numpy(), out.cpu().detach().numpy(), squared=False)
    val_rms.append(rms)

import matplotlib.pyplot as plt

plt.plot([i+1 for i in range(len(val_rms))], val_loss)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

model = GAT(dataset1.num_features, nhid = 8, nclass = dataset1.num_classes, dropout = 0.6, alpha = 0.1, nheads = 8).to(device)

data = dataset1[0].to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


dataloader = DataLoader(dataset, batch_size = 1)

def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss_train = F.nll_loss(output, data.y)
    optimizer.step()



for epoch in range(30):
    
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)

    loss = F.nll_loss(out, labels)
    
    if epoch%200 == 0:
        print(loss)
    
    loss.backward()
    optimizer.step()


for epoch in range(30):  
    for data in dataloader:
        node_features, edge_index, edge_features, labels = data
        node_features, edge_index, edge_features = node_features.squeeze().float(), edge_index.squeeze(), edge_features.squeeze().float()
        node_features, edge_index, edge_features = node_features.to(device), edge_index.to(device), edge_features.to(device)
        graph_data = Data(node_features, edge_index, labels)


        model.train()
        optimizer.zero_grad()
        out = model(graph_data)
        loss = F.nll_loss(out, labels)
        
        if epoch%200 == 0:
            print(loss)
        
        loss.backward()
        optimizer.step()

    break

graph_data.num_features

print(data)

print(data.edge.shape)

model.eval()
_, pred = model(data).max(dim=1)
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))


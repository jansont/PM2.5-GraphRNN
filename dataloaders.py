import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import gather_graph_data
from sklearn.preprocessing import MinMaxScaler


class StaticGraphDataset(Dataset):
    def __init__(self, 
                labels, 
                node_features,
                edge_features, 
                prenormalized = True, 
                prescaled = True):

            
        self.edge_features = torch.tensor(edge_features)
        self.labels = torch.tensor(labels)
        self.node_features = torch.tensor(node_features)

        if not prenormalized: 

            feat_mean = node_features.mean(axis=0)
            feat_sdev = node_features.std(axis=0)

            self.features = np.zeros(node_features.shape)
            for i in range(node_features.shape[1]):
                self.features[:, i] = (node_features[:, i] - feat_mean[i])/feat_sdev[i]
            self.features = torch.tensor(self.features)
            self.node_features = self.features

        if not prescaled: 
            for i in range(self.feautures.shape[0]):
                scaler = MinMaxScaler()
                self.features[i] = scaler.fit_transform(self.features[i])
            self.node_features = self.features


    def __len__(self):
        return self.node_features.shape[0]

    def __getitem__(self, idx):
        return self.node_features[idx], self.edge_features[idx], self.labels[idx]


class WeatherData(Dataset):
    def __init__(self, 
                 labels,   
                 node_features,
                 edge_features, 
                 historical_len, 
                 prediction_len, 
                 prenormalized = True, 
                 prescaled = True, 
                 ):
        
        self.historical_len = historical_len
        self.pred_len = prediction_len
        self.seq_len = historical_len + prediction_len

        self.edge_features = torch.tensor(edge_features)
        self.labels = torch.tensor(labels)

        feat_mean = node_features.mean(axis=0)
        feat_sdev = node_features.std(axis=0)

        if not prenormalized: 
            self.features = np.zeros(node_features.shape)
            for i in range(node_features.shape[1]):
                self.features[:, i] = (node_features[:, i] - feat_mean[i])/feat_sdev[i]
            self.features = torch.tensor(self.features)
        else: 
            self.features = torch.tensor(node_features)

        if not prescaled: 
            for i in range(self.feautures.shape[0]):
                scaler = MinMaxScaler()
                self.features[i] = scaler.fit_transform(self.features[i])


    def __len__(self):
        return len(self.labels - self.seq_len)

    def __getitem__(self, idx):
        features = self.features[idx: idx + self.historical_len]
        edge_features = self.edge_features[idx:idx + self.historical_len]
        labels_x = self.labels[idx: idx + self.historical_len]
        labels_y = self.labels[idx + self.historical_len: idx + self.seq_len]
        return features, edge_features, labels_x, labels_y



def get_iterators(data_file,
                  edge_cols,
                  node_cols,
                  split,
                  batch_size,
                  historical_len,
                  pred_len, 
                  dist_thresh, 
                  multi_edge_feature,
                  use_self_loops, 
                  num_workers):

    '''Returns training and validation dataloaders
    '''

    graph, graph_data = gather_graph_data(data_file,
                                          edge_cols,
                                          node_cols, 
                                          dist_thresh,
                                          multi_edge_feature,
                                          use_self_loops)

    graph_edge_features, graph_node_features, graph_labels = graph_data

    dataset = WeatherData(edge_features = graph_edge_features, 
                                    labels = graph_labels,
                                    node_features = graph_node_features,
                                    historical_len = historical_len,
                                    prediction_len = pred_len)


    train_size = int(split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


    def collate_batch(batch):
        '''Takes care of padding at end of sequence
        '''
        feature_batch = [item[0] for item in batch]
        lengths = [x.shape[0] for x in feature_batch]
        feature_batch = pad_sequence(feature_batch, batch_first=True)
        edge_batch = pad_sequence([item[1] for item in batch], batch_first=True)
        labels_x_b = pad_sequence([item[2] for item in batch])
        labels_x_b = labels_x_b.float()
        x = (feature_batch, edge_batch, labels_x_b, lengths)
        y = pad_sequence([item[3] for item in batch])       
        return x, y

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,drop_last=True,
                                                collate_fn=collate_batch, num_workers = num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers = num_workers,
                                                 drop_last=True, collate_fn=collate_batch)

    return train_dataloader, val_dataloader, graph.edge_index



def gather_static_graphs(data_file,
                        edge_cols,
                        node_cols, 
                        dist_thresh,
                        multi_edge_feature,
                        use_self_loops, 
                        pseudo_data = False, save=False):
    graph, graph_data = gather_graph_data(data_file,
                                          edge_cols,
                                          node_cols, 
                                          dist_thresh,
                                          multi_edge_feature,
                                          use_self_loops, 
                                          pseudo_data, save)

    graph_edge_features, graph_node_features, graph_labels = graph_data  
    graph_states_set = StaticGraphDataset(edge_features = graph_edge_features, 
                                    labels = graph_labels,
                                    node_features = graph_node_features)
    
    graph_states_loader = DataLoader(graph_states_set, batch_size= 1)
    return graph_states_loader, graph
    
from collections import OrderedDict
import numpy as np
import torch


def normalize(x):
    return (x - np.mean(x)) / np.std(x)

def generate_nodes(metadata):
    '''Creates ordered dict of latitudes and longitudes with station id key.'''
    stations = [s for s in metadata['STATION']]
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
    '''Use if altitude is included as node feature'''
    altitudes = [nodes[id]['altitude'] for id in nodes]
    node_features = np.array(altitudes)
    return node_features


def geo_distance(first_node, second_node):
        '''
        Computes Haversine formula for geodesic distance
        between two nodes
        Args: 
            first_node :: tuple (float)
                Latitude and longitude of node 1
            second_node :: tuple (float)
                Latitude and longitude of node 2
        Returns: 
            d :: float
                Geodesic distance between two nodes
        '''
        
        lat1, long1 = first_node
        lat2, long2 = second_node
        R = 6371e3 #m
        lat1 = lat1 * np.pi/180; #rad
        lat2 = lat2 * np.pi/180; #rad
        delta_lat = (lat2 - lat1) * np.pi/180
        delta_long = (long1 - long2) * np.pi/180
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

def add_self_loops_to_sparse_adj(edge_idx, n):
    ''' Add self loop to sparse adjacency list (Each node connects to itself)'''
    source_w_self_loop = np.append(edge_idx[0], [i for i in range(n)])    #add self loops
    target_w_self_loop = np.append(edge_idx[1], [i for i in range(n)])
    edge_idx = np.array([source_w_self_loop, target_w_self_loop])
    order = edge_idx[0].argsort()
    edge_idx[0].sort()
    edge_idx[1] = edge_idx[1][order] 
    return(edge_idx) 

def ReLU(x):
    return x * (x>0)


class Graph():
    def __init__(self,
                 graph_metadata,
                 edge_data, 
                 distance_threshold, 
                 multi_edge_feature, 
                 use_self_loops, 
                 ):
        '''
        Creates graph with all the necessary matrices.
        Args: 
            graph_metadata :: pd.DataFrame 
                Dataframe of metadata with PM station. Must include latitude
                and longitude of each station. 
            stations :: list [str]
                List of station ids to include in graph. Corresponds to stations which 
                have data on the desired time frame. 
            edge_data :: np.array (dates, num nodes, 2)
                Wind vector in x and y for all nodes and timesteps. 
            distance_threshold :: float
                Threshold distance after which two nodes are considered not connected. 
            multi_edge_feature :: bool
                Generate multi-dimensional edge feature (windx, windy, dx, dy) if True
                Generate scalar edge feature (convection) if False
            use_self_loops: bool 
                Include edges in which source and target node are the same. 
        '''
        self.distance_threshold = distance_threshold
        self.edge_data = edge_data    
        self.multi_edge_feature = multi_edge_feature
        self.use_self_loops = use_self_loops
        self.graph_metadata = graph_metadata #get coordinates of desired stations
        self.nodes = generate_nodes(self.graph_metadata) #ordered dict of lat, long with station key
        self.size = len(self.nodes) 
        self.edge_index, self.edge_attr = self.generate_edges()     
        self.adjacency_matrix = self.get_adjacency_matrix()         
        # self.edge_adjacency = self.get_edge_adjacency()
        # self.transformation_matrix = self.get_transformation_matrix()

    def generate_edges(self):
        '''
        Generates sparse adjacency matrix and edge features. 
        Returns: 
            Edge index: np.array (2 * num edges)
                Sparse adjacency matrix with vertically stacked [source nodes, target nodes]
            Edge weight: np.array (days, num edges)
                Scalar edge weight (convection) if self.multi_edge_feature = False
            Edge weight: np.array (days, num edges, 4)
                Vector edge feature [wind_x, wind_y, dx, dy] if self.multi_edge_feature = True
        '''
        nodes = self.nodes            
        node_list = list(self.nodes)        #get list of node ids
        coordinates = list(zip(self.graph_metadata['Latitudes'], self.graph_metadata['Longitudes'])) 
        #square matrix of distance to all other nodes
        distance_matrix = generate_node_distances(coordinates)  
        adj_matrix = np.zeros([self.size, self.size])
        #in adj matrix, set entry to 1 if distance between nodes below threshold
        adj_matrix[distance_matrix < self.distance_threshold] = 1 
        distance_matrix = distance_matrix * adj_matrix
        #edge_dist: same shape as above, distance values between those node indices
        #edge_idx : shape (2 * number of connected nodes (dis < threshold)) 
        edge_idx, edge_dist = sparse_adjacency(torch.tensor(distance_matrix)) 
        edge_idx, edge_dist = edge_idx.numpy(), edge_dist.numpy()             
        if self.use_self_loops: 
            edge_idx = add_self_loops_to_sparse_adj(edge_idx, len(node_list))
            edge_idx = add_self_loops_to_sparse_adj(edge_idx, len(node_list))

        windx, windy, dx, dy = [],[],[],[]
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
            distance_y = geo_distance((0, long0), (0, long1))
            #get wind along x and y vectors from both source and edge
            wind_source_x = self.edge_data[:, source_idx, 0]
            wind_dest_x = self.edge_data[:, dest_idx, 0]
            wind_source_y = self.edge_data[:, source_idx, 1]
            wind_dest_y = self.edge_data[:, dest_idx, 1]
            #average source and edge wind components to get net wind 
            wind_x = (wind_source_x + wind_dest_x)/2
            wind_y = (wind_source_y + wind_dest_y)/2
            windx.append(wind_x), windy.append(wind_y)
            dx.append(distance_x), dy.append(distance_y)

        wx = np.stack(windx).transpose()
        wy = np.stack(windy).transpose()
        dx = np.tile(np.stack(dx), [wx.shape[0],1])
        dy = np.tile(np.stack(dy), [wx.shape[0],1])


        #multidimensional edge feature (days, num_edges, 4)
        if self.multi_edge_feature: 
            #return normalized distance and wind vectors 
            dx = normalize(dx)
            dy = normalize(dy)
            wx = normalize(wx)
            wy = normalize(wy)
            edge_vectors = np.stack([wx, wy, dx, dy], axis = -1)
            return edge_idx, edge_vectors

        else: 
            #return convection coefficient
            distance = (dy**2 + dx**2)**0.5
            wind = (wx**2 + wy**2)**0.5
            theta_position = np.tan(dy + 1e-12 / (dx + 1e-12))
            theta_wind = np.tan(wy + 1e-12 / (wx + 1e-12))
            delta_angle = np.abs(theta_position- theta_wind)
            #convection value
            edge_weight = (wind * np.cos(delta_angle) / (distance+1e-3))
            edge_weight = ReLU(edge_weight)
            edge_weight = normalize(edge_weight)
            return edge_idx, edge_weight

        
    def edge_list_to_adj(self):
        '''Convert sparse adjacency list (2 x num_edge)
         to adjacency matrix (num nodes x num nodes)'''
        adj = np.identity(self.edge_index.size)
        for k,(i,j) in enumerate(zip(self.edge_index[0], self.edge_index[1])):
            adj[i,j] = self.edge_attr[j]
        return adj

    def edge_list_sequence_to_adj(self):
        '''
        Convert sparse adjacency list (2 x num_edge) to edge weighted adjacency
        matrix (num nodes x num nodes) for all time steps. 
        '''
        adjacencies = []
        for i in range(self.edge_attr.shape[0]):
            adj = np.identity(self.size)
            for k,(i,j) in enumerate(zip(self.edge_index[0], self.edge_index[1])):
                adj[i,j] = self.edge_attr[i][k]
            adjacencies.append(adj)
        return np.array(adjacencies)

    def get_edge_adjacency(self):
        '''
        Generates edge adjacency matrix where edges replace nodes. 
        Shape: (num_edge x num_edge)[i,j] = 1 if edge i connect to edge j through a node
                                          = 0 otherwise 
        '''
        adj_e = np.zeros([self.edge_index.shape[1],self.edge_index.shape[1]])
        for i,s in enumerate(self.edge_index[0]):
            t = self.edge_index[1][i]
            adj_e[i,t] = 1
        assert adj_e.all() == adj_e.transpose().all()
        return adj_e

    def get_adjacency_matrix(self):
        '''
        Generates adjacency matrix with shape: (num nodes x num nodes) = 1
        if nodes are connected, 0 otherwise (distance below thresh)
        '''
        nodes = self.nodes            
        node_list = list(self.nodes)        #get list of node ids
        coordinates = list(zip(self.graph_metadata['Latitudes'], self.graph_metadata['Longitudes']))  #zip latitude and longitude of all nodes
        distance_matrix = generate_node_distances(coordinates)  #square matrix of distance to all other nodes

        adj = np.identity(self.size)
        adj[distance_matrix < self.distance_threshold] = 1 #in adj matrix, set entry to 1 if distance between nodes below threshold
        assert adj.all() == adj.transpose().all()
        return adj

    def get_transformation_matrix(self):
        '''
        Transformation matrix of shape [num nodes x num edges]
        T[i,j] = 1 if edge j connects to node i
        '''
        T = np.zeros([self.size, self.edge_adjacency.shape[0]])
        for i,source in enumerate(self.edge_index[0]):
            for j,t in enumerate(self.edge_index[1]): 
                if self.edge_index[0][j]  == source:        
                    T[source, t] = 1      
        assert T.all() == T.transpose().all()
        return T

import datetime
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from graph import Graph


def get_date_range(file):
    if file == 'LA_DATA_2018_02_to_2018_06.csv':
        start = date(2018, 2, 1)
        end = date(2018, 6, 8)
        date_range = pd.date_range(start, end-timedelta(days=1))
        date_range = [str(x)[:10] for x in date_range]
        return date_range
    else:
        print('wrong file name.')


def data_to_numpy(weather_data, edge_cols, node_cols, date_range, pseudo_data=False, save=True):
    '''Converts pandas whether data to np array. 
    Args: 
        weather_data :: pd.DataFrame
            Dataframe containing weather data from various stations and times
        edge_cols :: list [str]
            List of column names to be used as edge features (ie: wind)
        node_cols :: list [str]
            List of column names to be used as node features
        stations :: list [str]
            List of station ids to select stations which have data in desired date range. (Sorted)
        date_range :: list [str]
            List of dates to select from weather data. 
    '''
    if pseudo_data:
        checkpt = 'pseudo_checkpt'
    else:
        checkpt = 'checkpt'

    if not save or not os.path.exists(checkpt):
        print('Checkpt doesnt exist, making it')
        if save:
            os.makedirs(checkpt)

        stations = weather_data['STATION'].unique()
        stations.sort()

        present_dates = sorted(weather_data['DATE'].unique(
        ), key=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))

        graph_node_features = np.empty(
            (len(date_range), len(stations), len(node_cols)))
        graph_edge_features = np.empty(
            (len(date_range), len(stations), len(edge_cols)))
        graph_labels = np.empty((len(date_range), len(stations)))
        stations.sort()



        for day_idx in range(len(date_range)):

            if date_range[day_idx] not in present_dates: 
                earliest_date_with_data = day_idx
                while date_range[earliest_date_with_data] not in present_dates:
                    earliest_date_with_data -= 1
                next_day_with_data = day_idx
                while date_range[next_day_with_data] not in present_dates:
                    next_day_with_data += 1

                x = [date_range[earliest_date_with_data], date_range[next_day_with_data]]
                vals = weather_data[weather_data['DATE'].isin(x)]
            
            else: 
                vals = weather_data[weather_data['DATE'] == date_range[day_idx]]

            for station_idx in range(len(stations)):
                # crop dataframe to desired date and station
                vals = vals[vals['STATION'] == stations[station_idx]]


                if date_range[day_idx] not in present_dates: 
                    if len(vals.index) == 0 :
                        node_vals = weather_data[weather_data['DATE'] ==
                                                date_range[earliest_date_with_data]][node_cols].mean().values
                        edge_vals = weather_data[weather_data['DATE'] ==
                                                date_range[earliest_date_with_data]][edge_cols].mean().values
                        pm = weather_data[weather_data['DATE'] ==
                                        date_range[earliest_date_with_data]]['pm25'].mean()  

                    else:
                        vals = vals[vals['STATION'] == stations[station_idx]]
                        node_vals  = vals[node_cols].mean().values
                        edge_vals = vals[edge_cols].mean().values
                        pm = vals['pm25'].mean()
                        
                        if np.isnan(node_vals).any():
                            print(len(vals.index))
                            print(vals)
                            assert False  


                else: 

                    pm = vals['pm25'].values  # get pm
                    # crop out edge features
                    edge_vals = vals[edge_cols]
                    edge_vals = np.array(edge_vals.values.tolist()).flatten()
                    # crop out node features
                    node_vals = vals[node_cols]
                    # node features as array
                    node_vals = np.array(node_vals.values.tolist()).flatten()

                    # certain stations have missing data on a given day, fill with geo mean
                    if len(node_vals) == 0:
                        node_vals = weather_data[weather_data['DATE'] ==
                                                date_range[day_idx]][node_cols].mean().values

                    if len(pm) == 0:
                        pm = weather_data[weather_data['DATE'] ==
                                        date_range[day_idx]]['pm25'].mean()

                    if len(edge_vals) == 0:
                        edge_vals = weather_data[weather_data['DATE'] ==
                                                date_range[day_idx]][edge_cols].mean().values       

                graph_labels[day_idx, station_idx] = pm
                graph_node_features[day_idx, station_idx] = node_vals
                graph_edge_features[day_idx, station_idx] = edge_vals

        if save:
            print('Creating checkpoint')
            np.save(os.path.join(checkpt, 'graph_node_features'),
                    graph_node_features)
            np.save(os.path.join(checkpt, 'graph_edge_features'),
                    graph_edge_features)
            np.save(os.path.join(checkpt, 'graph_labels'), graph_labels)

    else:
        print('Found Checkpoint, loading')
        graph_node_features = np.load(
            os.path.join(checkpt, 'graph_node_features.npy'))
        graph_edge_features = np.load(
            os.path.join(checkpt, 'graph_edge_features.npy'))
        graph_labels = np.load(os.path.join(checkpt, 'graph_labels.npy'))

    return graph_node_features, graph_edge_features, graph_labels


def gather_graph_data(weather_file,
                      edge_cols,
                      node_cols,
                      dist_thresh,
                      multi_edge_feature,
                      use_self_loops,
                      pseudo_data=False,
                      save=True):
    '''
    Read csv weather, aod, pm and meta data from file. Convert to np. Create graph.
    Args: 
        weather_file : name of file containing all data
        edge_cols : list [str]
            name of columns to be edge features
        node_cols : list [str]
            name of columns to be node features
    '''
    path = './data/'
    if isinstance(weather_file, Path):
        weather_data = pd.read_csv(weather_file)
    else:
        weather_data = pd.read_csv(path+weather_file)

    date_range = get_date_range(weather_file)

    stations = np.random.choice(weather_data['STATION'].unique(), 16)
    weather_data = weather_data.loc[weather_data['STATION'].isin(stations)]
    # convert to np
    graph_node_features, graph_edge_features, graph_labels = data_to_numpy(
        weather_data, edge_cols, node_cols, date_range, pseudo_data, save)
    # build graph
    metadata = weather_data[['STATION', 'Latitudes',
                             'Longitudes']].drop_duplicates()
    metadata = metadata.reset_index(drop=True).sort_values('STATION')
    graph = Graph(metadata,
                  graph_edge_features,
                  dist_thresh,
                  multi_edge_feature,
                  use_self_loops)
    graph_edge_features = graph.edge_attr
    graph_data = graph_edge_features, graph_node_features, graph_labels
    return graph, graph_data


def assign_id(weather_data):
    ids = np.zeros(len(weather_data['Latitudes']))
    node_pos = list(weather_data.groupby(['Latitudes', 'Longitudes']).groups)
    for i, node in enumerate(node_pos):
        indices_lats = weather_data['Latitudes'].values == node_pos[i][0]
        indices_longs = weather_data['Longitudes'].values == node_pos[i][1]
        indices = indices_lats & indices_longs
        ids[np.argwhere(indices)] = int(i+1)
    weather_data['STATION'] = ids
    return weather_data

# ECSE552_Project

1 Accurate prediction of Particulate Matter (PM2.5) values can better inform public
2 to take protective actions on daily basis. However, no accurate predictor for ground-
3 level PM2.5 values exists. Additionally, the information needed to predict PM2.5 is
4 not well understood. This work is part of a NASA’s challenge to develop highly
5 accurate PM2.5 predictor. In order to build models, a comprehensive exploration
6 of available data from different sources was conducted which led to the selection
7 of both satellite and weather information. Three baseline models namely Naive
8 One Step Lookbehind, FNN, and RNN were developed. The proposed models are
9 Edge-GNN which used a GRU and an edge-weighted graph model, and a GCN
10 with GRU to generate PM2.5 values. The results showed similar, but slightly better
11 performance for the GCN model (percent error: 22.84%) over the Edge-GNN
12 (percent error: 24.7%). Overall, we find better performance from our proposed
13 models when compared to the baselines.

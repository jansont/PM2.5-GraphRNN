import numpy as np
import pandas as pd
from scipy import stats
from weather_data_cleaning_utilities import clean_all_fields


def group_dates(data):
    '''
    Weather data contains hourly samples for each day. Group samples
    into a dict of dataframe of hourly samples for each day. 
    Args: 
        data: weather dataframe
    Returns: 
        dict of weather dataframes with (key, val) = (date, dataframe) 
    '''
    date_indices = {}
    data = data.sort_values(by = ['DATE'])
    dates = data['DATE']
    current_date = dates.iloc[0][0:10]
    date_indices[current_date] = [0]
    #group data from same date into date-keyed dict with list
    # of corresponding indices in original df
    for i,row in enumerate(dates):
        date = row[0:10]
        if date != current_date:
            current_date = date
            date_indices[date] = []
        date_indices[date].append(i) 
    date_grouped_dfs = {}
    #for each date, slice the original df to date-keyed dict of dfs
    for key, value in date_indices.items():
        date_df = data.iloc[value]
        date_df = date_df.reset_index(drop = True)
        date_grouped_dfs[key] = date_df
    return date_grouped_dfs
        

def mean_remove_nan(series):
    '''Removes nan then takes mean of pd series'''
    array = np.array(series)
    array = array[~np.isnan(array)]
    mean = np.mean(array)
    return mean

def time_weighted_aggregate(field, day_minus1, day, day_plus1, center, radius, method = 'mean'):
    '''Does time delta weighted aggregate of pandas series
    Args: 
        field: string representing dataframe column to aggregate
        day_minus1: dataframe containing yesterday's data (in case time radius extends into previous day)
        day: dataframe containing today's data
        day_plus1: dataframe containing tomorrow's data (in case time radius extends into tomorrow)
        center: center of from which to compute time delta, pd.datetime object
        radius: number of hours into past and future for weighted aggregate
        method: 'mean' for delta time weighted mean, 'mode' used for categorical data (ie: cloud codes)
    Returns: 
        aggregate: time weighted aggregate of specified field
    '''
    argcenter = np.argmin([(center - time).seconds / 3600 for time in day['Datetime']])
    day = day.sort_values(by='Datetime', ascending=True)
    #handle edge case (trim radius to start of day)
    if day_minus1 is None or day_plus1 is None:
        radius1 = argcenter
        radius2 = len(day) - argcenter
        radius = np.min([radius1, radius2])
        times = list(day['Datetime'])
        values = list([day[field]])
    else: 
        day_minus1 = day_plus1.sort_values(by='Datetime', ascending=True)
        day_plus1 = day_plus1.sort_values(by='Datetime', ascending=True)
        times = list(day_minus1['Datetime']) + list(day['Datetime']) + list(day_plus1['Datetime'])
        values = list(day_minus1[field]) + list(day[field]) + list(day_plus1[field])
        argcenter += len(day_minus1[field])
    #get weight of time difference between center and neighbours
    values = list(np.array(values).flatten())
    timedeltas = [((times[i] - center).seconds / 3600) for i in range(argcenter-radius, argcenter+radius)]
    timedelta_weights = np.array([float(dt)/sum(timedeltas) for dt in timedeltas])
    values = np.array(values[argcenter-radius : argcenter+radius])
    timedelta_weights = timedelta_weights[~np.isnan(values)]
    #remove nan
    values = values[~np.isnan(values)]
    if method == 'mode':
        aggregate = stats.mode(values)[0]
    else:
        aggregate = sum(timedelta_weights*values)
    return aggregate


def remove_nan_and_aggregate(series, method = 'mean'):
    series = series[~np.isnan(series)]
    if method == 'mode':
        aggregate = stats.mode(a)[1][0][0]
    else: 
        aggregate = series.mean()
    return aggregate



def aggregate_daily_station_data(df_dict, pm_station_measurement_times, radius): 
    '''Takes daily weather data dict and returns a time weighted aggregate dataframe for that weather station
    Args: 
        df_dict: dict of daily weather dataframes
    Returns: 
        Pandas dataframe of daily averages 
    '''
    avg_dict = {
        'Date': [], 
        'Wind_X': [],
        'Wind_Y': [],
        'Cloud_Height':[],
        'Temperature':[],
        'Visibility':[],
        'Atmospheric_Pressure':[],
        'Dew_Point':[],
        'Precipitation_Duration_1':[],
        'Precipitation_Depth_1':[],
        'Precipitation_Duration_2':[],
        'Precipitation_Depth_2':[],
        'Precipitation_Duration_3':[],
        'Precipitation_Depth_3':[],
        'Cloud_Coverage_1':[], 
        'Cloud_Base_Height_1':[], 
        'Cloud_Type_Code_1':[], 
        'Cloud_Coverage_2':[], 
        'Cloud_Base_Height_2':[], 
        'Cloud_Type_Code_2':[], 
        'Total_Coverage':[], 
        'Total_Opaque_Coverage':[], 
        'Lowest_Cloud_Height':[], 
        'Low_Cloud_Genus':[], 
        'Mid_Cloud_Genus':[], 
        'High_Cloud_Genus':[], 
    }
    #clean data first so that cleaned data from past is accessible by current iterated date
    for _, df in df_dict.items():
        df = clean_all_fields(df)

    for day in df_dict:
    
        df = df_dict[day]

        avg_dict['Date'].append(day)

        wnd_x = remove_nan_and_aggregate(df['Wind_X'])
        wnd_y = remove_nan_and_aggregate(df['Wind_Y'])
        avg_dict['Wind_X'].append(wnd_x)
        avg_dict['Wind_Y'].append(wnd_y)

        height = remove_nan_and_aggregate(df['Cloud_Height'])
        avg_dict['Cloud_Height'].append(height)

        vis = remove_nan_and_aggregate(df['Visibility'])
        avg_dict['Visibility'].append(vis)

        temp = remove_nan_and_aggregate(df['Temperature'])
        avg_dict['Temperature'].append(temp)
          
        pressure = remove_nan_and_aggregate(df['Atmospheric_Pressure'])
        avg_dict['Atmospheric_Pressure'].append(pressure)

        dp = remove_nan_and_aggregate(df['Dew_Point'])
        avg_dict['Dew_Point'].append(dp)
        
        for e in range(1,4):
            dur = remove_nan_and_aggregate(df[f'Precipitation_Duration_{e}'])
            depth = remove_nan_and_aggregate(df[f'Precipitation_Depth_{e}'])
            avg_dict[f'Precipitation_Depth_{e}'].append(depth), avg_dict[f'Precipitation_Duration_{e}'].append(dur)

        for e in range(1,3):
            cover = remove_nan_and_aggregate(df[f'Cloud_Coverage_{e}'])
            base_height = remove_nan_and_aggregate(df[f'Cloud_Base_Height_{e}'])
            cloud_code = remove_nan_and_aggregate(df[f'Cloud_Type_Code_{e}'], method = 'mode')
            avg_dict[f'Cloud_Coverage_{e}'].append(cover), avg_dict[f'Cloud_Base_Height_{e}'].append(base_height)
            avg_dict[f'Cloud_Type_Code_{e}'].append(cloud_code)

        total_coverage = remove_nan_and_aggregate(df['Total_Coverage'])
        avg_dict['Total_Coverage'].append(total_coverage)
        toc = remove_nan_and_aggregate(df['Total_Opaque_Coverage'])
        avg_dict['Total_Opaque_Coverage'].append(toc)
        lch = remove_nan_and_aggregate(df['Lowest_Cloud_Height'])
        avg_dict['Lowest_Cloud_Height'].append(lch)
        lcg = remove_nan_and_aggregate(df['Low_Cloud_Genus'], method = 'mode')
        avg_dict['Low_Cloud_Genus'].append(lcg)
        mcg = remove_nan_and_aggregate(df['Mid_Cloud_Genus'], method = 'mode')
        avg_dict['Mid_Cloud_Genus'].append(mcg)  
        hcg = remove_nan_and_aggregate(df['High_Cloud_Genus'], method = 'mode')
        avg_dict['High_Cloud_Genus'].append(hcg)

    return pd.DataFrame(avg_dict)


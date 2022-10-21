import numpy as np 
import pandas as pd

'''
Documentation for data: 
    Documentation of NOAA data:  
    https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc%3AC00532/html
'''

def clean_weather_data_columns(data, remove_cols):
    #TO-DO: Explore what the columns mean and what can be removed, 
    # and if columns with much missing data are important
    '''
    Removes irrelevant columns and columns with many missing values
    for historical weather dataset. 
    Args: 
        data: weather dataframe
        remove_cols: columns to remove
    Returns: 
        Cleaned dataframe
    '''
    data = data.drop(remove_cols, axis = 1)
    to_drop = []
    for col in data:
        if data[col].isna().sum() > 0.7 * len(col):
            data = data.drop(to_drop, axis = 1)
    return data


def clean_wind_data(data):
    '''
    Parses wind data string into relevant fields. Creates new columns in weather dataframe
    Removes samples with corrupted wind data, or missing wind data. 
    'Wind direction: 0-360 degs
    'Wind speed: m/s
    '''
    wind_direction, wind_speed = [], []
    acceptable_quality = (0,1,4,5,9) #see docs
    missing_dir, missing_spd = 999, 9999
    for i,values in enumerate(data['WND']):
        values = values.split(",")
        direction, direction_quality = int(values[0]), int(values[1])
        speed, speed_quality = int(values[3]), int(values[4])
        #ID erroneuous or missing.
        if (direction_quality not in acceptable_quality or direction == missing_dir):
            direction = np.nan
        if (speed_quality not in acceptable_quality or speed == missing_spd):
            speed = np.nan
        wind_direction.append(direction), wind_speed.append(speed)
    data['Wind_Speed'] = wind_speed
    data['Wind_Direction'] = wind_direction
    return data

def clean_ceiling_height_data(data):
    '''
    The height above ground level (AGL) of the lowest cloud or obscuring
    phenomena layer aloft with 5/8 or more summation total sky cover,
    which may be predominantly opaque, or the vertical visibility into a
    surface-based obstruction. Unlimited = 22000.
    height (m) above ground level of lowest cloud (unlimited = 22000)
    '''
    ceiling_height = []
    acceptable_quality = (0,1,4,5,9) #see docs
    missing = 99999
    for i,cig in enumerate(data['CIG']):
        cig = cig.split(",")
        height = int(cig[0])
        quality = int(cig[1])
        #ID erroneuous or missing. 
        if (quality not in acceptable_quality or height == missing):
            height = np.nan
        ceiling_height.append(height)
    data['Cloud_Height'] = ceiling_height
    return data

def clean_visibility_data(data):
    '''The horizontal distance at which an object can be seen and identified. (meters)
    '''
    visilibilites = []
    acceptable_quality = (0,1,4,5,9) #see docs
    missing = 999999
    for i,vis in enumerate(data['VIS']):
        vis = vis.split(',')
        visibility = int(vis[0])
        quality = int(vis[1])
        #ID erroneuous or missing. 
        if (quality not in acceptable_quality or visibility == missing):
            visibility = np.nan
        visilibilites.append(visibility)
    data['Visibility'] = visilibilites
    return data

def clean_temperature_data(data): 
    '''Air Temperature data in C'''
    temperatures = []
    missing = 9999
    acceptable_quality = ('0','1','4','5','9','C','I','M','P','R','U') #see docs
    for i,sample in enumerate(data['TMP']):
        sample = sample.split(',')
        temperature = int(sample[0])
        quality = sample[1]
        #ID erroneuous or missing. 
        if (quality not in acceptable_quality or temperature == missing):
            temperature = np.nan
        temperatures.append(temperature / 10)
    data['Temperature'] = temperatures
    return data

def clean_pressure_data(data): 
    '''The air pressure relative to Mean Sea Level (MSL).
    (Hectopascals)'''
    pressures = []
    missing = 99999
    acceptable_quality = ('0','1','4','5','9') #see docs
    for i,sample in enumerate(data['SLP']):
        sample = sample.split(',')
        pressure = int(sample[0])
        quality = sample[1]
        #ID erroneuous or missing. 
        if (quality not in acceptable_quality or pressure == missing):
            pressure = np.nan
        pressures.append(pressure)
    data['Atmospheric_Pressure'] = pressures
    return data

def clean_dew_point_data(data): 
    '''The temperature to which a given parcel of air must be cooled
     at constant pressure and water vapor content in order for saturation to occur. (C)'''
    dew_points = []
    missing = 9999
    acceptable_quality = ('0','1','4','5','9','C','I','M','P','R','U') #see docs
    for i,sample in enumerate(data['DEW']):
        sample = sample.split(',')
        dp = int(sample[0])
        quality = sample[1]
        #ID erroneuous or missing. 
        if (quality not in acceptable_quality or dp == missing):
            dp = np.nan
        dew_points.append(dp)
    data['Dew_Point'] = dew_points
    return data

def clean_precipitation_data(data, event_number): 
    '''episode of LIQUID-PRECIPITATION.
    - The quantity of time over which the LIQUID-PRECIPITATION was measured. (hours)
    - The depth of LIQUID-PRECIPITATION that is measured at the time of an observation. (mm)
    Note that there data contains AA1-AA3 fields for multiple precipitation events
    '''
    times, depths = [], []
    missing_depth, missing_time = 9999, 99
    acceptable_quality = ('0','1','4','5','9','C','I','M','P','R','U') #see docs
    event_column = f'AA{event_number}'
    for i,sample in enumerate(data[event_column]):
        if isinstance(sample, str):
            sample = sample.split(',')
            time = int(sample[0])
            depth = int(sample[1])
            quality = sample[-1]
        else: 
            time = depth = quality = np.nan
        #ID erroneuous or missing. 
        if (quality not in acceptable_quality):
            depth = time = np.nan
        if (depth == missing_depth):
            depth = np.nan
        if (time == missing_time):
            time = np.nan
        depths.append(depth),  times.append(time)
    data[f'Precipitation_Duration_{event_number}'] = times
    data[f'Precipitation_Depth_{event_number}'] = depths
    return data


def clean_sky_cover_data(data, event_number): 
    '''SKY-COVER-LAYER..
    - Field 1: The code that denotes the fraction of the total celestial dome covered by a SKY-COVER-LAYER.
    - Field 2: SKY-COVER-LAYER base height dimension
    - Field 3: The code that denotes the classification of the clouds that comprise a SKY-COVER-LAYER.
    Note that there data contains GA1-GA6 fields for multiple cloud layers
    GA2-GA3 are 89+% nan (no secondary cloud covered), so these are ignored 
    TO-DO: Check above statement on total dataset
    T0-DO: compare cloud cover fields and choose appropriate one(s) (GA1, GD1, GF1, )
    '''
    covers, height, clouds = [], [], []
    acceptable_quality = ('0','1','4','5','9','M') #see docs
    missing_cover = missing_cloud_type = 99
    missing_base_height = 99999
    #convert octas (or code) to coverage fraction
    conversion_values = {0:0, 1:0.1, 2:0.25, 3:0.4, 4:0.5, 5:0.6, 6:0.75, 7:0.95, 8:1.0, 9:np.nan, 10:np.nan, 99:np.nan}
    event_column = f'GA{event_number}'
    for i,sample in enumerate(data[event_column]):
        if isinstance(sample, str):
            sample = sample.split(',')
            coverage, coverage_quality = int(sample[0]), sample[1] #fraction
            coverage = conversion_values[coverage]
            base_height, base_height_quality = int(sample[2]), sample[3]
            cloud_type, cloud_type_quality = int(sample[4]), sample[5]
        else: 
            coverage = base_height = cloud_type = coverage_quality = base_height_quality = cloud_type_quality = np.nan
        if (coverage_quality not in acceptable_quality or coverage == missing_cover):
            coverage = np.nan
        if (base_height_quality not in acceptable_quality or base_height == missing_base_height):
            base_height = np.nan
        if (cloud_type_quality not in acceptable_quality or cloud_type == missing_cloud_type):
            cloud_type = np.nan
        covers.append(coverage),  height.append(base_height), clouds.append(cloud_type)
    data[f'Cloud_Coverage_{event_number}'] = covers
    data[f'Cloud_Base_Height_{event_number}'] = height
    data[f'Cloud_Type_Code_{event_number}'] = clouds
    return data


def clean_sky_condition_observation(data):
    '''SKY-CONDITION-OBSERVATION
    fields: 
    - total coverage code
    - total opaque coverage code
    - quality total coverage code
    - total lowest cloud cover code
    - quality total lowest cloud cover code 
    - low cloud genus code
    - quality low cloud genus code
    - lowest cloud base height dimension 
    - lowest cloud base height quality code 
    - mid cloud genus code 
    - quality mid cloud genus code
    - high cloud genus code 
    - quality high cloud genus code
    '''
    coverage, opaque_coverage, lowest_cloud_cover, low_clouds, mid_clouds, high_clouds = [],[],[],[],[],[]
    acceptable_quality = (0, 1, 4, 5, 9)
    missing_coverage, missing_cloud_genus, missing_height = 99, 99, 99999
    conversion_values = {0:0, 1:0.1, 2:0.25, 3:0.4, 4:0.5, 5:0.6, 6:0.75, 7:0.95, 8:1.0, 9:np.nan, 10:np.nan,
                        11:0.3, 13:0.3, 14:0.4, 15:0.5, 16:0.6, 17:0.8, 18:0.9, 19:1.0, 99:np.nan}
    for i,sample in enumerate(data['GF1']):
        if isinstance(sample, str):
            sample = sample.split(',')
            total_coverage, total_coverage_quality = int(sample[0]), sample[2]
            total_coverage = conversion_values[total_coverage]
            total_opaque_coverage = int(sample[1])
            total_opaque_coverage = conversion_values[total_opaque_coverage]
            total_lowest_cloud_cover, total_lowest_cloud_cover_quality = int(sample[3]), int(sample[4])
            total_lowest_cloud_cover = conversion_values[total_lowest_cloud_cover]
            low_cloud_genus, low_cloud_genus_quality = int(sample[5]), int(sample[6])
            lowest_cloud_height, lowest_cloud_height_quality = sample[7], int(sample[8])
            mid_cloud_genus, mid_cloud_genus_quality = int(sample[9]), int(sample[10])
            high_cloud_genus, high_cloud_genus_quality = int(sample[11]), int(sample[12])
        else: 
            total_coverage = total_opaque_coverage = total_lowest_cloud_cover = low_cloud_genus = np.nan
            mid_cloud_genus = lowest_cloud_height = high_cloud_genus = np.nan
            total_coverage_quality = total_lowest_cloud_cover_quality = low_cloud_genus_quality = np.nan
            mid_cloud_genus_quality = lowest_cloud_height_quality = high_cloud_genus_quality = np.nan
        if (total_coverage_quality not in acceptable_quality or total_coverage == missing_coverage):
            total_coverage = np.nan
        if (total_lowest_cloud_cover_quality not in acceptable_quality or total_lowest_cloud_cover == missing_coverage):
            total_coverage = np.nan
        if (low_cloud_genus_quality not in acceptable_quality or low_cloud_genus == missing_coverage):
            low_cloud_genus = np.nan
        if (mid_cloud_genus_quality not in acceptable_quality or mid_cloud_genus == missing_coverage):
            mid_cloud_genus = np.nan
        if (high_cloud_genus_quality not in acceptable_quality or high_cloud_genus == missing_coverage):
            high_cloud_genus = np.nan
        coverage.append(total_coverage), opaque_coverage.append(total_opaque_coverage)
        lowest_cloud_cover.append(total_lowest_cloud_cover), low_clouds.append(low_cloud_genus)
        mid_clouds.append(mid_cloud_genus), high_clouds.append(high_cloud_genus)

    data['Total_Coverage'] = coverage
    data['Total_Opaque_Coverage'] = total_opaque_coverage
    data['Lowest_Cloud_Height'] = lowest_cloud_cover
    data['Low_Cloud_Genus'] = low_clouds
    data['Mid_Cloud_Genus'] = mid_clouds
    data['High_Cloud_Genus'] = high_clouds

    return data

def clean_date(data):
    data['Datetime'] = pd.to_datetime(data['DATE'])
    return data

def get_wind_vector(df):
    '''
    Args: 
        wind_data: dataframe with wind direction and speed cols
    Returns: 
        tuple of mean x,y wind vectors
    '''
    #data is calibrated to north as 0 deg ?
    wind_data = df[['Wind_Direction', 'Wind_Speed']]
    wind_vect_x, wind_vect_y = [], []
    for dir,speed in wind_data.itertuples(index=False):
        #degrees to vectors
        wind_vect_x.append(np.cos(dir + 90) * speed) #positive = west
        wind_vect_y.append(np.sin(dir + 90) * speed) #positive = north
    df['Wind_X'] = wind_vect_x    
    df['Wind_Y'] = wind_vect_y
    return df

def clean_all_fields(df):
    '''Applies all the data cleaning function
    '''
    df = clean_date(df)
    df = clean_wind_data(df)
    df = get_wind_vector(df)
    df = clean_ceiling_height_data(df)
    df = clean_visibility_data(df)
    df = clean_temperature_data(df)
    df = clean_pressure_data(df)
    df = clean_dew_point_data(df)
    df = clean_precipitation_data(df, event_number = 1)
    df = clean_precipitation_data(df, event_number = 2)
    df = clean_precipitation_data(df, event_number = 3)
    df = clean_sky_cover_data(df, event_number = 1)
    df = clean_sky_cover_data(df, event_number = 2)
    df = clean_sky_condition_observation(df)
    return df
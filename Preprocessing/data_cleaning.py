import os
import pandas as pd 
from weather_aggregate_utilities import group_dates, aggregate_daily_station_data


def clean_location_data(weather_path, pm_path, write_path):
    location_files = os.listdir(read_path)
    for file in location_files: 
        weather_df = pd.read_csv(file)
        df_dict = group_dates
        pm_data = [df['Datetime'].iloc[8] for _,df in df_dict.items()] #test variable







def main():
    parser = ArgumentParser()
    parser.add_argument("read_path", type=str, required=True)
    parser.add_argument("write_path", type=str, required=True)
    args = parser.parse_args()


    args = sys.argsv
    weather_df = pd.read_csv(path1)
    df_dict = group_dates(weather_df)

    # pm_data = pd.read_csv(path2)
    pm_data = [df['Datetime'].iloc[8] for _,df in df_dict.items()] #test variable

    aggregate_daily_station_data(df_dict, pm_data, radius = 6)

    pd.to_csv(path3)

if __name__ == 'main':
    main()
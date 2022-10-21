# https://registry.opendata.aws/noaa-gfs-bdp-pds/
# https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast
# https://www.drivendata.co/blog/predict-pm25-benchmark/

import os
import random
import re
from pathlib import Path
from typing import Dict, List, Union

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from cloudpathlib import S3Client, S3Path
from pqdm.processes import pqdm
from pyhdf.SD import SD, SDC, SDS
from pyproj import CRS, Proj

DATA_PATH = Path("/mnt/d/airdata")
RAW = DATA_PATH / "raw"
INTERIM = DATA_PATH / "interim"
pm_md = pd.read_csv(
    INTERIM / "pm25_satellite_metadata.csv",
    parse_dates=["time_start", "time_end"],
    index_col=0
)

grid_md = pd.read_csv(
    DATA_PATH / "la_generated_grid.csv",
)
grid_md['location'] = "Los Angeles (SoCAB)"
grid_md['grid_id'] = grid_md['wkt'].apply(lambda x: hash(str(x)))


def transform_arrays(
    xv: Union[np.array, float],
    yv: Union[np.array, float],
    crs_from: CRS,
    crs_to: CRS
):
    """Transform points or arrays from one CRS to another CRS.

    Args:
        xv (np.array or float): x (longitude) coordinates or value.
        yv (np.array or float): y (latitude) coordinates or value.
        crs_from (CRS): source coordinate reference system.
        crs_to (CRS): destination coordinate reference system.

    Returns:
        lon, lat (tuple): x coordinate(s), y coordinate(s)
    """
    transformer = pyproj.Transformer.from_crs(
        crs_from,
        crs_to,
        always_xy=True,
    )
    lon, lat = transformer.transform(xv, yv)
    return lon, lat


def calculate_features(
    feature_df: gpd.GeoDataFrame,
):
    """Given processed AOD data and training labels (train) or 
    submission format (test), return a feature GeoDataFrame that contains
    features for mean, max, and min AOD.

    Args:
        feature_df (gpd.GeoDataFrame): GeoDataFrame that contains
            Points and associated values.
        label_df (pd.DataFrame): training labels (train) or
            submission format (test).
        stage (str): "train" or "test".

    Returns:
        full_data (gpd.GeoDataFrame): GeoDataFrame that contains `mean_aod`,
            `max_aod`, and `min_aod` for each grid cell and datetime.   
    """
    # Add `day` column to `feature_df` and `label_df`
    feature_df["datetime"] = pd.to_datetime(
        feature_df.granule_id.str.split("_", expand=True)[0],
        format="%Y%m%dT%H:%M:%S",
        utc=True
    )
    feature_df["day"] = feature_df.datetime.dt.date
    # label_df["day"] = label_df.datetime.dt.date

    # Calculate average AOD per day/grid cell for which we have feature data
    feature_df['geometry'] = str(feature_df['geometry'])
    avg_aod_day = feature_df.groupby(["day", "geometry"]).agg(
        {"value": ["mean", "min", "max"]}
    )
    avg_aod_day.columns = ["mean_aod", "min_aod", "max_aod"]
    avg_aod_day = avg_aod_day.reset_index()
    return avg_aod_day
    # # Join labels/submission format with feature data
    # how = "inner" if stage == "train" else "left"
    # full_data = pd.merge(
    #     label_df,
    #     avg_aod_day,
    #     how=how,
    #     left_on=["day", "grid_id"],
    #     right_on=["day", "grid_id"]
    # )
    # return full_data


def create_meshgrid(alignment_dict: Dict, shape: List[int]):
    """Given an image shape, create a meshgrid of points
    between bounding coordinates.

    Args:
        alignment_dict (Dict): dictionary containing, at a minimum,
            `upper_left` (tuple), `lower_right` (tuple), `crs` (str),
            and `crs_params` (tuple).
        shape (List[int]): dataset shape as a list of
            [orbits, height, width].

    Returns:
        xv (np.array): x (longitude) coordinates.
        yv (np.array): y (latitude) coordinates.
    """
    # Determine grid bounds using two coordinates
    x0, y0 = alignment_dict["upper_left"]
    x1, y1 = alignment_dict["lower_right"]

    # Interpolate points between corners, inclusive of bounds
    x = np.linspace(x0, x1, shape[2], endpoint=True)
    y = np.linspace(y0, y1, shape[1], endpoint=True)

    # Return two 2D arrays representing X & Y coordinates of all points
    xv, yv = np.meshgrid(x, y)
    return xv, yv


def calibrate_data(dataset: SDS, shape: List[int], calibration_dict: Dict):
    """Given a MAIAC dataset and calibration parameters, return a masked
    array of calibrated data.

    Args:
        dataset (SDS): dataset in SDS format (e.g. blue band AOD).
        shape (List[int]): dataset shape as a list of [orbits, height, width].
        calibration_dict (Dict): dictionary containing, at a minimum,
            `valid_range` (list or tuple), `_FillValue` (int or float),
            `add_offset` (float), and `scale_factor` (float).

    Returns:
        corrected_AOD (np.ma.MaskedArray): masked array of calibrated data
            with a fill value of nan.
    """
    corrected_AOD = np.ma.empty(shape, dtype=np.double)
    for orbit in range(shape[0]):
        data = dataset[orbit, :, :].astype(np.double)
        invalid_condition = (
            (data < calibration_dict["valid_range"][0]) |
            (data > calibration_dict["valid_range"][1]) |
            (data == calibration_dict["_FillValue"])
        )
        data[invalid_condition] = np.nan
        data = (
            (data - calibration_dict["add_offset"]) *
            calibration_dict["scale_factor"]
        )
        data = np.ma.masked_array(data, np.isnan(data))
        corrected_AOD[orbit, ::] = data
    corrected_AOD.fill_value = np.nan
    return corrected_AOD


def convert_array_to_df(
    corrected_arr: np.ma.MaskedArray,
    lat: np.ndarray,
    lon: np.ndarray,
    granule_id: str,
    crs: CRS,
    total_bounds: np.ndarray = None
):
    """Align data values with latitude and longitude coordinates
    and return a GeoDataFrame.

    Args:
        corrected_arr (np.ma.MaskedArray): data values for each pixel.
        lat (np.ndarray): latitude for each pixel.
        lon (np.ndarray): longitude for each pixel.
        granule_id (str): granule name.
        crs (CRS): coordinate reference system
        total_bounds (np.ndarray, optional): If provided, will filter out points that fall
            outside of these bounds. Composed of xmin, ymin, xmax, ymax.
    """
    lats = lat.ravel()
    lons = lon.ravel()
    n_orbits = len(corrected_arr)
    size = lats.size
    values = {
        "value": np.concatenate([d.data.ravel() for d in corrected_arr]),
        "lat": np.tile(lats, n_orbits),
        "lon": np.tile(lons, n_orbits),
        "orbit": np.arange(n_orbits).repeat(size),
        "granule_id": [granule_id] * size * n_orbits

    }

    df = pd.DataFrame(values).dropna()
    if total_bounds is not None:
        x_min, y_min, x_max, y_max = total_bounds
        df = df[df.lon.between(x_min, x_max) & df.lat.between(y_min, y_max)]

    gdf = gpd.GeoDataFrame(df)
    gdf["geometry"] = gpd.points_from_xy(gdf.lon, gdf.lat)
    gdf.crs = crs
    return gdf[["granule_id", "orbit", "geometry", "value"]].reset_index(drop=True)


def create_calibration_dict(data: SDS):
    """Define calibration dictionary given a SDS dataset,
    which contains:
        - name
        - scale factor
        - offset
        - unit
        - fill value
        - valid range

    Args:
        data (SDS): dataset in the SDS format.

    Returns:
        calibration_dict (Dict): dict of calibration parameters.
    """
    return data.attributes()


def create_alignment_dict(hdf: SD):
    """Define alignment dictionary given a SD data file, 
    which contains:
        - upper left coordinates
        - lower right coordinates
        - coordinate reference system (CRS)
        - CRS parameters

    Args:
        hdf (SD): hdf data object

    Returns:
        alignment_dict (Dict): dict of alignment parameters.
    """
    group_1 = hdf.attributes()["StructMetadata.0"].split("END_GROUP=GRID_1")[0]
    hdf_metadata = dict([x.split("=") for x in group_1.split() if "=" in x])
    alignment_dict = {
        "upper_left": eval(hdf_metadata["UpperLeftPointMtrs"]),
        "lower_right": eval(hdf_metadata["LowerRightMtrs"]),
        "crs": hdf_metadata["Projection"],
        "crs_params": hdf_metadata["ProjParams"]
    }
    return alignment_dict


def preprocess_maiac_data(
    granule_path: str,
    grid_cell_gdf: gpd.GeoDataFrame,
    dataset_name: str,
    total_bounds: np.ndarray = None
):
    """
    Given a granule s3 path and competition grid cells, 
    create a GDF of each intersecting point and the accompanying
    dataset value (e.g. blue band AOD).

    Args:
        granule_path (str): a path to a granule on s3.
        grid_cell_gdf (gpd.GeoDataFrame): GeoDataFrame that contains,
            at a minimum, a `grid_id` and `geometry` column of Polygons.
        dataset_name (str): specific dataset name (e.g. "Optical_Depth_047").
        total_bounds (np.ndarray, optional): If provided, will filter out points that fall
            outside of these bounds. Composed of xmin, ymin, xmax, ymax.    
    Returns:
        GeoDataFrame that contains Points and associated values.
    """
    # Load blue band AOD data
    # s3_path = S3Path(granule_path, S3Client(no_sign_request=True))
    hdf = SD(str(granule_path), SDC.READ)
    aod = hdf.select(dataset_name)
    shape = aod.info()[2]

    # Calibrate and align data
    calibration_dict = aod.attributes()
    alignment_dict = create_alignment_dict(hdf)
    corrected_AOD = calibrate_data(aod, shape, calibration_dict)
    xv, yv = create_meshgrid(alignment_dict, shape)
    lon, lat = transform_arrays(xv, yv, sinu_crs, wgs84_crs)

    # Save values that align with granules
    granule_gdf = convert_array_to_df(
        corrected_AOD, lat, lon, granule_path.name, wgs84_crs, grid_cell_gdf.total_bounds)
    df = gpd.sjoin(grid_cell_gdf, granule_gdf, how="inner")

    # Clean up files
    # Path(s3_path.fspath).unlink()
    hdf.end()
    return df.drop(columns="index_right").reset_index()


def preprocess_aod_47(granule_paths, grid_cell_gdf, n_jobs=8):
    """
    Given a set of granule s3 paths and competition grid cells, 
    parallelizes creation of GDFs containing AOD 0.47 µm values and points.

    Args:
        granule_paths (List): list of paths on s3.
        grid_cell_gdf (gpd.GeoDataFrame): GeoDataFrame that contains,
            at a minimum, a `grid_id` and `geometry` column of Polygons.
        n_jobs (int, Optional): The number of parallel processes. Defaults to 2.

    Returns:
        GeoDataFrame that contains Points and associated values for all granules.
    """
    args = ((gp, grid_cell_gdf, "Optical_Depth_047") for gp in granule_paths)

    results = pqdm(args, preprocess_maiac_data,
                   n_jobs=n_jobs, argument_type="args")
    return pd.concat(results)


if __name__ == "__main__":
    reg_code, reg_name = ("la", "Los Angeles (SoCAB)")
    split = 'train'

    maiac_md = pm_md[(pm_md["product"] == "maiac") &
                     (pm_md["split"] == split)].copy()
    la_file = maiac_md[maiac_md.location == reg_code].iloc[0]
    la_file_path = S3Path(la_file.us_url, S3Client(no_sign_request=True))

    hdf = SD(la_file_path.fspath, SDC.READ)
    print(hdf.info())
    for dataset, metadata in hdf.datasets().items():
        dimensions, shape, _, _ = metadata
        print(f"{dataset}\n    Dimensions: {dimensions}\n    Shape: {shape}")

    blue_band_AOD = hdf.select("Optical_Depth_047")

    name, num_dim, shape, types, num_attr = blue_band_AOD.info()

    print(
        f"""Dataset name: {name}
    Number of dimensions: {num_dim}
    Shape: {shape}
    Data type: {types}
    Number of attributes: {num_attr}"""
    )
    calibration_dict = blue_band_AOD.attributes()
    print(calibration_dict)
    raw_attr = hdf.attributes()["StructMetadata.0"]

    group_1 = raw_attr.split("END_GROUP=GRID_1")[0]
    hdf_metadata = dict([x.split("=") for x in group_1.split() if "=" in x])

    # Parse expressions still wrapped in apostrophes
    for key, val in hdf_metadata.items():
        try:
            hdf_metadata[key] = eval(val)
        except:
            pass

    alignment_dict = {
        "upper_left": hdf_metadata["UpperLeftPointMtrs"],
        "lower_right": hdf_metadata["LowerRightMtrs"],
        "crs": hdf_metadata["Projection"],
        "crs_params": hdf_metadata["ProjParams"]
    }

    # DATA PROCESSING

    # Loop over orbits to apply the attributes

    corrected_AOD = calibrate_data(blue_band_AOD, shape, calibration_dict)

    xv, yv = create_meshgrid(alignment_dict, shape)

    # Source: https://spatialreference.org/ref/sr-org/modis-sinusoidal/proj4js/
    sinu_crs = Proj(
        f"+proj=sinu +R={alignment_dict['crs_params'][0]} +nadgrids=@null +wktext").crs
    wgs84_crs = CRS.from_epsg("4326")

    # Project sinu grid onto wgs84 grid
    lon, lat = transform_arrays(xv, yv, sinu_crs, wgs84_crs)

    gdf = convert_array_to_df(corrected_AOD, lat, lon,
                              la_file_path.stem, wgs84_crs)

    # Identify LA granule filepaths (2019) and grid cells
    maiac_md = maiac_md[maiac_md.location == reg_code]

    la_fp = list(Path('/mnt/d/airdata/raw/train/maiac').glob(r'**/*.hdf'))
    la_gc = grid_md[grid_md.location == reg_name].copy()

    # Load training labels
    # train_labels = pd.read_csv(DATA_PATH / "train_labels.csv", parse_dates=["datetime"])
    # train_labels.rename(columns={"value": "pm25"}, inplace=True)
    la_polys = gpd.GeoSeries.from_wkt(
        la_gc.wkt, crs=wgs84_crs)  # used for WGS 84
    la_polys.name = "geometry"
    la_polys_gdf = gpd.GeoDataFrame(la_polys)
    la_polys_gdf['grid_id'] = la_polys_gdf['geometry'].apply(
        lambda x: hash(str(x)))
    xmin, ymin, xmax, ymax = la_polys_gdf.total_bounds
    gpd.sjoin(la_polys_gdf, gdf.cx[xmin:xmax, ymin:ymax], how="inner").groupby(
        "grid_id")["value"].mean()

    for i, part in enumerate(np.array_split(la_polys_gdf, 5)):
        print(i)
        la_train_gdf = preprocess_aod_47(la_fp, part)
        # la_train_gdf.to_file(f"{reg_code}_{split}_{i}.shp")
        full_data = calculate_features(la_train_gdf)
        full_data.to_csv(f"{reg_code}_{split}_{i}.csv", index=False)

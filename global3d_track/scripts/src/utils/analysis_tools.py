from timezonefinder import TimezoneFinder
import pytz
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from dask import delayed, compute
from scipy.interpolate import interp1d
import xarray as xr


def convert_utc_to_local(lat, lon, utc_time):
    # Step 1: Get time zone from latitude and longitude
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lng=lon, lat=lat)

    if timezone_str is None:
        raise ValueError("Could not determine timezone for the given coordinates.")

    # Step 2: Convert UTC time to local time
    utc_zone = pytz.utc
    local_zone = pytz.timezone(timezone_str)

    # Ensure UTC time is timezone-aware
    if utc_time.tzinfo is None:
        utc_time = utc_zone.localize(utc_time)

    local_time = utc_time.astimezone(local_zone)
    return local_time

def color_axis(ax, color='red', side='right', axis='y', pad=60):
    ax.spines[side].set_position(('outward', pad))
    ax.spines[side].set_color(color)  # Change spine color
    ax.yaxis.label.set_color(color)  # Change y-axis label color
    ax.tick_params(axis=axis, colors=color)  # Change tick color
    return ax


def normalise_by_lifetime(dataarray, cloud_exists, bins = np.arange(0, 1.05, 0.05)):

    def normalise_by_lifetime_vectorized(data, exists):
        data = data[exists]
        ntimes = data.size
        if ntimes < 2:
            return np.full((len(bins),), np.nan)
        time_percentage = np.linspace(0, 1, ntimes)
        return np.interp(bins, time_percentage, data)

    result = xr.apply_ufunc(
        normalise_by_lifetime_vectorized,
        dataarray,
        cloud_exists,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[['interp_time']],
        vectorize=True,
        dask='parallelized',
        dask_gufunc_kwargs={"output_sizes": {"interp_time": len(bins)}},
        output_dtypes=[float],
    )
    return result.assign_coords(interp_time=bins)


def find_max_coords(da):
    # stack to determine index of maxima
    stacked = da.stack(points=("lat", "lon"))
    flat_idx = stacked.argmax("points")

    # unravel_index via apply_ufunc (Dask-safe, returns a (2, ...) array)
    lat_len, lon_len = da.sizes["lat"], da.sizes["lon"]
    lat_idx, lon_idx = xr.apply_ufunc(
        lambda idx: np.unravel_index(idx, (lat_len, lon_len)),
        flat_idx,
        input_core_dims=[[]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int64, np.int64],
    )

    # Convert indices into coordinate values
    lat_vals = xr.apply_ufunc(
        np.take,
        da.lat,
        lat_idx,
        input_core_dims=[["lat"], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da.lat.dtype],
    )

    lon_vals = xr.apply_ufunc(
        np.take,
        da.lon,
        lon_idx,
        input_core_dims=[["lon"], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da.lon.dtype],
    )

    return xr.Dataset({"lat": lat_vals, "lon": lon_vals})

def define_cloud_centre(ds):
    ''' Langrangian. Uses max CMF location when a core is present, then uses max IWP when the core is gone. '''

    # cloud CMF
    c_data = ds.core_column_cmf_cl.max('core')
    cmf_max = find_max_coords(c_data.fillna(0))
    cmf_max = cmf_max.where(ds.core_volume.max('core')>0)

    # IWP
    iwp_max = find_max_coords(ds.anvil_iwp.fillna(0))
    iwp_max = iwp_max.where(ds.anvil_volume>0)

    # Use IWP index if CMF index is null (i.e., no core present)
    return iwp_max.where(cmf_max.isnull(), cmf_max)


def interpolate_winddir_transect(ds, centre):
    from metpy.interpolate import cross_section

    # 1. determine transect start and end points using the mean anvil wind

    def iterate_path(ds, silon, silat, tlon, tlat, backward=False):
        if backward:
            tlon = -tlon
            tlat = -tlat
        path = []
        prev_lon, prev_lat = next_ilon, next_ilat = silon, silat
        while next_ilon in range(ds.lon.size) and next_ilat in range(ds.lat.size):
            next_lon = prev_lon + tlon
            next_lat = prev_lat + tlat
            prev_ilon, prev_ilat = int(prev_lon), int(prev_lat)
            next_ilon, next_ilat = int(next_lon), int(next_lat)
            path.append((prev_ilon, prev_ilat))
            prev_lon, prev_lat = next_lon, next_lat
        # get lat, lon of end point
        coord_idx = path[-1]
        d = ds.isel(lon=coord_idx[0], lat=coord_idx[1])
        return (d.lat.item(), d.lon.item())

    # motion increments
    u = ds.anvil_ua.mean('time').compute().item()
    v = ds.anvil_va.mean('time').compute().item()
    ratio = np.abs(u / v)
    travel_lon = ratio * (u / np.abs(u))
    travel_lat = (v / np.abs(v))

    # start indices
    start_ilon = (ds.lon == centre[0]).argmax().item()
    start_ilat = (ds.lat == centre[1]).argmax().item()

    # iterate to end
    end = iterate_path(ds, start_ilon, start_ilat, travel_lon, travel_lat, backward=False)
    start = iterate_path(ds, start_ilon, start_ilat, travel_lon, travel_lat, backward=True)

    # 2. interpolate data to track

    # provide CRS information
    projection_crs = dict(grid_mapping_name = 'latitude_longitude')
    metpy_data = ds.metpy.assign_crs(projection_crs)

    # parse
    metpy_data = metpy_data.metpy.parse_cf().squeeze()

    # fill NaN
    metpy_data = metpy_data[['anvil_iwp']].fillna(0)

    # compute
    cross = cross_section(metpy_data, start, end)

    return cross





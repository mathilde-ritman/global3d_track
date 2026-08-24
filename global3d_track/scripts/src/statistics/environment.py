
import numpy as np
import xarray as xr



def ingested(ds, cpoint, pres_level, radius, zdim='level_full'):
    ''' Determine downwind region of storm, return a mask '''
    # 0 - centre point
    clon, clat, ctime = cpoint.lon, cpoint.lat, cpoint.time
    ds = ds.sel(time=ctime.isoformat())
    # 1 - calculate direction vectors
    X, Y = ds.lon, ds.lat
    X, Y = np.meshgrid(X, Y)
    dX, dY = clon - X, clat - Y
    magnitude = np.hypot(dX, dY)
    direction = dX / magnitude, dY / magnitude
    # 2 - calulate dot product with winds
    dlevel = ds.sel({zdim: np.abs(ds.pfull - pres_level).idxmin(zdim)}) # @ level
    winds = dlevel.ua, dlevel.va
    ingestion = direction[0]*winds[0] + direction[1]*winds[1]
    # 3 - within radius
    distance = xr.DataArray(magnitude, dims=("lat", "lon"), coords={ "lon": X[0, :], "lat": Y[:, 0],},)
    return ((ingestion > 0) & (distance < radius))
    
def interacting(ds, cpoint, radius):
    ''' Determine radius of storm, return a mask '''
     # 0 - centre point
    clon, clat, ctime = cpoint.lon, cpoint.lat, cpoint.time
    ds = ds.sel(time=ctime.isoformat())
    # 1 - calculate direction vectors & distance (in degrees)
    X, Y = ds.lon, ds.lat
    X, Y = np.meshgrid(X, Y)
    dX, dY = clon - X, clat - Y
    magnitude = np.hypot(dX, dY)
    distance = xr.DataArray(magnitude, dims=("lat", "lon"), coords={ "lon": X[0, :], "lat": Y[:, 0],},)
    return distance < radius

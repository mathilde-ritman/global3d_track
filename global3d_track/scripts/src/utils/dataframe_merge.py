import pandas as pd
import pathlib

def merge_records(track_dir, updrafts_fname='tracks', clouds_fname='tracks', updrafts_colname='tracks', clouds_colname='tracks', shared_colname='dcc'):
    ''' 
    Combines the tobac metadata tables created by multivariate tracking.

    track_dir: str or pathlib.Path
        Path to the directory containing the all the tracking dataframes.
    updrafts_fname: str
        Name of the updraft tracking dataframe (default: 'tracks').
    clouds_fname: str
        Name of the ice cloud tracking dataframe (default: 'tracks').
    updrafts_colname: str
        Name of the column in the updraft tracking dataframe that contains the final updraft track IDs (default: 'tracks').
    clouds_colname: str
        Name of the column in the ice cloud tracking dataframe that contains the final ice cloud track IDs (default: 'tracks').
    shared_colname: str
        Name of the column in the dataframes that recorded their shared system ID (default: 'dcc').

    '''

    track_dir = pathlib.Path(track_dir)

    # load track records
    updrafts = pd.read_hdf(list(track_dir.glob(f'{updrafts_fname}.h5'))[0], 'table').rename(columns={updrafts_colname:'u_tracks'})
    clouds = pd.read_hdf(list(track_dir.glob(f'{clouds_fname}.h5'))[0], 'table').rename(columns={clouds_colname:'cld_tracks'})

    # merge the two dataframes, this reports the overall time range of the system, and it's associated updraft tracks
    dcc_times = pd.concat((updrafts[['time',shared_colname,'u_tracks']], clouds[['time',shared_colname,]]))

    return {'w':updrafts, 'cld':clouds, 'df':dcc_times}


""" 
Example use

di = tools.merge_records(track_dir)
updrafts, clouds, dcc_times = di['w'], di['cld'], di['df']

number_systems = dcc_times.dcc.nunique()
first_updraft = updrafts.groupby('dcc').time.min()
number_cores = updrafts[['u_tracks','dcc']].groupby('dcc')['u_tracks'].nunique()
def get_lifetime(d):
    return (d.max() - d.min()).total_seconds() / (60*60) + 0.25 # to hours
df['lifetime'] = dcc_times[['time','dcc']].groupby('dcc')['time'].apply(get_lifetime)

"""
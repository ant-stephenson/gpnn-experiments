#%%
import pandas as pd
import numpy as np

#%%
path = ".tmp_data/household_power_consumption.txt"

#%% import the data and convert dtypes etc
dtype_dict = {'Global_active_power': np.float32, 'Global_reactive_power': np.float32, 'Voltage': np.float32, 'Global_intensity': np.float32, 'Sub_metering_1': np.float32, 'Sub_metering_2': np.float32, 'Sub_metering_3': np.float32}
data = pd.read_csv(path,sep=";", dtype=dtype_dict, na_values=["?"], parse_dates=["Date", "Time"])
# %% - rescale date and time
# date ->  to day of year / 365 -> (0,1]
# time ->  time of day / 24*60 -> (0,1]
data.loc[:, "dayofyear"] = data.Date.dt.dayofyear / 365
data.loc[:, "timeofday"] = (data.Time.dt.minute + data.Time.dt.hour * 60) / (24*60)
# %% set target
data.loc[:, "y"] = data.Global_active_power * 1000/60 - data.Sub_metering_1 - data.Sub_metering_2 - data.Sub_metering_3
# put reactive power on the same scale
data.Global_reactive_power *= 1000/60
# drop the now redundant columns
data.drop(["Date", "Global_active_power", "Time"], axis=1, inplace=True)
data.dropna(inplace=True)
# %%
data.to_csv(path.replace("txt", "csv"))
# %%

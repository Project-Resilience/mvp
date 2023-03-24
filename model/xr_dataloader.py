"""
You will want to install the following packages:
xarray
pytorch
xbatcher
"""
import numpy as np
import xarray as xr
import torch
import xbatcher
import regionmask
import os
import einops


#f = "zip:///::https://huggingface.co/datasets/jacobbieker/project-resilience/resolve/main/merged_aggregated_dataset.zarr.zip"
#dataset = xr.open_dataset(f, engine='zarr', chunks={})
#if not os.path.exists("merged_aggregated_dataset.zarr"):
#    dataset.to_zarr("merged_aggregated_dataset.zarr", consolidated=True, compute=True)
dataset = xr.open_zarr("/run/media/jacob/data/merged_aggregated_dataset.zarr.zip", consolidated=True)
# Train on countries 0-80, and test on the rest, and test on the last 15 years of data
country_mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(dataset)
#print(country_mask)
test_da = dataset.where(country_mask > 80,  drop=True).where(dataset.time > 2007, drop=True)
train_da = dataset.where(country_mask <= 80, drop=True).where(dataset.time <= 2007, drop=True)
#print(train_da)
#print(test_da)

# Simple LSTM model
class LSTMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(28, 128, num_layers=2, batch_first=True)
        self.average = torch.nn.AdaptiveAvgPool1d(1)
        self.linear = torch.nn.Linear(128, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.relu(x)
        x = self.linear(x)
        x = torch.relu(x)
        x = torch.squeeze(x)
        x = self.average(x)
        return x


model = LSTMModel()
crit = torch.nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)

for time in train_da.time.values[11:]:
    # Select random lat/lons ones in batch
    example = train_da.isel(lat=np.random.randint(0, len(train_da.lat.values), 12), lon=np.random.randint(0, len(train_da.lon.values), 10)).sel(time=slice(time-10,time))
    # stack all data variables in example into a single data array
    data_names = list(example.data_vars.keys())
    example = xr.concat([example[var] for var in example.data_vars], dim='variable')
    example = example.assign_coords(variable=data_names)
    example = example.transpose('time', 'lat', 'lon', 'variable')
    target = example.sel(variable='ELUC').isel(time=9)
    target = einops.rearrange(np.expand_dims(target.values, axis=-1), 'lat lon features -> (lat lon) features')
    not_nan_values = np.argwhere(~np.isnan(target))
    target = target[not_nan_values]
    train = example.isel(time=slice(0,9))
    train = einops.rearrange(train.values, 'time lat lon features -> (lat lon) time features')
    train = train[not_nan_values]
    # Convert infinite values and NaN to 0.0
    torch_example = torch.from_numpy(np.nan_to_num(train, posinf=0.0, neginf=0.0))
    torch_target = torch.unsqueeze(torch.from_numpy(np.nan_to_num(target, posinf=0.0, neginf=0.0)), dim=-1)
    # Flatten with einops to get the shape (batch_size, time, features)
    # Run the model and pass back the loss
    # zero the parameter gradients
    optim.zero_grad()
    # forward + backward + optimize
    outputs = model(torch_example)
    loss = crit(outputs, torch_target)
    loss.backward()
    optim.step()
    print(loss.item())

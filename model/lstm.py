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
from huggingface_hub import PyTorchModelHubMixin


#f = "zip:///::https://huggingface.co/datasets/jacobbieker/project-resilience/resolve/main/merged_aggregated_dataset.zarr.zip"
#dataset = xr.open_dataset(f, engine='zarr', chunks={})
#if not os.path.exists("merged_aggregated_dataset.zarr"):
#    dataset.to_zarr("merged_aggregated_dataset.zarr", consolidated=True, compute=True)
dataset = xr.open_zarr("../data/merged_aggregated_dataset_1850_2022.zarr.zip", consolidated=True)
dataset['ELUC'] = dataset['ELUC'].shift(time=1)
dataset['ELUC_diff'] = dataset['ELUC_diff'].shift(time=1)
dataset['time'] = dataset.time - 1
mask = dataset['ELUC_diff'].isnull()
dataset = dataset.where(~mask, drop=True)
# Train on countries 0-80, and test on the rest, and test on the last 15 years of data
country_mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(dataset)
LAND_FEATURES = ['c3ann', 'c3nfx', 'c3per','c4ann', 'c4per',
                 'pastr', 'primf', 'primn', 'range', 'secdf', 'secdn', 'urban', 'cell_area']

LAND_DIFF_FEATURES = ['c3ann_diff', 'c3nfx_diff', 'c3per_diff','c4ann_diff', 'c4per_diff',
                      'pastr_diff', 'primf_diff', 'primn_diff', 'range_diff', 'secdf_diff', 'secdn_diff', 'urban_diff']

FEATURES = LAND_FEATURES + LAND_DIFF_FEATURES

LABEL = 'ELUC'

ALL_FEATURES = FEATURES + [LABEL]
dataset = dataset[ALL_FEATURES]
#print(country_mask)

#print(train_da)
#print(test_da)

# Simple LSTM model
class LSTMModel(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, layers=2, hidden_dim=128, input_dim=26, output_dim=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.average = torch.nn.AdaptiveAvgPool1d(1)
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.relu(x)
        x = self.linear(x)
        x = torch.relu(x)
        x = torch.squeeze(x)
        x = self.average(x)
        return x


model_config = {"layers": 2, "hidden_dim": 128, "input_dim": 26, "output_dim": 1}
def train_country_lstm(country_code):
    if len(country_code) == 1:
        test_da = dataset.where(country_mask == country_code[0], drop=True).where(dataset.time > 2007, drop=True).load()
        train_da = dataset.where(country_mask == country_code[0], drop=True).where(dataset.time <= 2007, drop=True).load() # 143 is the code for the UK
    else:
        c_mask = xr.DataArray(np.in1d(country_mask, country_code).reshape(country_mask.shape),
                              dims=country_mask.dims, coords=country_mask.coords)
        test_da = dataset.where(c_mask, drop=True).where(dataset.time > 2007, drop=True).load()
        train_da = dataset.where(c_mask, drop=True).where(dataset.time <= 2007, drop=True).load() # 143 is the code for the UK
    model = LSTMModel(**model_config).cuda()
    crit = torch.nn.MSELoss()
    stat = torch.nn.L1Loss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    train_times = train_da.time.values[11:]
    for epoch in range(50):
        np.random.shuffle(train_times)
        for time in train_times:
            # Select random lat/lons ones in batch
            example = train_da.isel(lat=np.random.randint(0, len(train_da.lat.values), 12), lon=np.random.randint(0, len(train_da.lon.values), 10)).sel(time=slice(time-10,time))
            #country_mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(example)
            # example = example.where(country_mask <= 80, drop=True)
            # stack all data variables in example into a single data array
            data_names = list(example.data_vars.keys())
            example = xr.concat([example[var] for var in example.data_vars], dim='variable')
            example = example.assign_coords(variable=data_names)
            example = example.transpose('time', 'lat', 'lon', 'variable')
            target = example.sel(variable='ELUC').isel(time=9)
            train = example.isel(time=slice(0,9))
            #assert np.isfinite(example['ELUC'].values).all()
            # Convert infinite values and NaN to 0.0
            train = train.fillna(0.0)
            target = target.fillna(0.0)
            torch_example = torch.from_numpy(np.nan_to_num(train.values, posinf=0.0, neginf=0.0))
            # Flatten with einops to get the shape (batch_size, time, features)
            torch_example = einops.rearrange(torch_example, 'time lat lon features -> (lat lon) time features')
            torch_target = torch.unsqueeze(torch.from_numpy(np.nan_to_num(target.values, posinf=0.0, neginf=0.0)), dim=-1)
            # Flatten with einops to get the shape (batch_size, time, features)
            torch_target = einops.rearrange(torch_target, 'lat lon features -> (lat lon) features')
            # Run the model and pass back the loss
            # zero the parameter gradients
            optim.zero_grad()
            # forward + backward + optimize
            outputs = model(torch_example.cuda())
            loss = crit(outputs, torch_target.cuda())
            loss.backward()
            optim.step()
            #print(loss.item())

    with torch.no_grad():
        model.eval()
        test_mse = 0.0
        test_mae = 0.0
        num = 0
        test_times = test_da.time.values[11:]
        for time in test_times:
            for lon in range(0, len(test_da.lon.values), 10):
                for lat in range(0, len(test_da.lat.values), 10):
                    example = test_da.isel(lat=slice(lat,lat+10), lon=slice(lon,lon+10)).sel(time=slice(time-10,time))
                    data_names = list(example.data_vars.keys())
                    example = xr.concat([example[var] for var in example.data_vars], dim='variable')
                    example = example.assign_coords(variable=data_names)
                    example = example.transpose('time', 'lat', 'lon', 'variable')
                    target = example.sel(variable='ELUC').isel(time=9)
                    train = example.isel(time=slice(0,9))
                    #assert np.isfinite(example['ELUC'].values).all()
                    # Convert infinite values and NaN to 0.0
                    train = train.fillna(0.0)
                    target = target.fillna(0.0)
                    torch_example = torch.from_numpy(np.nan_to_num(train.values, posinf=0.0, neginf=0.0))
                    # Flatten with einops to get the shape (batch_size, time, features)
                    torch_example = einops.rearrange(torch_example, 'time lat lon features -> (lat lon) time features')
                    torch_target = torch.unsqueeze(torch.from_numpy(np.nan_to_num(target.values, posinf=0.0, neginf=0.0)), dim=-1)
                    # Flatten with einops to get the shape (batch_size, time, features)
                    torch_target = einops.rearrange(torch_target, 'lat lon features -> (lat lon) features')
                    num += torch_target.shape[0]
                    outputs = model(torch_example.cuda())
                    loss = crit(outputs.cpu(), torch_target)
                    mae_loss = stat(outputs.cpu(), torch_target)
                    #print(loss.item())
                    test_mse += loss.item()
                    test_mae += mae_loss.item()
                    # Already averaged, so just need to track num for time, lat, lon
                    num += 1
        print(f"Test  {country_code} MSE: {test_mse/num}")
        print(f"Test {country_code} MAE: {test_mae/num}")
        names = [f'{c}_' for c in country_code]
        # Join the names together
        name = ''.join(names)
        torch.save(model, f"lstm_model_{name}.pt")
        # Save the weights to huggingface
        model.save_pretrained(f"lstm_model_{name}", push_to_hub=True, repo_name=f"project_resilience_lstm_model_{name}", use_auth_token=None, config=model_config)

train_country_lstm([143, 29]) # UK and India
train_country_lstm([143]) # UK
train_country_lstm([29]) # Brazil
# Check ELUC when no change
# Send over weights, put them on HF and integrate that as well as config options

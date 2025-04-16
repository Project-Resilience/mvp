# Heatmap Fix

I lost the RF weights and therefore had to regenerate them from the original figure.

Stored in `heatmap_data.csv` is the hexcode colors used in each cell of the heatmap.

`generate_new.py` is the script used to re-generate the RF predictions. The pixel values of the colorbar are hard-coded into it. The script converts the colors into positions on the colorbar, then the scale of the colorbar is computed with the pixel values to re-scale the positions into true values.

Then we use `create_heatmap.py` to create the new heatmap figure given the found data.

`create_heatmap.py` is the entrypoint to this experiment and will call `generate_new.py` and `generate_old.py` if the heatmap data is not found.
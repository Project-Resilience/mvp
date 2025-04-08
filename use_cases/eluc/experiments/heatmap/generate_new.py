"""
Reverse-engineering the values of a heatmap from the RGB values of the colors.
"""
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

TOP_TICK = 20
BOTTOM_TICK = -15
PIXELS_PER_TICK = 119
TICK_INTERVAL = 5
PIXELS_TO_TOP = 31
PIXELS_TO_BOTTOM = 39


def get_end_value(label: int, end_pixels: int, tick_interval: int, tick_pixels: int):
    """
    Gets the max/min value of the heatmap's color bar based on its pixels. We can measure the ratio between pixels
    and scale by dividing the pixels between 2 ticks by the interval between the ticks. Then we can see how far in
    scale the top of the color bar is from the top tick by counting the pixels between the top tick and the top of the
    color bar.
    """
    converted_distance = end_pixels * tick_interval / tick_pixels
    if label < 0:
        return label - converted_distance
    else:
        return label + converted_distance


def colors_to_scaled_values(colors: np.ndarray, lookup_table: np.ndarray) -> float:
    """
    Takes an array of colors in RGB format shape (n x m x 3) and using a lookup table of RGB values, finds the closest
    colors in the lookup table. The lookup table is ordered from lowest to highest value so we can scale from 0 to 1
    based off the index of the closest color. The color is expected to be in the range of 0 to 1.
    """
    scaled_values = []
    for color in colors.reshape(-1, 3):
        diffs = np.linalg.norm(lookup_table[:, :-1] - color, axis=1)
        scaled_val = np.argmin(diffs) / (len(lookup_table) - 1)
        scaled_values.append(scaled_val)
    scaled_values = np.array(scaled_values).reshape(colors.shape[:-1])
    return scaled_values


def generate_rf_heatmap_data():
    """
    Generates raw heatmap values from the RGB values of the heatmap.
    """
    # 1. Find the max and min values of the heatmap based on the pixels of the color bar.
    top_val = get_end_value(TOP_TICK, PIXELS_TO_TOP, TICK_INTERVAL, PIXELS_PER_TICK)
    bottom_val = get_end_value(BOTTOM_TICK, PIXELS_TO_BOTTOM, TICK_INTERVAL, PIXELS_PER_TICK)
    end_val = max(abs(top_val), abs(bottom_val))

    # 2. Load original heatmap colors
    original_colors = pd.read_csv("experiments/heatmap/heatmap_data.csv")
    rgb = original_colors[["r", "g", "b"]].values
    rgb = rgb / 255.0
    rgb = rgb.reshape((8, 5, 3))

    # 3. Create a lookup table of RGB values to "true" values using the colormap from the original heatmap.
    # NOTE: This is a copy of how we made the colormap in the first place.
    colors = ["darkgreen", "white", "red"]
    bins = 1000
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=bins)
    lookup_table = cmap(np.linspace(0, 1, bins))

    # 4. Find the closest RGB value in the lookup table for each color in the heatmap and get its scaled value from
    # 0 to 1.
    scaled_values = colors_to_scaled_values(rgb, lookup_table)

    # 5. Scale data back up to the original values based on pixels.
    heatmap_values = scaled_values * end_val * 2 - end_val

    # The heatmap is flipped upside down, so we need to flip it back.
    heatmap_values = np.flip(heatmap_values, axis=0)

    # 6. Save to disk.
    np.save("experiments/heatmap/rf_heatmap_data.npy", heatmap_values)


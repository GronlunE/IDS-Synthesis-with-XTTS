"""
Created on 31.8. 2024

@author: GronlunE

Description:

This script generates various types of plots using predefined plotting functions from the `plotting` module. The script is designed to create visualizations including scatter plots, radar plots, and kernel density estimates (KDE).

The script calls functions to produce:
- KDE plots for visualizing the distribution of data.
- Scatter plots to display the relationships between pairs of variables.
- Radar plots for comparing multiple variables on a common scale.

Usage:
- Ensure the `plotting` module is available and correctly implemented.
- Run the script to generate the plots. Each type of plot will be created and displayed or saved as specified in the respective plotting functions.

Dependencies:
- `plotting` module, which should include `scatter`, `radar`, and `kde` submodules with the appropriate plotting functions.

"""

from plotting import scatter, radar, kde

# Generate all KDE plots
kde.plot_all_kde()

# Generate scatter plots
scatter.plot_scatters()

# Generate radar plots
radar.plot_radar_plots()

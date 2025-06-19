Please download the full depository, otherwise the .exe file will not work. 

The species_dict.pkl file contains all data necessary for heatmap.py, best_glide_curve_fits.py, and general_fit.py.  

heatmap.py creates a heatmap of loss in the CDb and CDpro space and shows the linear fit in the minimum loss region.

best_glide_curve_fits.py shows the glide polars created by CDb and CDpro parameter pairs on the linear fit.

general_fit.py creates a heatmap combining species (8 species). 

GlideTool.exe is an installer file that generates outputs based on user input and stores our dataset of birds.

GlideTool.py is the source code of GlideTool.exe.

There are two .pkl files: one from all original gliding points and the second from only steady gliding points.

To use the Python files, please copy the scripts and the .pkl file to your environment, then run the scripts.

GlideTool Instructions

The program requires several inputs, including mass, body frontal area, wing area, wingspan, and observed gliding points with their horizontal and vertical components.
Species studied in the associated manuscript are available in the dropdown menu.
There are several options to save the figures in different formats: as images, CSV files, or Excel spreadsheets.

Note: This program operates based on Pennycuick's gliding formulas, using all of the original assumptions.

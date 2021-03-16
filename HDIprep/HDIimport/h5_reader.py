# Module for h(df)5 imaging data parsing
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external modules
from pathlib import Path
import h5py
import numpy as np

# Define class object
class H5reader:
    """H(df)5 cytometry data reader using h5py"""

    def __init__(self, path_to_h5):
        """Initialize the class by using the path to the image."""

        # Initialize the objects in this class
        self.image = None

        # Create a pathlib object for the path_to_tif
        path_to_h5 = Path(path_to_h5)

        # Read h(df)5 data and return the parsed data
        f = h5py.File(path_to_h5, "r+")
        # Get the dataset name from the h(df)5 file
        dat_name = list(f.keys())[0]
        ###If the hdf5 is exported from ilastik fiji plugin, the dat_name will be 'data'
        # Get the image data
        im = np.array(f[dat_name])
        # Remove the first axis (ilastik convention)
        im = im.reshape((im.shape[1], im.shape[2], im.shape[3]))

        # Close the h(df)5 file
        f.close()

        # Assign the data to the class
        self.image = im

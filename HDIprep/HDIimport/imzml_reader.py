# Module for imzML parsing of imaging mass spectrometry (IMS) data
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external modules
from pathlib import Path
import os
from pyimzml.ImzMLParser import getionimage
from pyimzml.ImzMLParser import ImzMLParser
import numpy as np
import pandas as pd
from operator import itemgetter
import scipy
import skimage
import skimage.io

# Import custom modules
from .utils import SubsetCoordinates


# Create a class object to store attributes and functions in
class imzMLreader:
    """Class for parsing and storing IMS data that is in the imzML format. Depends
    on and contains the pyimzML python package distributed from the Alexandrov team:
    https://github.com/alexandrovteam/pyimzML in a data object.

    path_to_imzML: string indicating path to imzML file (Ex: 'path/IMSdata.imzML')
    """

    def __init__(
        self,
        path_to_imzML,
        flatten,
        subsample,
        mask=None,
        path_to_markers=None,
        **kwargs
    ):
        """Initialize class to store data in. Ensure appropriate file format
        and return a data object with pixel table.
        """

        # Create a pathlib object for the path_to_imzML
        path_to_imzML = Path(path_to_imzML)

        # Set the file extensions that we can use with this class
        ext = [".imzML"]

        # Check to make sure the string is a valid path
        if not os.path.exists(str(path_to_imzML)):
            print("Not a valid path. Try again")
        else:
            print("Valid path...")
            # Check to see if there is a valid file extension for this class
            if str(path_to_imzML).endswith(tuple(ext)):
                print(
                    "Valid file extension...",
                    "\nfile name:",
                    str(path_to_imzML),
                    "\nparsing imzML...",
                )
                # Read imzML and return the parsed data
                self.data = ImzMLParser(str(path_to_imzML))
                print("Finished parsing imzML")
            else:
                print("Not a valid file extension")

        # Add the image size to the data object
        self.data.array_size = (
            self.data.imzmldict["max count of pixels y"],
            self.data.imzmldict["max count of pixels x"],
        )

        # Check to see if the mask exists
        if mask is not None:
            # Check to see if the mask is a path (string)
            if isinstance(mask, str):
                ##############Change in future to take arbitrary masks not just tiff################
                mask = skimage.io.imread(str(mask), plugin="tifffile")
            # Ensure the mask is a sparse boolean array
            mask = scipy.sparse.coo_matrix(mask, dtype=np.bool)

        # Add the mask to the class object -- even if it is none. Will not be applied to image yet
        self.data.mask = mask
        # Create an object for a filtered/processed working
        self.data.processed_image = None

        # Check to see if creating a pixel table (used for dimension reduction)
        if flatten:

            # Check to see if we are using a mask
            if mask is not None:

                # Ensure that the mask is boolean
                mask = np.array(mask.toarray(), dtype=np.bool)
                # Get the coordinates where the mask is
                where = np.where(mask)
                # Create list of tuples where mask coordinates are (1-indexed) -- form (x,y,z) with z=1 (same as imzML)
                coords = list(
                    zip(
                        where[1] + 1, where[0] + 1, np.ones(len(where[0]), dtype=np.int)
                    )
                )
                # intersect the mask coordinates with the IMS coordinates from imzML parser
                mask_coords = list(set(coords) & set(self.data.coordinates))

                # Reorder the mask coordinates for F style column major format (imzml format)
                mask_coords = sorted(mask_coords, key=itemgetter(0, 1))

                # Clear the old coordinates for memory
                coords, where, mask = None, None, None

                # Zip the coordinates into dictionary with list index (Faster with itemgetter)
                full_coords_dict = dict(
                    zip(self.data.coordinates, range(0, len(self.data.coordinates)))
                )
                # Find the indices of the mask coordinates -- need for creating dataframe
                coords_idx = list(itemgetter(*mask_coords)(full_coords_dict))

                # Remove the dictionary to save memory
                full_coords_dict = None

                # Reset the coordinates object to be only the mask coordinates
                self.data.coordinates = mask_coords

            # Otherwise create a coords_idx from the full list of coordinates
            else:
                # Create list
                coords_idx = [x for x in range(len(self.data.coordinates))]

            # Check to see if subsampling
            if subsample is not None:
                # Use the coordinates for subsampling
                sub_mask, coords = SubsetCoordinates(
                    coords=self.data.coordinates,
                    array_size=self.data.array_size,
                    **kwargs
                )

                # Alter the order to be in column major format Fortran style
                coords = sorted(coords, key=itemgetter(0, 1))

                # Clear space with the mask
                sub_mask = None

                # Get the indices now of these coordinates from the coords_idx
                # coords_idx = [self.data.coordinates.index(x) for x in coords]
                # Zip the coordinates into dictionary with list index (Faster with itemgetter)
                tmp_coords_dict = dict(
                    zip(self.data.coordinates, range(0, len(self.data.coordinates)))
                )
                # Find the indices of the mask coordinates -- need for creating dataframe
                coords_idx = list(itemgetter(*coords)(tmp_coords_dict))

                # Clear the coordinates dictionary to save memory
                tmp_coords_dict = None

                # Add the subset coordinates to our object
                self.data.sub_coordinates = coords

            # Otherwise there is no subsampling so leave the coordinates as they are
            else:
                # Keep the full list of coordinates
                coords = self.data.coordinates
                # Add the subset coordinates as None
                self.data.sub_coordinates = None

            # Create numpy array with cols = m/zs an rows = pixels (create pixel table)
            tmp = np.empty([len(coords), len(self.data.getspectrum(0)[0])])

            # iterate through pixels and add to the array
            print("Fetching Spectrum Table...")
            for i, (x, y, z) in enumerate(coords):
                # Get the coordinate index
                idx = coords_idx[i]
                # Now use the index to extract the spectrum
                mzs, intensities = self.data.getspectrum(idx)
                # Use the original i index to add to the array the data
                tmp[i, :] = intensities
                # Clear memory by removing mzs and intensities
                mzs, intensities = None, None

            # Create a pandas dataframe from numpy array
            tmp_frame = pd.DataFrame(
                tmp, index=coords, columns=self.data.getspectrum(0)[0]
            )
            # Delete the temporary object to save memory
            tmp = None
            # Assign the data to an array in the data object
            self.data.pixel_table = tmp_frame

            # Get the image shape of the data
            self.data.image_shape = (
                self.data.imzmldict["max count of pixels y"],
                self.data.imzmldict["max count of pixels x"],
                self.data.pixel_table.shape[1],
            )
        else:
            # Create a pixel table as None
            self.data.pixel_table = None
            # Set the image shape as None
            self.data.image_shape = None

        # Add the filename to the data object
        self.data.filename = path_to_imzML
        # Add None to the data image (not currently parsing full array)
        self.data.image = None

        # Print an update that the import is finished
        print("Finished")

    def SubsetData(self, range=None):
        """Subset an IMS peak list to fall between a range of values.

        range: tuple indicating range (Ex (400,1000)). Note for memory reasons
        the PixelTable is overwritten, and a new subset of the peak list isnt created.
        """

        # Get the lowest value
        low = next(
            x for x, val in enumerate(self.data.pixel_table.columns) if val >= range[0]
        )
        # Get the highest value
        hi = [n for n, i in enumerate(self.data.pixel_table.columns) if i <= range[1]][
            -1
        ]
        # Assign the new peak list to the pixel_table (add +1 because python isnt inclusive)
        self.data.pixel_table = self.data.pixel_table.iloc[:, low : hi + 1]

    def ExportChannels(self):
        """Export a txt file with channel names for downstream analysis."""

        # Print a sheet for m/z and channel numbers
        sheet = pd.DataFrame(self.data.pixel_table.columns, columns=["channels"])
        # Write out the sheet to csv
        sheet.to_csv(path_to_imzML.stem + "_channels.csv", sep="\t")

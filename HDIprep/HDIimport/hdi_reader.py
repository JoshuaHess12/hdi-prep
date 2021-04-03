# Module for high-dimensional imaging data importing
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external modules
from pathlib import Path
import os

# Import custom modules
from .cyt_reader import CYTreader
from .imzml_reader import imzMLreader
from .nifti1_reader import NIFTI1reader


# Create class object to store high-dimensional imaging data
class HDIreader:
    """Class for importing high-dimensional imaging data or histology data."""

    def __init__(
        self,
        path_to_data,
        path_to_markers=None,
        flatten=False,
        subsample=None,
        mask=None,
        save_mem=False,
        **kwargs
    ):
        """Initialize class to store data in. Ensure appropriate file format
        and return a data object with pixel table.

        path_to_data: path to imaging data (Ex: path/mydata.extension)
        path_to_markers: path to marker list (Ex: path/mymarkers.csv or None)
        flatten: True to return a flattened pixel data table for dimension reduction
        mask: Path to tif mask to use for selecting a region to focus on in downstream preparation
        **kwargs: - input to SubsetCoordinates utils function
        """

        # Initialize objects
        self.hdi = None

        # Create a pathlib object for the path_to_cyt
        path_to_data = Path(path_to_data)

        # Set the file extensions that we can use with this class
        all_ext = [
            ".ome.tif",
            ".ome.tiff",
            ".tif",
            ".tiff",
            ".h5",
            ".hdf5",
            ".imzML",
            ".nii",
        ]
        # Get file extensions for cytometry files
        cyt_ext = [".ome.tif", ".ome.tiff", ".tif", ".tiff", ".h5", ".hdf5"]
        # Get file exntensions for h(df)5 files
        imzML_ext = [".imzML"]
        # Get file extensions for NIFTI1 files
        nii_ext = [".nii"]

        # Check to see if there is a valid file extension for this class
        if str(path_to_data).endswith(tuple(cyt_ext)):
            # Read the data with CYTreader
            self.hdi = CYTreader(
                path_to_cyt=path_to_data,
                path_to_markers=path_to_markers,
                subsample=subsample,
                flatten=flatten,
                mask=mask,
                **kwargs
            )

        # Otherwise read imzML file
        elif str(path_to_data).endswith(tuple(imzML_ext)):
            # Read the data with imzMLreader (CURRENTLY DOES NOT SUPPORT A MASK -- set default to None in class object)
            self.hdi = imzMLreader(
                path_to_imzML=path_to_data,
                path_to_markers=path_to_markers,
                subsample=subsample,
                flatten=flatten,
                mask=mask,
                **kwargs
            )

        # Otherwise read nifti file
        elif str(path_to_data).endswith(tuple(nii_ext)):
            # Read the data with NIFTI1reader
            self.hdi = NIFTI1reader(
                path_to_nifti=path_to_data,
                path_to_markers=path_to_markers,
                subsample=subsample,
                flatten=flatten,
                mask=mask,
                **kwargs
            )

        # check to see if saving storage
        if save_mem:
            # set image arrays to none
            self.hdi.data.image = None
            self.hdi.data.mask = None

        # If none of the above print an update and an error
        else:
            # Raise an error saying that the file extension is not recognized
            raise ValueError("File extension not recognized.")

# General utily functions for HDI data parsing
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external modules
from pathlib import Path
import os
import h5py
import skimage.io
import numpy as np
import pandas as pd
import random
import scipy.sparse
from operator import itemgetter
from ast import literal_eval


# Define function for reading csv marker names
def ReadMarkers(path_to_markers):
    """Function for reading the marker list csv file and returning a list
    of strings for the markers.

    path_to_markers: Path to csv file (Ex: path/to/markers.csv)
    """

    # Create a pathlib object for the path to markers
    path_to_markers = Path(path_to_markers)

    # Read the channels names to pass to the
    channel_names = pd.read_csv(path_to_markers, header=None)
    # Add a column index for ease
    channel_names.columns = ["marker"]
    # Convert the channel names to a list
    channel_names = list(channel_names.marker.values)
    # Return the channel_names
    return channel_names


# Define function for subsampling random coordinates
def SubsetCoordinates(
    coords, array_size, method="random", n=10000, grid_spacing=(2, 2)
):
    """Subset coordinates and return a list of tuples indicating
    the position of subsetting and where to find those tuples in the list. Also
    return a scipy coo matrix for boolean mask to use with subsetting.
    Note: Coordinates are 1-indexed and not python 0-indexed

    coords: list of 3D-tuples indicating the position of pixels
    array_size: tuple indicating the 2D array size (not counting channels)
    method: Method of subsampling. Currently supported "grid", "random", and "pseudo-random"
    n: integer (decimal less than 1) indicating the size of subsampling (percentage)
    """
    #####Note: indices are switched row = 1 col = 0 to match imzML parser#####
    # Check to see if the method is uniform random sampling
    if method == "random":
        # Check to see if the value is less than or equal to 1
        if n < 1:
            # Interpret this value as a percentage
            n = int(len(coords) * n)
        # Otherwise the value is total pixel count
        else:
            # Interpret the value as pixel count
            n = n

        # Set random seed
        random.seed(1234)
        # Take subsample of integers for indexing
        idx = list(np.random.choice(a=len(coords), size=n, replace=False))
        # Use the indices to subsample coordinates
        sub_coords = [coords[c] for c in idx]
        # Create data with True values same length as sub_coords for scipy coo matrix
        data = np.ones(len(sub_coords), dtype=np.bool)

        # Create row data for scipy coo matrix (-1 index for 0-based python)
        row = np.array([sub_coords[c][1] - 1 for c in range(len(sub_coords))])
        # Create row data for scipy coo matrix (-1 index for 0-based python)
        col = np.array([sub_coords[c][0] - 1 for c in range(len(sub_coords))])

    # Check to see if the method is uniform sampling with random after
    elif method == "pseudo_random":
        # Check to see if the value is less than or equal to 1
        # if n < 1:
        # Interpret this value as a percentage
        # num_resamples = int(len(coords) * n)

        if n >= 0.25:
            # intialize the sampling with 2x2 grid
            grdh = 2
            grdw = 2
        elif 0.25 > n >= 0.11:
            # intialize the sampling with 3x3 grid
            grdh = 3
            grdw = 3
        elif 0.11 > n >= 0.0625:
            # intialize the sampling with 4x4 grid
            grdh = 4
            grdw = 4
        else:
            # intialize the sampling with 5x5 grid
            grdh = 5
            grdw = 5
        # Get maximum indices for x and y directions
        max_nh, max_nw = (
            max(coords, key=itemgetter(1))[1],
            max(coords, key=itemgetter(0))[0],
        )
        # Get maximum indices for x and y directions
        min_nh, min_nw = (
            min(coords, key=itemgetter(1))[1],
            min(coords, key=itemgetter(0))[0],
        )
        # Get grid in height direction and width directions
        row = np.arange(min_nh, max_nh, grdh)
        col = np.arange(min_nw, max_nw, grdw)
        # Create data with True values same length as sub_coords for scipy coo matrix
        data = np.ones(len(row) * len(col), dtype=np.bool)
        # Create meshgrid from the grid coordinates
        row, col = np.meshgrid(row, col)

        # Create list of subcoordinates from mesh -- this is now a bounding box around the mask coordinates
        sub_coords = list(
            map(
                tuple,
                np.vstack(
                    (col.ravel() + 1, row.ravel() + 1, np.ones(len(data), dtype=np.int64))
                ).T,
            )
        )
        # Intersect the original coordinates with the grid coordinates so if mask or ROI is not square, we can capture
        sub_coords = list(set(sub_coords) & set(coords))
        # Get number of pixels of grid coordinates and calculate pixels left
        grid_perc = len(sub_coords) / len(coords)

        # Get number of pixels left to sample
        n = int(len(coords) * (n - grid_perc))

        # Create an inverse list of pixels
        inv_pix = list(set(coords).difference(set(sub_coords)))

        # Set random seed
        random.seed(1234)
        # Take subsample of integers for indexing from pixels left after grid
        idx = list(np.random.choice(a=len(inv_pix), size=n, replace=False))
        # Use the indices to subsample coordinates
        rand_sub_coords = [inv_pix[c] for c in idx]

        # Add these coordinates to the grid coordinates
        sub_coords = rand_sub_coords + sub_coords

        # Create data with True values same length as sub_coords for scipy coo matrix
        data = np.ones(len(sub_coords), dtype=np.bool)

        # Create row data for scipy coo matrix (-1 index for 0-based python)
        row = np.array([sub_coords[c][1] - 1 for c in range(len(sub_coords))])
        # Create row data for scipy coo matrix (-1 index for 0-based python)
        col = np.array([sub_coords[c][0] - 1 for c in range(len(sub_coords))])

    # Check to see if the method is uniform grid sampling
    elif method == "grid":

        # ensure the grid spacing is interpreted correctly
        if isinstance(grid_spacing, str):
            # Interpret the grid spacing as tuple
            grid_spacing = literal_eval(grid_spacing)

        # Get the size of the grid
        grdh = grid_spacing[0]
        grdw = grid_spacing[1]

        # Get maximum indices for x and y directions
        max_nh, max_nw = (
            max(coords, key=itemgetter(1))[1],
            max(coords, key=itemgetter(0))[0],
        )
        # Get maximum indices for x and y directions
        min_nh, min_nw = (
            min(coords, key=itemgetter(1))[1],
            min(coords, key=itemgetter(0))[0],
        )
        # Get grid in height direction and width directions
        row = np.arange(min_nh, max_nh, grdh)
        col = np.arange(min_nw, max_nw, grdw)
        # Create data with True values same length as sub_coords for scipy coo matrix
        data = np.ones(len(row) * len(col), dtype=np.bool)
        # Create meshgrid from the grid coordinates
        row, col = np.meshgrid(row, col)

        # Create list of subcoordinates from mesh -- this is now a bounding box around the mask coordinates
        sub_coords = list(
            map(
                tuple,
                np.vstack(
                    (col.ravel() + 1, row.ravel() + 1, np.ones(len(data), dtype=np.int64))
                ).T,
            )
        )
        # Intersect the original coordinates with the grid coordinates so if mask or ROI is not square, we can capture
        sub_coords = list(set(sub_coords) & set(coords))
        # Create row data for scipy coo matrix (-1 index for 0-based python)
        row = np.array([sub_coords[c][1] - 1 for c in range(len(sub_coords))])
        # Create row data for scipy coo matrix (-1 index for 0-based python)
        col = np.array([sub_coords[c][0] - 1 for c in range(len(sub_coords))])

        # Create data with True values same length as sub_coords for scipy coo matrix
        data = np.ones(len(sub_coords), dtype=np.bool)

    # Otherwise raise an error
    else:
        # Raise value error
        raise (
            Exception(
                "Method of subsampling entered is not supported. Please enter 'random', 'grid', or 'pseudo_random'"
            )
        )

    # Numpy flattens with row major order -- reorder coorinates to index resulting pandas data frame
    sub_coords = sorted(sub_coords, key=itemgetter(1, 0))
    # Create a subset mask
    sub_mask = scipy.sparse.coo_matrix((data, (row, col)), shape=array_size)

    # Return the objects
    return sub_mask, sub_coords


# Define function for flattening a z stack image
def FlattenZstack(z_stack, z_stack_shape, mask, subsample, **kwargs):
    """This function will flatten an ndarray. Assumes the order of dimensions
    is xyc with c being the third channel.

    z_stack: xyc image with c channels
    mask: Optional mask to extract only pixels that fall within that region.
    If no mask is given (Mask=None), then the full pixel array is returned. A
    mask is useful for very large cytometry images (e.g. CyCIF) so that dimension
    reduction doesn't include the pixels in the background.
    subsample: Optional to subsample the data to preserve memory (Ex: subsample = 0.5).
    If the number for subsampling is less than or equal to 1, the number is interpretated as a
    percentage of total pixels to be subsampled instead of pixel counts.
    """

    # Check to see if this is a single-channel image or not
    if len(z_stack_shape) > 2:
        # Get the number of channels in the z_stack
        num_channels = z_stack_shape[2]
    # Otherwise this is a single-channel
    else:
        # Set number of channels to be one
        num_channels = 1

    # Check to see if using a mask to extract only a subregion
    if mask is None:
        # Create numpy boolean mask array with cols = channels an rows = pixels
        mask = np.ones([int(z_stack_shape[0]), z_stack_shape[1]], dtype=np.bool)

    else:
        # Check to see if the mask is a path (string)
        if isinstance(mask, str):
            ##############Change in future to take arbitrary masks not just tiff?################
            mask = skimage.io.imread(str(mask), plugin="tifffile")
        # Otherwise check for sparse coo matrix
        elif isinstance(mask, scipy.sparse.coo_matrix):
            # Convert to numpy array and boolean
            mask = mask.toarray()

        # Ensure that the mask is boolean
        mask = np.array(mask, dtype=np.bool)

    # Get the coordinates where the mask is
    where = np.where(mask)
    # Create list of tuples where mask coordinates are (1-indexed) -- form (x,y,z) with z=1 (same as imzML)
    coords = list(zip(where[1] + 1, where[0] + 1, np.ones(len(where[0]), dtype=np.int64)))
    # Reorder the coordinates to be c style
    coords = sorted(coords, key=itemgetter(1, 0))

    # Check to see if subsampling
    if subsample:
        # Subset the coordinates using custom function
        sub_mask, sub_coords = SubsetCoordinates(
            coords=coords, array_size=mask.shape, **kwargs
        )

        # Create an array from the sparse scipy matrix
        sub_mask = sub_mask.toarray()
        # Use the mask to extract all the pixels
        z_stack = z_stack[np.where(sub_mask)]
        # Remove the masks to save memory
        sub_mask = None
        # Create a pandas dataframe with columns being the number of channels
        z_stack = pd.DataFrame(
            z_stack,
            columns=[str(num) for num in range(0, num_channels)],
            index=sub_coords,
        )
    # Otherwise there is no subsampling
    else:
        # Use the non-subsampled mask to extract data
        z_stack = z_stack[np.where(mask)]
        # Create a pandas dataframe with columns being the  number of channels
        z_stack = pd.DataFrame(
            z_stack, columns=[str(num) for num in range(0, num_channels)], index=coords
        )

    # Return the flattened array
    return z_stack, coords

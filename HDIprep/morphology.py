# Morphological operation functions for HDI data preparation
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external modules
import numpy as np
import skimage.filters
import skimage.morphology
import skimage.color
import scipy.sparse


# Define function
def MedFilter(image, filter_size, parallel=False):
    """Median filtering of images to remove salt and pepper noise.
    A circular disk is used for the filtering. Images are automatically converted to
    single channel grayscale images if they arent already in single channel format

    filter_size: size of disk to use for filter.
    parallel: number of proceses to use for calculations
    """

    # Ensure that the image is grayscale
    if len(image) > 2:
        # Not yet converted to grayscale - use grayscale image
        image = skimage.color.rgb2gray(image)

    # Check to see if parallel computation
    if parallel:
        # Filter the image to remove salt and pepper noise (use original image)
        filtered_im = skimage.util.apply_parallel(
            skimage.filters.median,
            image,
            extra_keywords={
                # Set size of circular filter
                "selem": skimage.morphology.disk(filter_size)
            },
        )
    # Use single processor
    else:
        # Otherwise use only singe processor
        filtered_im = skimage.filters.median(
            image, selem=skimage.morphology.disk(filter_size)
        )

    # Return the filtered image
    return filtered_im


def Thresholding(image, type, thresh_value=None, correction=1.0):
    """Otsu (manual) thresholding of grayscale images. Returns a sparse boolean
    mask.

    image: numpy array that represents image
    type: Type of thresholding to use. Options are 'manual' or "otsu"
    thresh_value: If manual masking, insert a threshold value
    correction: Correction factor after thresholding. Values >1 make the threshold more
    stringent. By default, value with be 1 (identity)
    """

    # Ensure that the image is grayscale
    if len(image) > 2:
        # Not yet converted to grayscale - use grayscale image
        image = skimage.color.rgb2gray(image)

    # Check is the threshold type is otsu
    if type == "otsu":
        # Otsu threshold
        thresh_value = skimage.filters.threshold_otsu(image) * correction

    # Check if manual thresholding
    elif type == "manual":
        # manual threshold
        thresh_value = thresh_value * correction

    # Otherwise raise an exception!
    else:
        # Raise exception
        raise (Exception("Threshold type not supported!"))

    # Create a mask from the threshold value
    thresh_img = image < thresh_value
    # Convert the mask to a boolean sparse matrix
    return scipy.sparse.coo_matrix(thresh_img, dtype=np.bool)


def Opening(mask, disk_size, parallel=False):
    """Morphological opening on boolean array (mask). A circular disk is used for the filtering.

    disk_size: size of disk to use for filter.
    parallel: number of proceses to use for calculations
    """

    # Ensure that the image is boolean
    if not mask.dtype is np.dtype(np.bool):
        # Raise an exception
        raise (Exception("Mask must be a boolean array!"))

    # Proceed to process the mask as an array
    if isinstance(mask, scipy.sparse.coo_matrix):
        # Convert to array
        mask = mask.toarray()

    # Check to see if parallel computation
    if parallel:
        # Filter the image to remove salt and pepper noise (use original image)
        mask = skimage.util.apply_parallel(
            skimage.morphology.opening,
            mask,
            extra_keywords={
                # Set size of circular filter
                "selem": skimage.morphology.disk(disk_size)
            },
        )
    # Use single processor
    else:
        # Otherwise use only singe processor
        mask = skimage.morphology.opening(
            mask, selem=skimage.morphology.disk(disk_size)
        )

    # Convert the mask back to scipy sparse matrix for storage
    return scipy.sparse.coo_matrix(mask, dtype=np.bool)


def Closing(mask, disk_size, parallel=False):
    """Morphological closing on boolean array (mask). A circular disk is used for the filtering.

    disk_size: size of disk to use for filter.
    parallel: number of proceses to use for calculations
    """

    # Ensure that the image is boolean
    if not mask.dtype is np.dtype(np.bool):
        # Raise an exception
        raise (Exception("Mask must be a boolean array!"))

    # Proceed to process the mask as an array
    if isinstance(mask, scipy.sparse.coo_matrix):
        # Convert to array
        mask = mask.toarray()

    # Check to see if parallel computation
    if parallel:
        # Filter the image to remove salt and pepper noise (use original image)
        mask = skimage.util.apply_parallel(
            skimage.morphology.closing,
            mask,
            extra_keywords={
                # Set size of circular filter
                "selem": skimage.morphology.disk(disk_size)
            },
        )
    # Use single processor
    else:
        # Otherwise use only singe processor
        mask = skimage.morphology.closing(
            mask, selem=skimage.morphology.disk(disk_size)
        )

    # Convert the mask back to scipy sparse matrix for storage
    return scipy.sparse.coo_matrix(mask, dtype=np.bool)


def MorphFill(mask):
    """Morphological filling on a binary mask. Fills holes"""

    # Ensure that the image is boolean
    if not mask.dtype is np.dtype(np.bool):
        # Raise an exception
        raise (Exception("Mask must be a boolean array!"))

    # Proceed to process the mask as an array
    if isinstance(mask, scipy.sparse.coo_matrix):
        # Convert to array
        mask = mask.toarray()

    # Filling in the mask
    mask = scipy.ndimage.binary_fill_holes(mask)
    # Return the mask
    return scipy.sparse.coo_matrix(mask, dtype=np.bool)


def NonzeroSlice(mask, original):
    """Slice original image and mask to be a certain size based on bounding
    region around mask"""

    # Ensure that the image is boolean
    if not mask.dtype is np.dtype(np.bool):
        # Raise an exception
        raise (Exception("Mask must be a boolean array!"))

    # Proceed to process the mask as an array
    if isinstance(mask, scipy.sparse.coo_matrix):
        # Convert to array
        mask = mask.toarray()

    # Get nonzero indices from your mask so we can apply to original image
    nonzero = np.nonzero(mask)
    # Get bounding box
    minx = min(nonzero[0])
    maxx = max(nonzero[0])
    miny = min(nonzero[1])
    maxy = max(nonzero[1])
    # Extract sliced, nonzero regions from your original image
    original = original[minx:maxx, miny:maxy]
    # Extract sliced, nonzero regions from your mask image
    mask = mask[minx:maxx, miny:maxy]

    # Return the original image and then the mask as a sparse matrix
    return original, scipy.sparse.coo_matrix(mask, dtype=np.bool)

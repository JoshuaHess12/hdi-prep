#General utily functions for HDI data preparation
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import external modules
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import skimage.filters
import skimage.morphology
import skimage.color
import scipy.sparse


#Define function
def CreateHyperspectralImage(embedding,array_size,coordinates):
    """Fill a hyperspectral image from n-dimensional embedding of high-dimensional
    imaging data by rescaling each channel from 0-1

    array_size: tuple indicating size of image
    embedding: Embedding resulting from dimension reduction
    coordinates: 1-indexed list of tuples indicating coordinates of image

    All coordinates in the image not listed in coordinates object will be masked
    and set to 0 (background)
    """

    #Create zeros array to fill with number channels equal to embedding dimension
    im = np.zeros((array_size[0],array_size[1],embedding.shape[1]), dtype = np.float32)

    #Run through the data coordinates and fill array
    for i, (x, y, z) in enumerate(coordinates):
        #Run through each slice of embedding (dimension)
        for dim in range(embedding.shape[1]):
            #Add data to this slice
            im[y - 1, x - 1,dim] = embedding.values[i,dim]

    #Create a mask to use for excluding pixels not used in dimension reduction
    im_bool = np.zeros((array_size[0],array_size[1]), dtype = np.bool)

    #Fill the mask array with True values at the coordinates used
    for i, (x, y, z) in enumerate(coordinates):
        #Add boolean mask
        im_bool[y - 1, x - 1] = True

    #Scale the data 0-1 for hyperspectral image construction
    for dim in range(im.shape[2]):
        #min-max scaler
        im[:,:,dim]=(im[:,:,dim]-im[:,:,dim].min())/(im[:,:,dim].max()-im[:,:,dim].min())

    #Mask the image with the boolean array to remove unused pixels
    im[~im_bool] = 0

    #Return the hyperspectral image
    return im



def ExportNifti(image,filename,padding=None):
    """This function will export your final images to nifti format for image
    registration with elastix.

    image: numpy ndarray containing imaging data
    filename: path (filename) of resulting exporting image (Ex: path/to/new/image.nii or image.nii)
    padding: tuple indicating the pad to be added onto the image in the height and length direction

    Your filename endings must be .nii!!
    """

    #Create pathlib object from the filename
    filename = Path(filename)

    #Print update
    print('Exporting nifti image stack...')
    #Check to see if padding
    if padding is not None:
        image = np.pad(image, [(padding[0],padding[0]),(padding[1],padding[1]),(0,0)], mode = 'constant')
    #Create nifti object -- transpose axes because of the transformation!
    nifti_im = nib.Nifti1Image(image.transpose(1,0,2), affine=np.eye(4))
    #Save the image
    nib.save(nifti_im, str(filename))
    #Print update
    print('Finished exporting '+str(filename))



def MedFilter(image,filter_size,parallel=False):
    """Median filtering of images to remove salt and pepper noise.
    A circular disk is used for the filtering. Images are automatically converted to
    single channel grayscale images if they arent already in single channel format

    filter_size: size of disk to use for filter.
    parallel: number of proceses to use for calculations
    """

    #Ensure that the image is grayscale
    if len(image)>2:
        #Not yet converted to grayscale - use grayscale image
        image = skimage.color.rgb2gray(image)

    #Check to see if parallel computation
    if parallel:
        #Filter the image to remove salt and pepper noise (use original image)
        filtered_im = skimage.util.apply_parallel(
            skimage.filters.median,
            image,
            extra_keywords={
                #Set size of circular filter
                "selem":skimage.morphology.disk(filter_size)
            }
        )
    #Use single processor
    else:
        #Otherwise use only singe processor
        filtered_im = skimage.filters.median(image,selem=skimage.morphology.disk(filter_size))

    #Return the filtered image
    return filtered_im



def Threshold(image,type,thresh_value,correction=1.0):
    """Otsu (manual) thresholding of grayscale images. Returns a sparse boolean
    mask.

    image: numpy array that represents image
    type: Type of thresholding to use. Options are 'manual' or "otsu"
    thresh_value: If manual masking, insert a threshold value
    correction: Correction factor after thresholding. Values >1 make the threshold more
    stringent. By default, value with be 1 (identity)
    """

    #Ensure that the image is grayscale
    if len(image)>2:
        #Not yet converted to grayscale - use grayscale image
        image = skimage.color.rgb2gray(image)

    #Check is the threshold type is otsu
    if type == 'otsu':
        #Otsu threshold
        thresh_value = skimage.filters.threshold_otsu(image)*correction

    #Check if manual thresholding
    elif type == 'manual':
        #manual threshold
        thresh_value = thresh_value*correction

    #Otherwise raise an exception!
    else:
        #Raise exception
        raise(Exception('Threshold type not supported!'))

    #Create a mask from the threshold value
    thresh_img = image < thresh_value
    #Convert the mask to a boolean sparse matrix
    return scipy.sparse.coo_matrix(thresh_img,dtype=np.bool)



def Opening(mask,disk_size,parallel=False):
    """Morphological opening on boolean array (mask). A circular disk is used for the filtering.

        disk_size: size of disk to use for filter.
        parallel: number of proceses to use for calculations
    """

    #Ensure that the image is boolean
    if not mask.dtype is np.dtype(np.bool):
        #Raise an exception
        raise(Exception("Mask must be a boolean array!"))

    #Proceed to process the mask as an array
    if isinstance(mask, scipy.sparse.coo_matrix):
        #Convert to array
        mask = mask.toarray()

    #Check to see if parallel computation
    if parallel:
        #Filter the image to remove salt and pepper noise (use original image)
        mask = skimage.util.apply_parallel(
            skimage.morphology.opening,
            mask,
            extra_keywords={
                #Set size of circular filter
                "selem":skimage.morphology.disk(disk_size)
            }
        )
    #Use single processor
    else:
        #Otherwise use only singe processor
        mask = skimage.morphology.opening(mask,selem=skimage.morphology.disk(disk_size))

    #Convert the mask back to scipy sparse matrix for storage
    return scipy.sparse.coo_matrix(mask,dtype=np.bool)



def Closing(mask,disk_size,parallel=False):
    """Morphological closing on boolean array (mask). A circular disk is used for the filtering.

        disk_size: size of disk to use for filter.
        parallel: number of proceses to use for calculations
    """

    #Ensure that the image is boolean
    if not mask.dtype is np.dtype(np.bool):
        #Raise an exception
        raise(Exception("Mask must be a boolean array!"))

    #Proceed to process the mask as an array
    if isinstance(mask, scipy.sparse.coo_matrix):
        #Convert to array
        mask = mask.toarray()

    #Check to see if parallel computation
    if parallel:
        #Filter the image to remove salt and pepper noise (use original image)
        mask = skimage.util.apply_parallel(
            skimage.morphology.closing,
            mask,
            extra_keywords={
                #Set size of circular filter
                "selem":skimage.morphology.disk(disk_size)
            }
        )
    #Use single processor
    else:
        #Otherwise use only singe processor
        mask = skimage.morphology.closing(mask,selem=skimage.morphology.disk(disk_size))

    #Convert the mask back to scipy sparse matrix for storage
    return scipy.sparse.coo_matrix(mask,dtype=np.bool)


def MorphFill(self,image):
    """This function will perform morphological filling on your histology image
    mask"""
    #Filling in the mask
    print('Performing Morphological Fill...')
    filled_mask = ndimage.binary_fill_holes(image).astype('uint8')
    plt.imshow(filled_mask)
    plt.title("Filled Mask")
    plt.show()

    return filled_mask


import skimage.io
image = skimage.io.imread("/Users/joshuahess/Desktop/tmp/15gridspacing.tif",plugin='tifffile')
import matplotlib.pyplot as plt
mask = skimage.color.rgb2gray(image) <0.5
plt.imshow(mask)
mask.dtype
test=scipy.sparse.coo_matrix(mask,dtype=np.bool)
test.dtype is np.dtype(np.bool)

disk = skimage.morphology.disk(5)
skimage.morphology.opening(mask,selem = disk)

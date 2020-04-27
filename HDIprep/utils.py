#General utily functions for HDI data preparation
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import external modules
import numpy as np
import pandas as pd
import nibabel as nib



#Define function
def CreateHyperspectralImage(embedding,array_size,coordinates):
    """Fill a hyperspectral image from n-dimensional embedding of high-dimensional
    imaging data.

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



def ExportNifti(image,filename=None,rgb=False):
    """This function will export your final images to nifti format for image
    registration with elastix

    Your filename endings must be .nii
    new_size must be a tuple of integers"""
    
    #Get a copy of the image so we dont change it
    tmp = image.copy()
    #Add in your rotation so we can store it for later registration
    self.data.Image_rot = rot
    #Optional to change the image size
    if resize:
        tmp = cv2.resize(tmp,new_size)
    self.data.Image_resize_shape = new_size
    #Create the nifti objects
    print('Exporting nifti stack image...')
    nifti_col = nib.Nifti1Image(np.rot90(tmp,rot), affine=np.eye(4))
    nib.save(nifti_col, filename)
    print('Finished exporting nifti stack image')
    #Convert to grayscale if you choose
    if grayscale:
        fin_gray = rgb2gray(tmp)
        filename_gray = filename+'_gray.nii'
        #Export the grayscale image
        print('Exporting nifti gray image...')
        nifti_org = nib.Nifti1Image(np.rot90(fin_gray,rot), affine=np.eye(4))
        nib.save(nifti_org, filename_gray)
        print('Finished exporting nifti gray image')

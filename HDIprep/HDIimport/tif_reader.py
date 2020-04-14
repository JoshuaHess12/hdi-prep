#Module for ome.tif(f) and tif(f) imaging data parsing
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import external modules
from pathlib import Path
import skimage
import numpy as np


#Define class object
class TIFreader:
    """TIF cytometry data reader using scikit image. Ca
    """

    def __init__(self,path_to_tif):
        """Initialize the class by using the path to the image.

        path_to_tif: Path to tif image (Ex: path/to/image.extension)
        """

        #Initialize the objects in this class
        self.image = None

        #Create a pathlib object for the path_to_tif
        path_to_tif = Path(path_to_tif)

        #Read tif(f) or ome.tif(f) data and return the parsed data
        im = skimage.io.imread(path_to_tif, plugin = 'tifffile')
        #Check to see if the number of channels is greater than one
        im_shape = im.shape
        #Check to see if the image is considered xyc or just xy(single channel)
        #Note: skimage with tifffile plugin reads channel numbers of 1 as xy array,
        #and reads images with 3 and 4 channels in the correct order. Channel numbers
        #of 2 or >5 need axis swapping
        if len(im_shape) > 2:
            ##########This will fail if the array is a 3 or 4 channel image with 5 pixels in the x direction...shouldnt happen##########
            #Check if channel numbers are 3 or 4
            if (im_shape[2] is 3) or (im_shape[2] is 4):
                pass
            else:
                #If number of channels is less than two then swap the axes to be zyxc
                im = np.swapaxes(im,0,2)
                #Swap the axes to be in the order zyxc
                im = np.swapaxes(im,0,1)
        #Assign the data to the class
        self.image = im

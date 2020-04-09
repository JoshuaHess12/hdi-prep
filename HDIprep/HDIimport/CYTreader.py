#Module for cytometry data parsing
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import external modules
from pathlib import Path
import os
import h5py
import skimage
import numpy as np
import pandas as pd
import random

#Import custom modules
import SubsetCoordinates
import TIFreader
import H5reader


#Define functions to be used in the class CYTreader
def FlattenZstack(z_stack, z_stack_shape, mask, subsample):
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

    #Check to see if this is a single-channel image or not
    if len(z_stack_shape) > 2:
        #Get the number of channels in the z_stack
        num_channels = z_stack_shape[2]
    #Otherwise this is a single-channel
    else:
        #Set number of channels to be one
        num_channels = 1

    #Check to see if using a mask to extract only a subregion
    if mask is None:
        #Create numpy boolean mask array with cols = channels an rows = pixels
        mask = np.ones([int(z_stack_shape[0]),z_stack_shape[1]],dtype=np.bool)
    else:
        ##############Change in future to take arbitrary masks??################
        #Read the image using TIFreader class
        mask = TIFreader.TIFreader(mask).image

        #Ensure that the mask is boolean
        mask = np.array(mask,dtype=np.bool)
    #Get the coordinates where the mask is
    where = np.where(mask)
    #Create list of tuples where mask coordinates are (1-indexed) -- form (x,y,z) with z=1 (same as imzML)
    coords = list(zip(where[1]+1,where[0]+1,np.ones(len(where[0]),dtype=np.int)))

    #Check to see if subsampling
    if subsample is not None:
        #Check to see if the value is less than or equal to 1
        if subsample <= 1:
            #Interpret this value as a percentage
            n = int(len(coords) * subsample)
        #Otherwise the value is total pixel count
        else:
            #Interpret the value as pixel count
            n = subsample

        #Subset the coordinates using custom function
        sub_mask, sub_coords = SubsetCoordinates.SubsetCoordinates(coords=coords,n=n,array_size=mask.shape)

        #Create an array from the sparse scipy matrix
        sub_mask = sub_mask.toarray()
        #Use the mask to extract all the pixels
        flat_im = z_stack[sub_mask]
        #Remove the masks to save memory
        mask, sub_mask = None, None
        #Create a pandas dataframe with columns being the number indexes for number of channels
        flat_im = pd.DataFrame(flat_im,\
            columns = [str(num) for num in range(0,num_channels)],\
            index = sub_coords)
    #Otherwise there is no subsampling
    else:
        #Use the non-subsampled mask to extract data
        flat_im = z_stack[mask]
        #Create a pandas dataframe with columns being the number indexes for number of channels
        flat_im = pd.DataFrame(flat_im,\
            columns = [str(num) for num in range(0,num_channels)],\
            index = coords)
    #Return the flattened array
    return flat_im, coords


def ReadMarkers(path_to_markers):
    """Function for reading the marker list csv file and returning a list
    of strings for the markers.

    path_to_markers: Path to csv file (Ex: path/to/markers.csv)
    """

    #Create a pathlib object for the path to markers
    path_to_markers = Path(path_to_markers)

    #Read the channels names to pass to the
    channel_names = pd.read_csv(path_to_markers,header=None)
    #Add a column index for ease
    channel_names.columns = ["marker"]
    #Convert the channel names to a list
    channel_names = list(channel_names.marker.values)
    #Return the channel_names
    return channel_names



#Create a class object to store attributes and functions in
class CYTreader:
    """Class for parsing and storing cytometry data that is in the ome.tif(f),
    h(df)5, or tif(f) format.

    path_to_cyt: string indicating path to cytometry file (Ex: 'path/CYTdata.ome.tif')
    """

    def __init__(self,path_to_cyt,path_to_markers,flatten,subsample,mask=None):
        """Initialize class to store data in. Ensure appropriate file format
        and return a data object with pixel table.
        """

        #Create a pathlib object for the path_to_cyt
        path_to_cyt = Path(path_to_cyt)

        #Set the file extensions that we can use with this class
        all_ext = [".ome.tif",".ome.tiff",".tif",".tiff",".h5",".hdf5"]
        #Get file extensions for ome.tif(f) or tif(f) files
        tif_ext = [".ome.tif",".ome.tiff",".tif",".tiff"]
        #Get file exntensions for h(df)5 files
        h5_ext = [".h5",".hdf5"]

        #Check to make sure the string is a valid path
        if not os.path.exists(str(path_to_cyt)):
            print('Not a valid path. Try again')
        else:
            print('Valid path...')
            #Check to see if there is a valid file extension for this class
            if str(path_to_cyt).endswith(tuple(all_ext)):
                print("Valid file extension...","\nfile name:",str(path_to_cyt),"\nparsing cytometry data...")

                #Read the data by searching through the extensions
                if str(path_to_cyt).endswith(tuple(tif_ext)):
                    #Read the ome.tif(f) or tif(f)
                    self.data = TIFreader.TIFreader(path_to_cyt)

                elif str(path_to_cyt).endswith(tuple(h5_ext)):
                    #Read h(df)5 data and return the parsed data
                    self.data = H5reader.H5reader(path_to_cyt)
            else:
                print("Not a valid file extension")

        #Add the shape of the image to the class object for future use
        self.data.image_shape = self.data.image.shape
        #Get the array size for the image
        self.data.array_size = (self.data.image_shape[0],self.data.image_shape[1])

        #Check for a marker list
        if path_to_markers is not None:
            #Read the channels list
            channels = ReadMarkers(path_to_markers)
        else:
            #Check to see if the image shape includes a channel (if not, it is one channel)
            if len(self.data.image.shape) > 2:
                #Get the number of channels in the imaging data
                num_channels = self.data.image_shape[2]
                #Create a numbered marker list based on channel number
                channels = [str(num) for num in range(0,num_channels)]
            #Otherwise just create a single entry for single-channel image
            else:
                #Create a numbered marker list based on channel number
                channels = [str(num) for num in range(0, 1)]

            #Add the channels to the class object
            self.data.channels = channels

        #Check to see if creating a pixel table (used for dimension reduction)
        if flatten:
            #Create a pixel table and extract the full list of coordinates being used
            pix, coords = FlattenZstack(z_stack=self.data.image, z_stack_shape=self.data.image_shape,\
                mask=mask, subsample=subsample)
            #Add the pixel table to our object
            self.data.pixel_table = pd.DataFrame(pix,columns = channels, index = pix.index)
            #Clear the pixel table object to save memory
            pix = None
            #Check to see if we subsampled
            if subsample is None:
                #Assign subsampled coordinates to be false
                self.data.sub_coordinates = None
            else:
                #Add pixel coordinates to the class object (similar to imzML parser) subsampled
                self.data.sub_coordinates = list(self.data.pixel_table.index)
            #Assign full coordinates to be coords
            self.data.coordinates = coords

        else:
            #Create a pixel table as None
            self.data.pixel_table = None
            #Set the pixel coordinates as None
            self.data.coordinates = None

        #Add the filename to the data object
        self.data.filename = path_to_cyt

        #Print an update on the parsing of cytometry data
        print("Finished parsing cytometry data")

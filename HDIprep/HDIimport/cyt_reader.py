#Module for cytometry data parsing
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import external modules
from pathlib import Path
import os
import h5py
import skimage.io
import numpy as np
import pandas as pd
import random

#Import custom modules
from .tif_reader import TIFreader
from .h5_reader import H5reader
from .utils import ReadMarkers, FlattenZstack



#im1 = hdi_reader.HDIreader(path_to_cyt = "/Users/joshuahess/Desktop/tmp/image.ome.tiff",\
#    path_to_markers=None,flatten=True,subsample=False,mask=None)
#
#
#        #Check to make sure the string is a valid path
#        if not os.path.exists(str(path_to_cyt)):
#            print('Not a valid path. Try again')
#        else:
#            print('Valid path...')
#            #Check to see if there is a valid file extension for this class
#            if str(path_to_cyt).endswith(tuple(all_ext)):
#                print("Valid file extension...","\nfile name:",str(path_to_cyt),"\nparsing cytometry data...")
#
#                #Read the data by searching through the extensions
#                if str(path_to_cyt).endswith(tuple(tif_ext)):
#                    #Read the ome.tif(f) or tif(f)
#                    data = TIFreader(path_to_cyt)
#
#                elif str(path_to_cyt).endswith(tuple(h5_ext)):
#                    #Read h(df)5 data and return the parsed data
#                    data = H5reader(path_to_cyt)
#            else:
#                print("Not a valid file extension")
#
#        #Add the shape of the image to the class object for future use
#        image_shape = data.image.shape
#        #Get the array size for the image
#        array_size = (image_shape[0],image_shape[1])
#        #Get the number of channels in the imaging data
#        num_channels = image_shape[2]
#        #Create a numbered marker list based on channel number
#        channels = [str(num) for num in range(0,num_channels)]


#z_stack = data.image
#z_stack_shape = image_shape
#coords = list(zip(where[1]+1,where[0]+1,np.ones(len(where[0]),dtype=np.int)))

    #Check to see if using a mask to extract only a subregion
#    if mask is None:
        #Create numpy boolean mask array with cols = channels an rows = pixels
#        mask = np.ones([int(z_stack_shape[0]),z_stack_shape[1]],dtype=np.bool)

#    else:
        #Check to see if the mask is a path (string)
#        if isinstance(mask, str):
            ##############Change in future to take arbitrary masks not just tiff??################
#            mask = skimage.io.imread(str(mask),plugin='tifffile')

        #Ensure that the mask is boolean
#        mask = np.array(mask,dtype=np.bool)

    #Get the coordinates where the mask is
#    where = np.where(mask)
    #Create list of tuples where mask coordinates are (1-indexed) -- form (x,y,z) with z=1 (same as imzML)
#    coords = list(zip(where[1]+1,where[0]+1,np.ones(len(where[0]),dtype=np.int)))

#n=0.3

#        if method is "random":
            #Check to see if the value is less than or equal to 1
#            if n < 1:
                #Interpret this value as a percentage
#                n = int(len(coords) * n)
            #Otherwise the value is total pixel count
#            else:
                #Interpret the value as pixel count
#                n = n

            #Set random seed
#            random.seed(1234)
            #Take subsample of integers for indexing
#            idx = list(np.random.choice(a=len(coords), size=n,replace=False))
            #Use the indices to subsample coordinates
#            sub_coords = [coords[c] for c in idx]
            #Create data with True values same length as sub_coords for scipy coo matrix
#            data = np.ones(len(sub_coords),dtype=np.bool)

            #Create row data for scipy coo matrix (-1 index for 0-based python)
#            row = np.array([sub_coords[c][1]-1 for c in range(len(sub_coords))])
            #Create row data for scipy coo matrix (-1 index for 0-based python)
#            col = np.array([sub_coords[c][0]-1 for c in range(len(sub_coords))])
#####Numpy flattens with row major...so need to reorder sub_coords based on row major
#x = list(zip(where[1],where[0]))

#import operator
#test = sorted(sub_coords, key = operator.itemgetter(1, 0))



#sub_mask = scipy.sparse.coo_matrix((data, (row,col)), shape=array_size)

#import matplotlib.pyplot as plt
#plt.imshow(sub_mask.toarray())
#skimage.io.imsave('test.tif',sub_mask.toarray().astype('uint16'),plugin='tifffile')


#z_stack[np.where(sub_mask.toarray())]


#flat_im = z_stack[sub_mask.toarray()]

#z_stack[0, 21,:]

#Create a class object to store attributes and functions in
class CYTreader:
    """Class for parsing and storing cytometry data that is in the ome.tif(f),
    h(df)5, or tif(f) format.

    path_to_cyt: string indicating path to cytometry file (Ex: 'path/CYTdata.ome.tif')
    """

    def __init__(self,path_to_cyt,path_to_markers,flatten,subsample,mask=None,**kwargs):
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
                    self.data = TIFreader(path_to_cyt)

                elif str(path_to_cyt).endswith(tuple(h5_ext)):
                    #Read h(df)5 data and return the parsed data
                    self.data = H5reader(path_to_cyt)
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
                mask=mask, subsample=subsample, **kwargs)
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

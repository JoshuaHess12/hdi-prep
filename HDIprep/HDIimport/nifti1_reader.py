#Module for NIFTI format data parsing
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import external modules
from pathlib import Path
import os
import nibabel as nib
import numpy as np
import pandas as pd

#Import custom modules
from .utils import ReadMarkers, FlattenZstack


#Create a class object to store attributes and functions in
class NIFTI1reader:
    """Class for parsing and storing data that is in the NIFTI1 format. Depends
    on and contains the NiBabel python package:
    https://nipy.org/nibabel/ in a data object.

    path_to_nifti: string indicating path to .nii file (Ex: 'path/IMSdata.nii')
    """

    def __init__(self,path_to_nifti,flatten,subsample,mask,path_to_markers=None,**kwargs):
        """Initialize class to store data in. Ensure appropriate file format
        and return a data object with pixel table.
        """

        #Create a pathlib object for the path_to_imzML
        path_to_nifti = Path(path_to_nifti)

        #Set the file extensions that we can use with this class
        ext = [".nii"]

        #Check to make sure the string is a valid path
        if not os.path.exists(str(path_to_nifti)):
            print('Not a valid path. Try again')
        else:
            print('Valid path...')
            #Check to see if there is a valid file extension for this class
            if str(path_to_nifti).endswith(tuple(ext)):
                print("Valid file extension...","\nfile name:",str(path_to_nifti),"\nparsing nifti...")
                #Read imzML and return the parsed data
                self.data = nib.load(str(path_to_nifti))

                #####Currently transpose the image as default####
                #Add the image from nibabel to the image object (memmap object)
                self.data.image = self.data.get_data().T
                print("Finished parsing nifti")
            else:
                print("Not a valid file extension")


        #Add the shape of the image to the class object for future use
        self.data.image_shape = self.data.image.shape
        #Add the image size to the data object -- note, nifti is transposed!
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
        self.data.filename = path_to_nifti

        #Print an update that the import is finished
        print('Finished')

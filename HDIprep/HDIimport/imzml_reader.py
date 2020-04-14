#Module for imzML parsing of imaging mass spectrometry (IMS) data
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import external modules
from pathlib import Path
import os
from pyimzml.ImzMLParser import getionimage
from pyimzml.ImzMLParser import ImzMLParser
import numpy as np
import pandas as pd

#Import custom modules
from .utils import SubsetCoordinates


#Create a class object to store attributes and functions in
class imzMLreader:
    """Class for parsing and storing IMS data that is in the imzML format. Depends
    on and contains the pyimzML python package distributed from the Alexandrov team:
    https://github.com/alexandrovteam/pyimzML in a data object.

    path_to_imzML: string indicating path to imzML file (Ex: 'path/IMSdata.imzML')
    """

    def __init__(self,path_to_imzML,flatten,subsample,path_to_markers=None,**kwargs):
        """Initialize class to store data in. Ensure appropriate file format
        and return a data object with pixel table.
        """

        #Create a pathlib object for the path_to_imzML
        path_to_imzML = Path(path_to_imzML)

        #Set the file extensions that we can use with this class
        ext = [".imzML"]

        #Check to make sure the string is a valid path
        if not os.path.exists(str(path_to_imzML)):
            print('Not a valid path. Try again')
        else:
            print('Valid path...')
            #Check to see if there is a valid file extension for this class
            if str(path_to_imzML).endswith(tuple(ext)):
                print("Valid file extension...","\nfile name:",str(path_to_imzML),"\nparsing imzML...")
                #Read imzML and return the parsed data
                self.data = ImzMLParser(str(path_to_imzML))
                print("Finished parsing imzML")
            else:
                print("Not a valid file extension")

        #Add the image size to the data object
        self.data.array_size = (self.data.imzmldict["max count of pixels y"],\
            self.data.imzmldict["max count of pixels x"])

        #Check to see if creating a pixel table (used for dimension reduction)
        if flatten:
            #Check to see if subsampling
            if subsample is not None:
                #Subset the coordinates using custom function
                sub_mask, coords = utils.SubsetCoordinates(coords=self.data.coordinates,array_size=self.data.array_size,**kwargs)
                #Clear space with the mask
                sub_mask = None

                #Add the subset coordinates to our object
                self.data.sub_coordinates = coords

            #Otherwise there is no subsampling so leave the coordinates as they are
            else:
                #Keep the full list of coordinates
                coords = self.data.coordinates
                #Add the subset coordinates as None
                self.data.sub_coordinates = None

            #Create numpy array with cols = m/zs an rows = pixels (create pixel table)
            tmp = np.empty([len(coords),\
                len(self.data.getspectrum(0)[0])])
            #iterate through pixels and add to the array
            print('Fetching Spectrum Table...')
            for i, (x,y,z) in enumerate(coords):
                mzs, intensities = self.data.getspectrum(i)
                tmp[i,:] = intensities
                #Clear memory by removing mzs and intensities
                mzs, intensities = None, None

            #Create a pandas dataframe from numpy array
            tmp_frame = pd.DataFrame(tmp,index = coords,\
                columns = self.data.getspectrum(0)[0])
            #Delete the temporary object to save memory
            tmp = None
            #Assign the data to an array in the data object
            self.data.pixel_table = tmp_frame

            #Get the image shape of the data
            self.data.image_shape = (self.data.imzmldict["max count of pixels y"],\
                self.data.imzmldict["max count of pixels x"],self.data.pixel_table.shape[1])
        else:
            #Create a pixel table as None
            self.data.pixel_table = None
            #Set the image shape as None
            self.data.image_shape = None

        #Add the filename to the data object
        self.data.filename = path_to_imzML
        #Add None to the data image (not currently parsing full array)
        self.data.image = None

        #Print an update that the import is finished
        print('Finished')


    def SubsetData(self,range=None):
        """Subset an IMS peak list to fall between a range of values.

        range: tuple indicating range (Ex (400,1000)). Note for memory reasons
        the PixelTable is overwritten, and a new subset of the peak list isnt created.
        """

        #Get the lowest value
        low = next(x for x, val in enumerate(self.data.pixel_table.columns)
                                      if val >= range[0])
        #Get the highest value
        hi = [n for n,i in enumerate(self.data.pixel_table.columns) if i <= range[1]][-1]
        #Assign the new peak list to the pixel_table (add +1 because python isnt inclusive)
        self.data.pixel_table = self.data.pixel_table.iloc[:,low:hi+1]


    def ExportChannels(self):
        """Export a txt file with channel names for downstream analysis.
        """

        #Print a sheet for m/z and channel numbers
        sheet = pd.DataFrame(self.data.pixel_table.columns,columns=["channels"])
        #Write out the sheet to csv
        sheet.to_csv(path_to_imzML.stem+'_channels.csv',sep = "\t")

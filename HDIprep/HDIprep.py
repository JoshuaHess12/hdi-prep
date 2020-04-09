#Module for high-dimensional imaging data importing
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import external modules
from pathlib import Path
import os

#Import custom modules
from HDIimport import HDIimport


#Create class object to store high-dimensional imaging data
class HDIimport:
    """Class for importing high-dimensional imaging data or histology data.
    """

    def __init__(self,path_to_data,path_to_markers,flatten,subsample,mask=None):
        """Initialize class to store data in. Ensure appropriate file format
        and return a data object with pixel table.

        path_to_data: path to imaging data (Ex: path/mydata.extension)
        path_to_markers: path to marker list (Ex: path/mymarkers.csv or None)
        flatten: True to return a flattened pixel data table for dimension reduction
        """

        #Initialize objects
        self.hdi = None

        #Create a pathlib object for the path_to_cyt
        path_to_data = Path(path_to_data)

        #Set the file extensions that we can use with this class
        all_ext = [".ome.tif",".ome.tiff",".tif",".tiff",".h5",".hdf5",".imzML"]
        #Get file extensions for cytometry files
        cyt_ext = [".ome.tif",".ome.tiff",".tif",".tiff",".h5",".hdf5"]
        #Get file exntensions for h(df)5 files
        imzML_ext = [".imzML"]

        #Check to see if there is a valid file extension for this class
        if str(path_to_data).endswith(tuple(cyt_ext)):
            #Read the data with CYTreader
            self.hdi = CYTreader.CYTreader(path_to_cyt = path_to_data, path_to_markers = path_to_markers,\
                flatten = flatten, subsample = subsample, mask = mask)
        #Otherwise read imzML file
        elif str(path_to_data).endswith(tuple(imzML_ext)):
            #Read the data with imzMLreader (CURRENTLY DOES NOT SUPPORT A MASK -- set default to None in class object)
            self.hdi = imzMLreader.imzMLreader(path_to_imzML = path_to_data, path_to_markers = path_to_markers,\
                subsample = subsample, flatten = flatten)
        #If none of the above print an update and an error
        else:
            #Raise an error saying that the file extension is not recognized
            raise ValueError("File extension not recognized.")

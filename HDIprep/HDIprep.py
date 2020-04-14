#Module for high-dimensional imaging data importing
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import external modules
from pathlib import Path
import os

#Import custom modules
from HDIimport import hdi_reader

hdi_reader.HDIreader


im1 = hdi_reader.HDIreader(path_to_data = "/Users/joshuahess/Desktop/tmp/image.ome.tiff",\
    path_to_markers=None,flatten=True,subsample=True,mask=None,n=100,method="random")

im2 = hdi_reader.HDIreader(path_to_data = "/Users/joshuahess/Desktop/tmp/image.tif",\
    path_to_markers=None,flatten=True,subsample=True,mask=None,n=100,method='random')


test = Modalitymerge([im1,im2],modality="IMC")
test.set_dict

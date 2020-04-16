#Class for dimension reduction of high-dimensional imaging data -- dependent on HDIimported data
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import external moduless
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import umap

#Import custom modules
from intramodality_merger import IntraModalityMerger



#Import custom modules
from HDIimport import hdi_reader
im1 = hdi_reader.HDIreader(path_to_data = "/Users/joshuahess/Desktop/tmp/image.ome.tiff",\
    path_to_markers=None,flatten=True,subsample=True,mask=None,n=100,method="random")
im2 = hdi_reader.HDIreader(path_to_data = "/Users/joshuahess/Desktop/tmp/image.tif",\
    path_to_markers=None,flatten=True,subsample=True,mask=None,n=100,method='random')

test = IntraModalityMerger([im1,im2],modality="IMC")

test_dimred = DimensionReduction(test)

test_dimred.dimension_reduction

#Create a class object to store attributes and functions in
class DimensionReduction(IntraModalityMerger):
    """Dimension reduction class for performing linear or manifold based dimension
    reduction on high-dimensional imaging data.
    """

    def __init__(self,IntraModalityMerger):
        """This function will initialize class to store data in.

        HDIimport: Class object from HDIimport module"""

        #Initialize class objects for dimension reduction objects and images
        self.dimension_reduction = None


    def MSIdimRed(self, algorithm="umap", save_as=None, dist_mat = None,**kwargs):
        """Creates an embedding of high-dimensional imaging data. Each
        pixel will be represented by its coordinates scaled from 0-1 after
        hyperspectral image construction.

        Returned will be a numpy array containing pixels and their
        respective embedded coordinates.
        """

        #Check if using a precomputed distance matrix
        if dist_mat is not None:
            #Assign the tmp_frame to be that matrix
            tmp_frame = dist_mat
        else:
            #Get the spectrum SpectrumTable
            tmp_frame = self.hdi.data.pixel_table()


        #Set up UMAP parameters
        UMAP = umap.UMAP(**kwargs).fit(tmp_frame)
        embedding = UMAP.embedding_
        im = self.FillHyperspectral(embedding)
        #Update the umap object
        if save_as is None:
            self.data.UMAP.update({"UMAP":[UMAP,embedding,stop-start]})
            self.data.UMAP_image.update({"UMAP":im})
        else:
            self.data.UMAP.update({str(save_as):[UMAP,embedding,stop-start]})
            self.data.UMAP_image.update({str(save_as):im})



    def FillHyperspectral(self,embedding):
        """Fill a hyperspectral image from n-dimensional embedding of high-dimensional
        imaging data.

        embedding: Embedding resulting from dimension reduction
        """

        #Create zeros array to fill with number channels equal to embedding dimension
        im = np.zeros((self.hdi.data.array_size[0],\
            self.hdi.data.array_size[1],embedding.shape[1]), dtype = np.float32)
        #Run through the data coordinates and fill array
        for i, (x, y, z) in enumerate(self.hdi.data.coordinates):
            #Run through each slice of embedding (dimension)
            for dim in range(embedding.shape[1]):
                im[y - 1, x - 1,dim] = embedding[i,dim]

        #Create a mask to use for excluding pixels not used in dimension reduction
        im_bool = np.zeros((self.hdi.data.array_size[0],\
            self.hdi.data.array_size[1]), dtype = np.bool)
        #Fill the mask array with True values at the coordinates used
        for i, (x, y, z) in enumerate(self.hdi.data.coordinates):
            im_bool[y - 1, x - 1] = True

        #Scale the data 0-1 for hyperspectral image construction
        for dim in range(im.shape[2]):
            #min-max scaler
            im[:,:,dim]=(im[:,:,dim]-im[:,:,dim].min())/(im[:,:,dim].max()-im[:,:,dim].min())
        #Mask the image with the boolean array to remove unused pixels
        im[~im_bool] = 0

        #Return the hyperspectral image
        return im

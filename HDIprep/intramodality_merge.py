#Class for merging data within a modality
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import external moduless
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import umap
import scipy.sparse

#Import custom modules
from HDIimport import hdi_reader


im1 = hdi_reader.HDIreader(path_to_data = "/Users/joshuahess/Desktop/tmp/image.ome.tiff",\
    path_to_markers=None,flatten=True,subsample=True,mask="/Users/joshuahess/Desktop/tmp/MaskCircle.tif",method="grid",grid_spacing=(3,3))
#im2 = hdi_reader.HDIreader(path_to_data = "/Users/joshuahess/Desktop/tmp/image2.ome.tiff",\
#    path_to_markers=None,flatten=True,subsample=True,mask=None,method="random",n=100)

test = IntraModalityDataset([im1],modality="IMC")


import time
start = time.time()
test.RunUMAP(n_components=3)
stop=time.time()
print(str(stop-start))

start = time.time()
hyp = test.SpatiallyMapUMAP()
stop=time.time()
print(str(stop-start))

plt.imshow(hyp)
skimage.io.imsave('SingleCell.tif',hyp,plugin='tifffile')


#Create a class for storing multiple datasets for a single modality
class IntraModalityDataset:
    """Merge HDIimport classes storing imaging datasets
    """

    #Create initialization
    def __init__(self,list_of_HDIimports, modality):
        """initialization function taking list of HDIimport class objects.

        list_of_HDIimports: a list containing files to be merged (HDIimport classes)
        modality: string indicating the modality name (Ex: "IMS", "IMC", or "H&E")
        """

        #Create objects
        self.set_dict = {}
        self.modality = str(modality)
        self.UMAP_object = None
        self.UMAP_embeddings = {}

        #Iterate through the list of HDIimports and add them to the set dictionary
        for dat in list_of_HDIimports:
            #Update the dictionary with keys being filenames
            self.set_dict.update({dat.hdi.data.filename:dat})


    #Create dimension reduction method
    def RunUMAP(self, **kwargs):
        """Creates an embedding of high-dimensional imaging data. Each
        pixel will be represented by its coordinates scaled from 0-1 after
        hyperspectral image construction.

        Returned will be a numpy array containing pixels and their
        respective embedded coordinates.
        """

        #Create a dictionary to store indices in
        file_idx = {}
        #Create a counter
        idx = 0
        #Create a list to store data tables in
        pixel_list = []
        #Create a blank frame
        tmp_frame = pd.DataFrame()

        #Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            #Get the number of rows in the spectrum table
            nrows = hdi_imp.hdi.data.pixel_table.shape[0]
            #update the list of concatenation indices with filename
            file_idx.update({f:(idx,idx+nrows)})
            #Update the index
            idx = idx+nrows

            #Get the spectrum SpectrumTable
            tmp_frame = pd.concat([tmp_frame,hdi_imp.hdi.data.pixel_table])
            #Clear the old pixel table from memory
            hdi_imp.hdi.data.pixel_table = None

        #print updates
        print('Running UMAP on concatenated '+str(self.modality)+' data...')
        #Set up UMAP parameters
        UMAP = umap.UMAP(**kwargs).fit(tmp_frame)
        #print update
        print('Finished UMAP')

        #Unravel the UMAP embedding for each sample
        for f, tup in file_idx.items():
            #Check to see if file has subsampling
            if self.set_dict[f].hdi.data.sub_coordinates is not None:
                #Extract the corresponding index from  UMAP embedding with subsample coordinates
                self.UMAP_embeddings.update({f:pd.DataFrame(UMAP.embedding_[tup[0]:tup[1],:],\
                    index = self.set_dict[f].hdi.data.sub_coordinates)})
            else:
                #Otherwise use the full coordinates list
                self.UMAP_embeddings.update({f:pd.DataFrame(UMAP.embedding_[tup[0]:tup[1],:],\
                    index = self.set_dict[f].hdi.data.coordinates)})

        #Add the umap object to the class
        self.UMAP_object = UMAP


    def SpatiallyMapUMAP(self):
        """Spatially fill arrays based on UMAP embeddings. Must be run after RunUMAP.
        """

        #Check to make sure that UMAP object in class is not empty
        if self.UMAP_object is None:
            #Raise an error
            raise ValueError("Spatially mapping an embedding is not possible yet! Please run UMAP first.")

        #Run through each object in the set dictionary
        for f, pixs in self.UMAP_embeddings.items():

            print("working on "+str(f)+'...')

            #Get the inverse pixels
            inv_pix = list(set(self.set_dict[f].hdi.data.coordinates).difference(set(list(pixs.index))))

            #Create a mask based off array size and current UMAP data points
            data = np.ones(len(inv_pix),dtype=np.bool)
            #Create row data for scipy coo matrix (-1 index for 0-based python)
            row = np.array([inv_pix[c][1]-1 for c in range(len(inv_pix))])
            #Create row data for scipy coo matrix (-1 index for 0-based python)
            col = np.array([inv_pix[c][0]-1 for c in range(len(inv_pix))])

            #Create a sparse mask from data and row column indices
            sub_mask = scipy.sparse.coo_matrix((data, (row,col)), shape=self.set_dict[f].hdi.data.array_size)

            #Remove the other objects used to create the mask to save memory
            data, row, col = None, None, None

            #Read the file and use the mask to create complementary set of pixels
            new_data = hdi_reader.HDIreader(path_to_data = f,\
                path_to_markers=None,flatten=True,subsample=False,mask=sub_mask.toarray())

            #Remove the mask to save memory
            sub_mask = None

            #print update
            print('Transforming pixels into existing UMAP embedding...')
            #Run the new pixel table through umap transformer
            embedding_projection = self.UMAP_object.transform(new_data.hdi.data.pixel_table)
            #Add the projection to dataframe and coerce with existing embedding
            embedding_projection = pd.DataFrame(embedding_projection,index = new_data.hdi.data.coordinates)

            #Remove the new data to save memory
            new_data = None

            #Concatenate with existing UMAP object
            self.UMAP_embeddings[f] = pd.concat([pixs,embedding_projection])
            #print update
            print('Finished projection')

            #Use the new embedding to map coordinates to the image
            hyper_im = CreateHyperspectralImage(embedding = self.UMAP_embeddings[f],\
                array_size = self.set_dict[f].hdi.data.array_size,coordinates = list(self.UMAP_embeddings[f].index))

        #return the image for now just to check output
        return hyper_im



def CreateHyperspectralImage(embedding,array_size,coordinates):
    """Fill a hyperspectral image from n-dimensional embedding of high-dimensional
    imaging data.

    embedding: Embedding resulting from dimension reduction
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

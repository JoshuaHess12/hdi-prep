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
from operator import itemgetter


#Import custom modules
from HDIimport import hdi_reader
from utils import CreateHyperspectralImage, ExportNifti



###TESTING IMC####
import skimage.io
import matplotlib.pyplot as plt
im1 = hdi_reader.HDIreader(path_to_data = "/Users/joshuahess/Desktop/tmp/image.ome.tiff",\
    path_to_markers=None,flatten=True,subsample=None,mask=None)
skimage.io.imsave("small_im2.tif",np.swapaxes(im1.hdi.data.image[:400,:500,:],0,2),plugin='tifffile')



#****Test iMC one more time :)

#Run for full data
im1 = hdi_reader.HDIreader(path_to_data = "/Users/joshuahess/Desktop/small_im.tif",\
    path_to_markers=None,flatten=True,subsample=None)
#Run for subsampled data
im2 = hdi_reader.HDIreader(path_to_data = "/Users/joshuahess/Desktop/small_im2.tif",\
    path_to_markers=None,flatten=True,subsample=None)
#Concatenate the data
test1 = IntraModalityDataset([im1,im2],modality="IMC")
test1.RunUMAP(n_components = 3)
results = test1.SpatiallyMapUMAP()
plt.imshow(results[0])
plt.imshow(results[1])


#Run for subsampled data
im3 = hdi_reader.HDIreader(path_to_data = "/Users/joshuahess/Desktop/small_im.tif",\
    path_to_markers=None,flatten=True,method='grid',grid_spacing=(2,2),subsample=True)
#Run for subsampled data
im4 = hdi_reader.HDIreader(path_to_data = "/Users/joshuahess/Desktop/small_im2.tif",\
    path_to_markers=None,flatten=True,method='grid',grid_spacing=(2,2),subsample=True)
#Concatenate the data
test2 = IntraModalityDataset([im3,im4],modality="IMC")
test2.RunUMAP(n_components = 3)
results2 = test2.SpatiallyMapUMAP()
plt.imshow(results2[0])
plt.imshow(results2[1])










#Get the sample name
key = list(test1.set_dict.keys())[0]

#Get the distances between objects -- assuming pixel order is the same
from scipy.spatial.distance import pdist
embed_dist = pdist(test1.UMAP_embeddings[key].iloc[:3500])
og_dist = pdist(test2.UMAP_embeddings[key].iloc[:3500])
#Calculate the spearman correlation between the two distance matrices
from scipy.stats import spearmanr, pearsonr
pearsonr(og_dist, embed_dist)




from scipy.spatial import procrustes
test3_embedding = procrustes(test1.UMAP_embeddings[key],test2.UMAP_embeddings[key])
new_image = CreateHyperspectralImage(pd.DataFrame(test3_embedding[1]),test2.set_dict[key].hdi.data.array_size,list(test2.UMAP_embeddings[key].index))

plt.imshow(new_image)

plt.scatter(test3_embedding[1][:10500,0],test3_embedding[1][:10500,1])

plt.scatter(test1.UMAP_embeddings[key].iloc[:10500,0],test1.UMAP_embeddings[key].iloc[:10500,1])
plt.scatter(test2.UMAP_embeddings[key].iloc[:10500,0],test2.UMAP_embeddings[key].iloc[:10500,1])

####################








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

            #Get the spectrum
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
                #Here, ensure that the appropriate order for the embedding is given (c-style...imzml parser is fortran)
                self.UMAP_embeddings[f] = self.UMAP_embeddings[f].reindex(sorted(list(self.UMAP_embeddings[f].index), key = itemgetter(1, 0)))

        #Add the umap object to the class
        self.UMAP_object = UMAP


    def SpatiallyMapUMAP(self):
        """Spatially fill arrays based on UMAP embeddings. Must be run after RunUMAP.
        """

        #Check to make sure that UMAP object in class is not empty
        if self.UMAP_object is None:
            #Raise an error
            raise ValueError("Spatially mapping an embedding is not possible yet! Please run UMAP first.")


        results_list = []


        #Run through each object in the set dictionary
        for f, locs in self.UMAP_embeddings.items():

            print("working on "+str(f)+'...')

            #Check to see if there is subsampling
            if self.set_dict[f].hdi.data.sub_coordinates is not None:

                #Get the inverse pixels
                inv_pix = list(set(self.set_dict[f].hdi.data.coordinates).difference(set(list(locs.index))))

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
                    path_to_markers=None,flatten=True,subsample=None,mask=sub_mask.toarray())

                #Remove the mask to save memory
                sub_mask = None

                #print update
                print('Transforming pixels into existing UMAP embedding...')
                #Run the new pixel table through umap transformer
                embedding_projection = self.UMAP_object.transform(new_data.hdi.data.pixel_table)
                #Add the projection to dataframe and coerce with existing embedding
                embedding_projection = pd.DataFrame(embedding_projection,index = list(new_data.hdi.data.pixel_table.index))

                #Remove the new data to save memory
                new_data = None

                #Concatenate with existing UMAP object
                self.UMAP_embeddings[f] = pd.concat([locs,embedding_projection])

                #Reindex data frame to row major orientation
                self.UMAP_embeddings[f] = self.UMAP_embeddings[f].reindex(sorted(list(self.UMAP_embeddings[f].index), key = itemgetter(1, 0)))

                #Use the new embedding to map coordinates to the image
                hyper_im = CreateHyperspectralImage(embedding = self.UMAP_embeddings[f],\
                    array_size = self.set_dict[f].hdi.data.array_size,coordinates = list(self.UMAP_embeddings[f].index))

            else:
                #Use the new embedding to map coordinates to the image
                hyper_im = CreateHyperspectralImage(embedding = self.UMAP_embeddings[f],\
                    array_size = self.set_dict[f].hdi.data.array_size,coordinates = list(self.UMAP_embeddings[f].index))

            #Update list
            results_list.append(hyper_im)


        #print update
        print('Finished spatial mapping')

        return results_list

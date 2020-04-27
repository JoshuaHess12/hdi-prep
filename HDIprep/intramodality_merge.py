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
from utils import CreateHyperspectralImage, ExportNifti







#####################
import skimage
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import sklearn
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from mayavi import mlab


mlab.options.offscreen = False






im1 = hdi_reader.HDIreader(path_to_data = "/Users/joshuahess/Desktop/tmp/image.ome.tiff",\
    path_to_markers=None,flatten=True,subsample=False,mask=None)

test = im1.hdi.data.image[200:400,200:400,:]
test = np.swapaxes(test,0,2)
skimage.io.imsave('/Users/joshuahess/Desktop/small_im2.tif',test,plugin='tifffile')


#Testing UMAP script
im1 = hdi_reader.HDIreader(path_to_data = '/Users/joshuahess/Desktop/small_im.tif',path_to_markers=None,flatten=True,subsample=None,mask=None)
#Testing UMAP script
im2 = hdi_reader.HDIreader(path_to_data = '/Users/joshuahess/Desktop/small_im2.tif',path_to_markers=None,flatten=True,subsample=None,mask=None)
test = IntraModalityDataset([im1,im2],modality="IMC")
test.RunUMAP(n_components=2,n_neighbors=5,random_state=5)
hyp = test.SpatiallyMapUMAP()

key = list(test.UMAP_embeddings.keys())[0]

knn_dist1 = sklearn.neighbors.kneighbors_graph(test.UMAP_embeddings[key], \
    n_neighbors=500, mode='distance', metric='minkowski', p=2, metric_params=None, include_self=False, n_jobs=None)

booleanarr1 = knn_dist1.astype(np.bool)


plt.imshow(hyp[0][:,:,0])
plt.imshow(hyp[1][:,:,0])



im3 = hdi_reader.HDIreader(path_to_data = "/Users/joshuahess/Desktop/small_im.tif",\
        path_to_markers=None,flatten=True,subsample=True,mask=None,method="grid",grid_spacing=(10,10))
im4 = hdi_reader.HDIreader(path_to_data = "/Users/joshuahess/Desktop/small_im2.tif",\
        path_to_markers=None,flatten=True,subsample=True,mask=None,method="grid",grid_spacing=(10,10))
test2 = IntraModalityDataset([im3,im4],modality="IMC")
test2.RunUMAP(n_components=2,n_neighbors=5,random_state=5)
hyp2 = test2.SpatiallyMapUMAP()

plt.imshow(hyp2[0][:,:,0])
plt.imshow(hyp2[1][:,:,0])






jacc_full = []
testing_list = []
spear_local = []
spear_local_full = []
for i in range(booleanarr2.shape[1]):

    #Get indices of full UMAP knn
    tmp = booleanarr1[i].nonzero()

    testing_list.append(scipy.spatial.distance.jaccard(booleanarr1[i][tmp], booleanarr2[i][tmp]))
    spear_local.append(spearmanr(knn_dist1[i].toarray().flatten(), knn_dist2[i].toarray().flatten()).correlation)

jacc_full.append(np.array(testing_list).mean())
spear_local_full.append(np.array(spear_local).mean())


plt.scatter()
plt.imshow(hyp2[1][:,:,1])
plt.imshow(hyp2[0][:,:,1])

plt.scatter(test2.UMAP_embeddings[key].values[:,0],test2.UMAP_embeddings[key].values[:,1])
plt.scatter(test.UMAP_embeddings[key].values[:,0],test.UMAP_embeddings[key].values[:,1])

embed_dist = pdist(test2.UMAP_embeddings[key].reindex(list(test.UMAP_embeddings[key].index)))

og_dist = pdist(test.UMAP_embeddings[key])
spearmanr(og_dist, embed_dist).correlation

##############################
g = nx.from_scipy_sparse_matrix(knn_dist1, create_using=nx.Graph())

# numpy array of x,y,z positions in sorted node order
xyz = np.array(test.UMAP_embeddings[key])
# scalar colors

scalars = np.array(list(g.nodes())) + 5

mlab.figure(1, bgcolor=(0, 0, 0))
mlab.clf()

pts = mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    scalars,
                    scale_factor=0.1,
                    scale_mode='none',
                    colormap='Blues',
                    resolution=400)

pts.mlab_source.dataset.lines = np.array(list(g.edges()))
tube = mlab.pipeline.tube(pts, tube_radius=0.01)
mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))

mlab.savefig('mayavi2_spring.png',size=(400,400))


#############################




spear_full = []
jacc_full = []
spear_local = []
for i in [2,5,10,25,35,45,55,65,75,85]:

    print('Working on grid spacing '+str(i))

    im2 = hdi_reader.HDIreader(path_to_data = "/Users/joshuahess/Desktop/tmp/DFU_Trimmed_700-850/dfu trimmed_tic.imzML",\
        path_to_markers=None,flatten=True,subsample=None,mask=sub_mask.toarray())

    im2 = hdi_reader.HDIreader(path_to_data = "/Users/joshuahess/Desktop/small_im.tif",\
        path_to_markers=None,flatten=True,subsample=True,mask=None,method="grid",grid_spacing=(10,10))


    test2 = IntraModalityDataset([im2],modality="IMC")
    test2.RunUMAP(n_components=2,n_neighbors=5,random_state=5)
    hyp2 = test2.SpatiallyMapUMAP()

    #Calculate embedding spearman global correlation

    knn_dist2 = sklearn.neighbors.kneighbors_graph(test2.UMAP_embeddings[key].reindex(list(test.UMAP_embeddings[key].index)),\
        n_neighbors=500, mode='distance', metric='minkowski', p=2, metric_params=None, include_self=False, n_jobs=None)

    booleanarr2 = knn_dist2.astype(np.bool)

    testing_list = []
    for i in range(booleanarr2.shape[1]):

        testing_list.append(scipy.spatial.distance.jaccard(booleanarr2[i,:].toarray().flatten(), booleanarr1[i,:].toarray().flatten()))

    jacc_full.append(np.array(testing_list).mean())

    #plt.imshow(1-np.array(testing_list).reshape(350,350),cmap='plasma')

    #spear_local.append(spearmanr(knn_dist1.toarray().flatten(), knn_dist2.toarray().flatten()).correlation)







plt.imshow(hyp2[:,:,3])





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


        results_list = []


        #Run through each object in the set dictionary
        for f, pixs in self.UMAP_embeddings.items():

            print("working on "+str(f)+'...')

            #Check to see if there is subsampling
            if self.set_dict[f].hdi.data.sub_coordinates is not None:

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
                    path_to_markers=None,flatten=True,subsample=None,mask=sub_mask.toarray())

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

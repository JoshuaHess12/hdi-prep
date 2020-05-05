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
import skimage
from operator import itemgetter


#Import custom modules
from HDIimport import hdi_reader
from utils import CreateHyperspectralImage, ExportNifti, MedFilter



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
        self.umap_object = None
        self.umap_embeddings = {}
        self.umap_spatial_images = None
        self.umap_spatial_images_export = None
        self.processed_images_export = None

        #Iterate through the list of HDIimports and add them to the set dictionary
        for dat in list_of_HDIimports:
            #Update the dictionary with keys being filenames
            self.set_dict.update({dat.hdi.data.filename:dat})


    #Create dimension reduction method with UMAP
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
                self.umap_embeddings.update({f:pd.DataFrame(UMAP.embedding_[tup[0]:tup[1],:],\
                    index = self.set_dict[f].hdi.data.sub_coordinates)})
            else:
                #Otherwise use the full coordinates list
                self.umap_embeddings.update({f:pd.DataFrame(UMAP.embedding_[tup[0]:tup[1],:],\
                    index = self.set_dict[f].hdi.data.coordinates)})
                #Here, ensure that the appropriate order for the embedding is given (c-style...imzml parser is fortran)
                self.umap_embeddings[f] = self.umap_embeddings[f].reindex(sorted(list(self.umap_embeddings[f].index), key = itemgetter(1, 0)))

        #Add the umap object to the class
        self.umap_object = UMAP


    #Add function for creating hyperspectral image from UMAP
    def SpatiallyMapUMAP(self):
        """Spatially fill arrays based on UMAP embeddings. Must be run after RunUMAP.
        """

        #Check to make sure that UMAP object in class is not empty
        if self.umap_object is None:
            #Raise an error
            raise ValueError("Spatially mapping an embedding is not possible yet! Please run UMAP first.")

        #For now, create a dictionary to store the results in
        results_dict = {}

        #Run through each object in the set dictionary
        for f, locs in self.umap_embeddings.items():

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
                    path_to_markers=None,flatten=True,subsample=None,mask=sub_mask)

                #Remove the mask to save memory
                sub_mask = None

                #print update
                print('Transforming pixels into existing UMAP embedding...')
                #Run the new pixel table through umap transformer
                embedding_projection = self.umap_object.transform(new_data.hdi.data.pixel_table)
                #Add the projection to dataframe and coerce with existing embedding
                embedding_projection = pd.DataFrame(embedding_projection,index = list(new_data.hdi.data.pixel_table.index))

                #Remove the new data to save memory
                new_data = None

                #Concatenate with existing UMAP object
                self.umap_embeddings[f] = pd.concat([locs,embedding_projection])

                #Reindex data frame to row major orientation
                self.umap_embeddings[f] = self.umap_embeddings[f].reindex(sorted(list(self.umap_embeddings[f].index), key = itemgetter(1, 0)))

                #Use the new embedding to map coordinates to the image
                hyper_im = CreateHyperspectralImage(embedding = self.umap_embeddings[f],\
                    array_size = self.set_dict[f].hdi.data.array_size,coordinates = list(self.umap_embeddings[f].index))

            else:
                #Use the new embedding to map coordinates to the image
                hyper_im = CreateHyperspectralImage(embedding = self.umap_embeddings[f],\
                    array_size = self.set_dict[f].hdi.data.array_size,coordinates = list(self.umap_embeddings[f].index))

            #Update list
            results_dict.update({f:hyper_im})

        #print update
        print('Finished spatial mapping')

        #Add the dictionary to the class object
        self.umap_spatial_images = results_dict

        #Return the resulting images
        return results_dict


    #Add function for exporting UMAP nifti image
    def ExportNiftiUMAP(self,output_dir,padding=None):
        """Exporting hyperspectral images resulting from UMAP and
        spatially mapping UMAP

        filename: input filename to use ExportNifti function from utils.
        - path (filename) of resulting exporting image (Ex: path/to/new/image.nii or image.nii)
        padding: tuple indicating height and length 0 pixels padding to add to the image before exporting
        """

        #Create dictionary with connected file names
        connect_dict = {}

        #Iterate through the results dictionary from spatially mapping UMAP
        for f, img in self.umap_spatial_images.items():
            #Create an image name -- remove .ome in the name if it exists and add umap suffix
            im_name = Path(os.path.join(output_dir,f.stem.replace(".ome.","")+"_umap.nii"))
            #Use utils export nifti function
            ExportNifti(img,im_name,padding)
            #Add exported file names to class object -- connect input file name with the exported name
            connect_dict.update({f:im_name})

        #Add the connecting dictionary to the class object
        self.umap_spatial_images_export = connect_dict

        #return the dictionary of input names to output names
        return connect_dict


    #Create definition for image filtering and processing
    def ApplyManualMask(self):
        """Applying the input mask to image. This function is
        primarily used on histology images and images that do not need dimension
        reduction. Dimension reduction with a mask will by default zero all other pixels
        in the image outside of the mask. Use this function if not performing dimension
        reduction.
        """

        #Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            #Ensure that the mask is not none
            if hdi_imp.hdi.data.mask is None:
                #Skip this image if there is no mask
                continue

            #Use the mask on the image and replace the image with the masked image
            hdi_imp.hdi.data.image[~hdi_imp.hdi.data.mask.toarray()] = 0


    def MedianFilter(self,filter_size,parallel=False):
        """Median filtering of images to remove salt and pepper noise.
        A circular disk is used for the filtering. Images that are not single channel
        are automatically converted to grayscale prior to filtering

        filter_size: size of disk to use for filter.
        n_jobs: number of proceses to use for calculations
        """

        #Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            #Ensure that the mask is not none
            if hdi_imp.hdi.data.image is None:
                #Skip this image if there is no mask
                continue

            #Check to see if the preprocessed is initiated
            if hdi_imp.hdi.data.processed_image is None:
                #If not, use the original image
                hdi_imp.hdi.data.processed_image = MedFilter(hdi_imp.hdi.data.image,filter_size,parallel)

            #Use the initiated image
            else:
                #Use the processed image
                hdi_imp.hdi.data.processed_image = MedFilter(hdi_imp.hdi.data.processed_image,filter_size,parallel)



    def Threshold(self,image,type='otsu',thresh_value):
        """This function will convert your image to a grayscale version and will
        perform otsu thresholding on your image to form a mask"""



    def Open(self,disk_size,parallel=False):
        """Morphological opening"""



    def Close(self,disk_size,parallel=False):
        """Morphological closing"""


    def Fill(self):
        """Morphological fill"""



#Read the data using hdi_reader and access all files
og_dat = hdi_reader.HDIreader(path_to_data = "/Users/joshuahess/Desktop/tmp/15gridspacing.tif",\
                path_to_markers=None,flatten=True,subsample=None,mask ="/Users/joshuahess/Desktop/tmp/MaskCircle.tif")
import matplotlib.pyplot as plt
plt.imshow(og_dat.hdi.data.image)

len(test.set_dict[key].hdi.data.processed_image.shape)

test = IntraModalityDataset([og_dat],"test")
test.MedianFilter(50,parallel=True)
#Get the key of the image
key = list(test.set_dict.keys())[0]
plt.imshow(test.set_dict[key].hdi.data.processed_image)
import skimage.filters

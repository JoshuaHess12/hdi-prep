#Class for merging data within a modality
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import external moduless
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import scipy.sparse
import skimage
import seaborn as sns
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit
from ast import literal_eval
from operator import itemgetter
import uncertainties.unumpy as unp
import uncertainties as unc

#Import custom modules
from HDIimport import hdi_reader
import fuzzy_operations as fuzz
import morphology
import utils


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
        self.umap_optimal_dim = None
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
            #hdi_imp.hdi.data.pixel_table = None

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



    def RunOptimalUMAP(self,dim_range,export_diagnostics=False, output_dir=None, n_jobs=1,**kwargs):
        """Run UMAP over a range of dimensions to choose optimal embedding by empirical
        observation of fuzzy set cross entropy.

        dim_range: tuple indicating a range of embedding dimensions (Ex: (1,11) is 1-10 -- python range)
        normalized_cutoff: error delta to stop embedding -- not always going to converge, so all embedding
        dimensions will be tested. If cutoff is not reached, a warning will appear, and the lowest delta
        will be used
        **kwargs: extra parameters to pass to UMAP (Ex: n_neighbors = 15, random_state = 223)
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
            #hdi_imp.hdi.data.pixel_table = None

        #Create list to store the results in
        ce_res = {}
        #Create a dictionary to store the embeddings in
        embed_dict = {}

        #Check to see if the dim_range is a string
        if isinstance(dim_range,str):
            dim_range = literal_eval(dim_range)

        #Set up the dimension range for UMAP
        dim_range = range(dim_range[0],dim_range[1])

        #Run UMAP on the first iteration -- we will skip simplicial set construction in next iterations
        UMAP_0 = umap.UMAP(n_components = dim_range[0],**kwargs)
        #Fit the concatenated dataframe
        UMAP = UMAP_0.fit(tmp_frame)
        #Update the embedding dictionary
        embed_dict.update({dim_range[0]:UMAP.embedding_})

        #Compute the fuzzy set cross entropy
        cs = fuzz.FuzzySetCrossEntropy(UMAP.embedding_,UMAP.graph_,UMAP.min_dist,n_jobs)
        #Update the results for fuzzy cross entropy
        ce_res.update({dim_range[0]:cs})

        #Iterate through each subsequent embedding dimension -- add +1 because we have already used range
        for dim in range(dim_range[1],dim_range[-1]+1):

            #Print update for this dimension
            print('Embedding in dimension '+str(dim))
            #Use previous simplicial set and embedding components to embed in higher dimension
            alt_embed = umap.umap_.simplicial_set_embedding(data = UMAP._raw_data,\
                        graph = UMAP.graph_,\
                        n_components = dim,\
                        initial_alpha = UMAP._initial_alpha,\
                        a = UMAP._a,\
                        b = UMAP._b,\
                        gamma = UMAP.repulsion_strength,\
                        negative_sample_rate = UMAP.negative_sample_rate,\
                        #Default umap behavior is n_epochs None -- converts to 0
                        n_epochs = 0,\
                        init = UMAP_0.init,\
                        random_state = check_random_state(UMAP.random_state),\
                        metric = UMAP._input_distance_func,\
                        metric_kwds = UMAP._metric_kwds,\
                        output_metric = UMAP._output_distance_func,\
                        output_metric_kwds = UMAP._output_metric_kwds,\
                        euclidean_output = UMAP.output_metric in ("euclidean", "l2"),\
                        parallel = UMAP.random_state is None,\
                        verbose = UMAP.verbose
                    )
            #Print update
            print('Finished embedding')

            #Update the embedding dictionary
            embed_dict.update({dim:alt_embed})

            #Compute the fuzzy set cross entropy
            cs = fuzz.FuzzySetCrossEntropy(alt_embed, UMAP.graph_,UMAP.min_dist, n_jobs)
            #Update list for now
            ce_res.update({dim:cs})

        #Construct a dataframe from the dictionary of results
        ce_res = pd.DataFrame(ce_res,index=["Cross-Entropy"]).T

        #Print update
        print('Finding optimal embedding dimension through exponential fit...')
        #Calculate the min-max normalized cross-entropy
        ce_res_norm = MinMaxScaler().fit_transform(ce_res)
        #Convert back to pandas dataframe
        ce_res_norm = pd.DataFrame(ce_res_norm,columns=["Scaled Cross-Entropy"],index = [x for x in dim_range])

        #Get the metric values
        met = ce_res_norm["Scaled Cross-Entropy"].values
        #Get the x axis information
        xdata = np.int64(ce_res_norm.index.values)
        #Fit the data using exponential function
        popt, pcov = curve_fit(utils.Exp, xdata, met, p0 = (0, 0.01, 1))

        #create parameters from scipy fit
        a, b, c = unc.correlated_values(popt, pcov)

        #Create a tuple indicating the 95% interval containing the asymptote in c
        asympt = (c.n-c.s,c.n+c.s)

        #create equally spaced samples between range of dimensions
        px = np.linspace(dim_range[0], dim_range[-1]+1, 100000)

        #use unumpy.exp to create samples
        py = a * unp.exp(-b * px) + c
        #extract expected values
        nom = unp.nominal_values(py)
        #extract stds
        std = unp.std_devs(py)

        #Iterate through samples to find the instance that value falls in 95% c value
        for val in range(len(py)):
            #Extract the nominal value
            tmp_nom = py[val].n
            #check if nominal value falls within 95% CI for asymptote
            if (asympt[0] <= tmp_nom <= asympt[1]):
                #break the loop
                break
        #Extract the nominal value at this index -- round up (any float value lower is not observed -- dimensions are int)
        opt_dim = int(np.ceil(px[val]))
        #Print update
        print('Optimal UMAP embedding dimension is '+ str(opt_dim))

        #Check to see if exporting plot
        if export_diagnostics:
            #Ensure that an output directory is entered
            if output_dir is None:
                #Raise and error if no output
                raise(ValueError('Please add an output directory -- none identified'))
            #Create a path based on the output directory
            else:
                #Create image path
                im_path = Path(os.path.join(output_dir,'OptimalUMAP.jpeg'))
                #Create csv path
                csv_path = Path(os.path.join(output_dir,'OptimalUMAP.csv'))

            #Plot figure and save results
            fig, axs = plt.subplots()
            #plot the fit value
            axs.plot(px, nom, c='r',label="Fitted Curve",linewidth=3)
            #add 2 sigma uncertainty lines
            axs.plot(px, nom - 2 * std, c='c',label="95% CI",alpha=0.6,linewidth=3)
            axs.plot(px, nom + 2 * std, c='c',alpha=0.6,linewidth=3)
            #plot the observed values
            axs.plot(xdata, met, 'ko', label="Observed Data",markersize=8)
            #Change axis names
            axs.set_xlabel('Dimension')
            axs.set_ylabel('Min-Max Scaled Cross-Entropy')
            fig.suptitle('Optimal Dimension Estimation', fontsize=12)
            axs.legend()
            #plt.show()
            plt.savefig(im_path,dpi=600)

            #Export the metric values to csv
            ce_res_norm.to_csv(csv_path)

        #Use the optimal UMAP embedding to add to the class object
        UMAP.embedding_ = embed_dict[opt_dim]

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

        #Update the optimal dimensionality
        self.umap_optimal_dim = opt_dim



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
                hyper_im = utils.CreateHyperspectralImage(embedding = self.umap_embeddings[f],\
                    array_size = self.set_dict[f].hdi.data.array_size,coordinates = list(self.umap_embeddings[f].index))

            else:
                #Use the new embedding to map coordinates to the image
                hyper_im = utils.CreateHyperspectralImage(embedding = self.umap_embeddings[f],\
                    array_size = self.set_dict[f].hdi.data.array_size,coordinates = list(self.umap_embeddings[f].index))

            #Update list
            results_dict.update({f:hyper_im})

            #add this hyperspectral image to the hdi_import object as processed_image
            self.set_dict[f].hdi.data.processed_image = hyper_im

        #print update
        print('Finished spatial mapping')

        #Return the resulting images
        #return results_dict



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
            #Ensure that the image itself is not none
            if hdi_imp.hdi.data.image is None:
                #Skip this image if there is no mask
                continue

            #Check to see if the preprocessed is initiated
            if hdi_imp.hdi.data.processed_image is None:
                #If not, use the original image
                hdi_imp.hdi.data.processed_image = hdi_imp.hdi.data.image.copy()
                #Use the mask on the image
                hdi_imp.hdi.data.processed_image[~hdi_imp.hdi.data.mask.toarray()] = 0
            #Otherwise the processed image exists and now check the data type
            else:
                #Proceed to process the processed image as an array
                if isinstance(hdi_imp.hdi.data.processed_image, scipy.sparse.coo_matrix):
                    #Convert to array
                    hdi_imp.hdi.data.processed_image = hdi_imp.hdi.data.processed_image.toarray()

                    #Use the mask on the image
                    hdi_imp.hdi.data.processed_image[~hdi_imp.hdi.data.mask.toarray()] = 0
                    #Turn the processed mask back to sparse matrix
                    hdi_imp.hdi.data.processed_image = scipy.sparse.coo_matrix(hdi_imp.hdi.data.processed_image,dtype=np.bool)



    def MedianFilter(self,filter_size,parallel=False):
        """Median filtering of images to remove salt and pepper noise.
        A circular disk is used for the filtering. Images that are not single channel
        are automatically converted to grayscale prior to filtering

        filter_size: size of disk to use for filter.
        parallel: parallel processing using all processors or not
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
                hdi_imp.hdi.data.processed_image = morphology.MedFilter(hdi_imp.hdi.data.image,filter_size,parallel)

            #Use the initiated image
            else:
                #Use the processed image
                hdi_imp.hdi.data.processed_image = morphology.MedFilter(hdi_imp.hdi.data.processed_image,filter_size,parallel)



    def Threshold(self,type,thresh_value=None,correction=1.0):
        """Otsu (manual) thresholding of grayscale images. Returns a sparse boolean
        mask.

        image: numpy array that represents image
        type: Type of thresholding to use. Options are 'manual' or "otsu"
        thresh_value: If manual masking, insert a threshold value
        correction: Correction factor after thresholding. Values >1 make the threshold more
        stringent. By default, value with be 1 (identity)
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
                hdi_imp.hdi.data.processed_image = morphology.Thresholding(hdi_imp.hdi.data.image,type,thresh_value,correction)

            #Use the initiated image
            else:
                #Use the processed image
                hdi_imp.hdi.data.processed_image = morphology.Thresholding(hdi_imp.hdi.data.processed_image,type,thresh_value,correction)



    def Open(self,disk_size,parallel=False):
        """Morphological opening on boolean array (mask). A circular disk is used for the filtering.

        disk_size: size of disk to use for filter.
        parallel: number of proceses to use for calculations
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
                hdi_imp.hdi.data.processed_image = morphology.Opening(hdi_imp.hdi.data.image,disk_size,parallel)

            #Use the initiated image
            else:
                #Use the processed image
                hdi_imp.hdi.data.processed_image = morphology.Opening(hdi_imp.hdi.data.processed_image,disk_size,parallel)



    def Close(self,disk_size,parallel=False):
        """Morphological closing on boolean array (mask). A circular disk is used for the filtering.

        disk_size: size of disk to use for filter.
        parallel: number of proceses to use for calculations
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
                hdi_imp.hdi.data.processed_image = morphology.Closing(hdi_imp.hdi.data.image,disk_size,parallel)

            #Use the initiated image
            else:
                #Use the processed image
                hdi_imp.hdi.data.processed_image = morphology.Closing(hdi_imp.hdi.data.processed_image,disk_size,parallel)



    def Fill(self):
        """Morphological filling on a binary mask. Fills holes
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
                hdi_imp.hdi.data.processed_image = morphology.MorphFill(hdi_imp.hdi.data.image)

            #Use the initiated image
            else:
                #Use the processed image
                hdi_imp.hdi.data.processed_image = morphology.MorphFill(hdi_imp.hdi.data.processed_image)



    def NonzeroBox(self):
        """Use a nonzero indices of a binary mask to create a bounding box for
        the mask itself and for the original image. This is to be used so that
        a controlled amount of padding can be added to the edges of the images in
        a consistent manner
        """

        #Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            #Ensure that the mask is not none
            if hdi_imp.hdi.data.image is None:
                #Skip this image if there is no mask
                continue

            #Check to see if the preprocessed is initiated
            if hdi_imp.hdi.data.processed_image is None:
                #Skip this iteration because the processed image must be present
                continue

            #If all conditions are satisfied, use the slicing on the images
            hdi_imp.hdi.data.image,hdi_imp.hdi.data.processed_image = morphology.NonzeroSlice(hdi_imp.hdi.data.processed_image,hdi_imp.hdi.data.image)



    #Create definition for image filtering and processing
    def ApplyMask(self):
        """Applying mask to image. This function is
        primarily used on histology images and images that do not need dimension
        reduction. Dimension reduction with a mask will by default zero all other pixels
        in the image outside of the mask. Use this function if not performing dimension
        reduction. The mask used in this process will be the processed image, and not
        the input mask with the image. Should be used after a series of morphological
        operations
        """

        #Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            #Ensure that the mask is not none
            if hdi_imp.hdi.data.processed_image is None:
                #Skip this image if there is no mask
                continue

            #Create a temporary image based on the current image
            tmp_im = hdi_imp.hdi.data.image.copy()
            #Use the mask on the image and replace the image with the masked image
            tmp_im[~hdi_imp.hdi.data.processed_image.toarray()] = 0
            #Set the processed image as the masked array
            hdi_imp.hdi.data.processed_image = tmp_im



    #Add function for exporting UMAP nifti image
    def ExportNifti1(self,output_dir,padding=None):
        """Exporting hyperspectral images resulting from UMAP and
        spatially mapping UMAP, or exporting processed histology images. Both of these
        conditions cant be true. One or the other will be exported, and it will be
        determined automatically from the class objects

        filename: input filename to use ExportNifti function from utils.
        - path (filename) of resulting exporting image (Ex: path/to/new/image.nii or image.nii)
        padding: tuple indicating height and length 0 pixels padding to add to the image before exporting
        """

        #Create dictionary with connected file names
        connect_dict = {}

        #Iterate through the set dictionary
        for f, hdi_imp in self.set_dict.items():
            #Create an image name -- remove .ome in the name if it exists and add umap suffix
            im_name = Path(os.path.join(output_dir,f.stem.replace(".ome.","")+"_"+str(self.modality)+"_processed.nii"))

            #Ensure that the mask is not none
            if hdi_imp.hdi.data.processed_image is None:
                #Make sure the image exists
                if hdi_imp.hdi.data.image is None:
                    continue
                #Otherwise export the image
                else:
                    #Export the original image
                    utils.ExportNifti(hdi_imp.hdi.data.image,im_name,padding)
            #Otherwise export the processed image
            else:
                #Use utils export nifti function
                utils.ExportNifti(hdi_imp.hdi.data.processed_image,im_name,padding)
            #Add exported file names to class object -- connect input file name with the exported name
            connect_dict.update({f:im_name})

        #Add the connecting dictionary to the class object
        self.processed_images_export = connect_dict

        #return the dictionary of input names to output names
        #return connect_dict



#Define function for reading data with multiple input paths
def CreateDataset(list_of_paths,modality,masks=None,**kwargs):
    """Create an intramodality imaging dataset based on a given list of paths
    for imaging files

    returns an instance of class IntraModalityDataset
    """

    #Create a list to store the hdi_reader sets in
    data = []
    #Iterate through each path
    for i in range(len(list_of_paths)):
        #Ensure that it is a pathlib object
        p = Path(list_of_paths[i])
        #Check if masks is none
        if masks is None:
            #Read the data using hdi_reader
            p_dat = hdi_reader.HDIreader(path_to_data=p,mask = None, **kwargs)
        #Otherwise read each image with each mask in order
        else:
            #Ensure the mask is a pathlib object
            m = Path(masks[i])
            #Read the data using hdi_reader
            p_dat = hdi_reader.HDIreader(path_to_data=p,mask = m, **kwargs)
        #Append this p_dat to the data list
        data.append(p_dat)
    print('Concatenating Data...')
    #Concatenate the list of data to a single intramodality dataset
    data = IntraModalityDataset(data,modality)
    print('Done')
    #Return the IntraModalityDataset
    return data

#Class for dimension reduction of high-dimensional imaging data -- dependent on HDIimported data
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import external moduless
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import phate
import umap
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

import nibabel as nib
import imageio
from skimage.color import rgb2gray
import cv2





um = umap.UMAP(n_neighbors=5, random_state=42).fit(X_train)

a = um.transform(new)









#######################

#Create a class object to store attributes and functions in
class DimensionReduction(HDIimport):
    """Dimension reduction class for performing linear or manifold based dimension
    reduction on high-dimensional imaging data.
    """

    def __init__(self,HDIimport):
        """This function will initialize class to store data in.

        HDIimport: Class object from HDIimport module"""

        #Initialize class objects for dimension reduction objects and images
        self.hdi.dimension_reduction_method = None
        self.hdi.dimension_reduction = None
        self.hdi.hyperspectral_image = None


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

        #See which method is being used for dimension reduction
        if algorithm == "PCA":
            #**scale the data
            sc = StandardScaler()
            dat_pca = sc.fit_transform(tmp_frame)
            pca = PCA(**kwargs)
            pca.fit(dat_pca)
            embedding = pca.transform(dat_pca)
            im = self.FillHyperspectral(embedding)
            #Update the pca object
            if save_as is None:
                self.data.pca.update({"PCA":[pca,embedding,stop-start]})
                self.data.pca_image.update({"PCA":im})
            else:
                self.data.pca.update({str(save_as):[pca,embedding,stop-start]})
                self.data.pca_image.update({str(save_as):im})


        elif algorithm == "NMF":
            #**scale the data
            sc = MinMaxScaler()
            dat_nmf = sc.fit_transform(tmp_frame)
            nmf = NMF(**kwargs)
            nmf.fit(dat_nmf)
            embedding = nmf.transform(dat_nmf)
            im = self.FillHyperspectral(embedding)
            #Update the NMF object
            if save_as is None:
                self.data.NMF.update({"NMF":[nmf,embedding,stop-start]})
                self.data.NMF_image.update({"NMF":im})
            else:
                self.data.NMF.update({str(save_as):[nmf,embedding,stop-start]})
                self.data.NMF_image.update({str(save_as):im})

        elif algorithm == "UMAP":
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

        elif algorithm == "PHATE":
            #Set up phate parameters
            phat = phate.PHATE(**kwargs).fit(tmp_frame)
            embedding = phat.transform()
            im = self.FillHyperspectral(embedding)
            if save_as is None:
                self.data.PHATE.update({"PHATE":[phat,embedding,stop-start]})
                self.data.PHATE_image.update({"PHATE":im})
            else:
                self.data.PHATE.update({str(save_as):[phat,embedding,stop-start]})
                self.data.PHATE_image.update({str(save_as):im})

        elif algorithm == "tSNE":
            #Set up tSNE parameters
            tsne = TSNE(**kwargs).fit(tmp_frame)
            embedding = tsne.embedding_
            im = self.FillHyperspectral(embedding)
            #Update the tSNE object
            if save_as is None:
                self.data.tSNE.update({"tSNE":[tsne,embedding,stop-start]})
                self.data.tSNE_image.update({"tSNE":im})
            else:
                self.data.tSNE.update({str(save_as):[tsne,embedding,stop-start]})
                self.data.tSNE_image.update({str(save_as):im})

        elif algorithm == "Isomap":
            #Set up isomap parameters
            isomap = Isomap(**kwargs).fit(tmp_frame)
            embedding = isomap.embedding_
            im = self.FillHyperspectral(embedding)
            #Update the isomap object
            if save_as is None:
                self.data.isomap.update({"Isomap":[isomap,embedding,stop-start]})
                self.data.isomap_image.update({"Isomap":im})
            else:
                self.data.isomap.update({str(save_as):[isomap,embedding,stop-start]})
                self.data.isomap_image.update({str(save_as):im})


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



    def UMAPkRange(self,k_range,**kwargs):
        """Function for running umap over a range of k_values and storing the results
        in the imzML parsed object

        k_range: set this parameter using range(minimum,max,step)"""

        #Extract the metric used so we can add it to the name of the computation
        if 'metric' in kwargs:
            #If manually set, extract
            met_val = str(kwargs['metric'])
        else:
            #If not manually set, UMAP defaults to euclidean
            met_val = str("euclidean")

        #iterate through the krange
        for k in k_range:
            #Run the dimensionality reduction
            self.MSIdimRed(method="UMAP", save_as="UMAP_k"+str(k)+"_"+met_val, n_neighbors=k, **kwargs)
            print('Finished k='+str(k))


    def RunMultipleDimRed(self,UMAP_par=None,tSNE_par=None,PHATE_par=None,isomap_par=None,pca_par=None,kernelpca_par=None,\
        MDS_par=None,NMF_par=None,DenseAutoEncoder_par=None,UMAP_krange=False):
        """Function for running all dimension reduction techniques.

        data_imzML: Class imzMLreader object
        [type]_par: Dictionary containing keyword arguments to pass to each dimensionality
        reduction technique

        Note:If you are using a k range for UMAP, you must specify UMAP_krange=True then
        pass keyword argumets to UMAP_par for the UMAPkRange function. If you choose UMAP_krange=False
        then ignore this and simply pass UMAP keyword arguments.

        Ex: UMAP_par = {'metric':'cosine',"n_neighbors": "5"}
        """
        #Check to see if running umap
        if UMAP_par is not None:
            #If using a krange for UMAP, run the function UMAP_krange
            if UMAP_krange:
                #Run the krange function
                self.UMAPkRange(**UMAP_par)
            else:
                #If not using a krange, run the single umap function
                self.MSIdimRed(algorithm="UMAP",**UMAP_par)

        #Check to see if running tSNE
        if tSNE_par is not None:
            #Run the tSNE algorithm
            self.MSIdimRed(algorithm="tSNE",**tSNE_par)
        #Check to see if running PHATE
        if PHATE_par is not None:
            #Run the tSNE algorithm
            self.MSIdimRed(algorithm="PHATE",**PHATE_par)
        #Check to see if running isomap
        if isomap_par is not None:
            #Run the isomap algorithm
            self.MSIdimRed(algorithm="Isomap",**isomap_par)
        #Check to see if running PCA
        if pca_par is not None:
            #Run PCA
            self.MSIdimRed(algorithm="PCA",**pca_par)
        #Check to see if running NMF
        if NMF_par is not None:
            #Run the NMF
            self.MSIdimRed(algorithm="NMF",**NMF_par)
        #Let the user know that all methods are finished
        print('Finished All Methods')


    def MSIdimRedtoImage(self, which_im, export_tiff=False,export_path=None,padx=50,pady=50):
        """This function will convert the im object returned from MSIdimRed to a
        viewable image with or without padding (recommended to pad for image
        registration)

        im: Array returned by MSIdimRed
        export_tiff: Boolean. Default is to export the tiff image of the umap image
        export_path: must contain the '.tif' file extension

        Returned will be the modified im numpy array"""

        im = np.copy(which_im)
        im = np.pad(im, [(padx,padx),(pady,pady),(0,0)], mode = 'constant')
        self.data.Image_padx, self.data.Image_pady = padx, pady

        #Write float32 tiff image using tifffile module
        if export_tiff:
            imwrite(export_path,im)
        #Return the new im obect
        return im

    def ExportNifti(self,image,filename=None,resize=False,new_size=None,rot=1,grayscale=False):
        """This function will export your final images to nifti format for image
        registration with elastix

        Your filename endings must be .nii
        new_size must be a tuple of integers"""
        #Get a copy of the image so we dont change it
        tmp = image.copy()
        #Add in your rotation so we can store it for later registration
        self.data.Image_rot = rot
        #Optional to change the image size
        if resize:
            tmp = cv2.resize(tmp,new_size)
        self.data.Image_resize_shape = new_size
        #Create the nifti objects
        print('Exporting nifti stack image...')
        nifti_col = nib.Nifti1Image(np.rot90(tmp,rot), affine=np.eye(4))
        nib.save(nifti_col, filename)
        print('Finished exporting nifti stack image')
        #Convert to grayscale if you choose
        if grayscale:
            fin_gray = rgb2gray(tmp)
            filename_gray = filename+'_gray.nii'
            #Export the grayscale image
            print('Exporting nifti gray image...')
            nifti_org = nib.Nifti1Image(np.rot90(fin_gray,rot), affine=np.eye(4))
            nib.save(nifti_org, filename_gray)
            print('Finished exporting nifti gray image')


def Plot3D_hyperspectral(embedding, export_path):
    """This function will export a 3D umap embedding
    color: takes values "leiden","infomap",or "kmeans". Default it to color
    the image with RGB
    export_path: path to export image to. Must include file extension"""
    #Create Graph object
    mpl_fig = plt.figure()
    ax = Axes3D(mpl_fig)
    #Get coloring scheme
    rgb = minmax_scale(embedding)
    ax.scatter(embedding[:,0],embedding[:,1],embedding[:,2],s=20,facecolor = rgb)
    #Now save the image with your export path
    mpl_fig.savefig(export_path, bbox_inches = 'tight',\
        pad_inches = 1)





#

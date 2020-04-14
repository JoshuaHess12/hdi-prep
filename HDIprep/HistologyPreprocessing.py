#Histology Preprocessing Python Module
#Joshua Hess

from skimage import util
from skimage.util import montage
import skimage.io
import tifffile as tiff
import os
import math
import numpy
import cv2
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.morphology as skmorph
from skimage.morphology import disk
import PIL
from skimage import data as skdata
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from scipy import ndimage
import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from pathlib import Path


#Class for histological image preprocessing
class HistologyImage(object):

    def __init__(self,object,mask_path=None):
        PIL.Image.MAX_IMAGE_PIXELS = None
        """Initial instance for the Histology Image class object"""
        #Read the image with matplotlib
        self.original = skimage.io.imread(object, plugin='tifffile')
        self.original_filename = str(object)
        if mask_path is not None:
            if not os.path.exists(mask_path):
                print('Not a valid mask path. Try again')
            else:
                self.manual_mask = plt.imread(mask_path)
                self.manual_mask_filename = str(mask_path)

    def HistologyFiltering(self,filter_size):
        """Function for filtering the image"""
        #Filter the image to remove salt and pepper noise
        print('Filtering Image...')
        filt_img = cv2.medianBlur(self.original,filter_size)
        #Show the filtering process
        plt.imshow(filt_img)
        plt.title("Image Filtering")
        plt.show()
        #Save the filtered object internally
        self.filtered = filt_img

    def ManualMasking(self,image):
        """This function will mask the histology image using otsu thresholding
        and morphological openings and closings"""
        #Use the hand-drawn mask for image preprocessing step 1
        tmp_im = image.copy()
        res = cv2.bitwise_and(tmp_im,tmp_im,mask = self.manual_mask)
        plt.imshow(res)
        plt.title("Extract Manually Masked Region")
        plt.show()
        #Save the masked object internally
        self.manually_masked_extract = res

    def ThresholdMasking(self,image,*thresh_value,type='otsu'):
        """This function will convert your image to a grayscale version and will
        perform otsu thresholding on your image to form a mask"""
        #Convert to grayscale image
        gray_img = rgb2gray(image)
        print (gray_img.shape,gray_img.dtype,gray_img.size)
        plt.imshow(gray_img, cmap = plt.cm.gray)
        plt.title("Convert to Grayscale")
        plt.show()
        if type == 'otsu':
            #Otsu thresholding
            thresh = threshold_otsu(gray_img)
            otsu_img = gray_img < thresh
            plt.imshow(otsu_img)
            plt.title("Otsu Thresholding - Mask")
            plt.show()
            self.thresholded_image = otsu_img
        elif type == 'manual':
            #Print the minimum and maximum values for the user to judge
            print("Min Value: ",str(gray_img.min()),",Max Value: ",str(gray_img.max()))
            thresh_img = gray_img < thresh_value
            plt.imshow(thresh_img)
            plt.title("Manual Thresholding - Mask")
            plt.show()
            self.thresholded_image = thresh_img
        else:
            print('Invalid threshold method')

    def MorphOpen(self,image,disk_size):
        """This function will perform morphological opening on your histology
        image mask by using disk shape. This function will not save the new
        image in the class object! Saves memory"""
        #Perform morphological operations on the thresholded mask
        print('Performing Morhological Opening...')
        open_img = cv2.morphologyEx(image.astype('uint8'), cv2.MORPH_OPEN, skmorph.disk(disk_size))
        plt.imshow(open_img)
        plt.title("Morphological Opening - Mask")
        plt.show()
        return open_img

    def MorphClose(self,image,disk_size):
        """This function will perform morphological closing on your histology
        image mask by using disk shape. This function will not save the new
        image in the class object! Saves memory"""
        #Perform morphological operations on the thresholded mask
        print('Performing Morphological Closing...')
        closed_img = cv2.morphologyEx(image.astype('uint8'), cv2.MORPH_CLOSE, skmorph.disk(disk_size))
        plt.imshow(closed_img)
        plt.title("Morphological Closing - Mask")
        plt.show()
        return closed_img

    def MorphFill(self,image):
        """This function will perform morphological filling on your histology image
        mask"""
        #Filling in the mask
        print('Performing Morphological Fill...')
        filled_mask = ndimage.binary_fill_holes(image).astype('uint8')
        plt.imshow(filled_mask)
        plt.title("Filled Mask")
        plt.show()
        return filled_mask

    def SliceImages(self,mask,original):
        """This function will take all the images that you input and will slice
        them all to be a certain size"""
        #Get nonzero indices from your mask so we can apply to original image
        print('Getting nonzero indices from mask...')
        nonzero = np.nonzero(mask)
        minx = min(nonzero[0])
        maxx = max(nonzero[0])
        miny = min(nonzero[1])
        maxy = max(nonzero[1])
        #Extract sliced, nonzero regions from your original image
        self.sliced_original = original[minx:maxx,miny:maxy]
        #Plot your image
        plt.imshow(self.sliced_original)
        plt.title('Sliced, Original Image')
        plt.show()
        #Extract sliced, nonzero regions from your mask image
        self.sliced_mask = mask[minx:maxx,miny:maxy]
        #Plot your image
        plt.imshow(self.sliced_mask)
        plt.title('Sliced Mask')
        plt.show()

    def PadImages(self,mask,original,padx=50,pady=50):
        """This function will pad the images so that registration performs well"""
        #Pad the masked image and the original image
        print('Adding Pad to Images...')
        self.padded_mask = np.pad(mask,[(padx,padx),(pady,pady)],mode='constant')
        self.padded_original = np.pad(original,[(padx,padx),(pady,pady),(0,0)],mode='constant')
        #Plot the padded mask
        plt.imshow(self.padded_mask)
        plt.title('Padded Mask')
        plt.show()
        #Plot the padded original image
        plt.imshow(self.padded_original)
        plt.title('Padded, Original Image')
        plt.show()

    def ExportNifti(self,filename_org,final_mask,final_original,rot=1,grayscale=False,filename_gray=None):
        """This function will export your final images to nifti format for image
        registration with elastix

        Your filename endings must be .nii"""
        #Get the current directory
        tmp_path = Path('..')
        home_dir=tmp_path.cwd()
        #Get the directory of your input images so that we export to the same
        parent = Path(self.original_filename).parent
        os.chdir(parent)
        #Extract the original image using the mask
        tmp = final_original.copy()
        fin_org = cv2.bitwise_and(tmp,tmp,mask = final_mask)
        #Plot the final histology masked image
        plt.imshow(fin_org)
        plt.title('Final Histology Image')
        plt.show()
        #Export the histology final image
        print('Exporting nifti color image...')
        nifti_org = nib.Nifti1Image(np.rot90(fin_org,rot), affine=np.eye(4))
        nib.save(nifti_org, filename_org)
        print('Finished exporting nifti color image')
        #Convert to grayscale if you choose
        if grayscale:
            fin_gray = rgb2gray(fin_org)
            plt.imshow(fin_gray)
            plt.title('Final Histology Image - Gray')
            plt.show()
            #Export the grayscale image
            print('Exporting nifti gray image...')
            nifti_gray = nib.Nifti1Image(np.rot90(fin_gray,rot), affine=np.eye(4))
            nib.save(nifti_gray, filename_gray)
            print('Finished exporting nifti gray image')
        #Change back to the home Directory
        os.chdir(home_dir)

    def MorphProcess_Parallel(self,image,process,disk_size,max_cores=True,num_cores=None):
        """This function will parallelize the morphological operations. If you
        choose to pass the number of cores to the function, your image size must be
        divisible by that number"""

        def MorphClose_General(image):
            """This function will perform morphological closing on your histology
            image mask by using disk shape. This function will not save the new
            image in the class object! Saves memory"""
            #Perform morphological operations on the thresholded mask
            print('Performing Morphological Closing...')
            closed_img = cv2.morphologyEx(image.astype('uint8'), cv2.MORPH_CLOSE, skmorph.disk(disk_size))
            return closed_img

        def MorphOpen_General(image):
            """This function will perform morphological opening on your histology
            image mask by using disk shape. This function will not save the new
            image in the class object! Saves memory"""
            #Perform morphological operations on the thresholded mask
            print('Performing Morhological Opening...')
            open_img = cv2.morphologyEx(image.astype('uint8'), cv2.MORPH_OPEN, skmorph.disk(disk_size))
            return open_img

        #Get the input image shape to use for blocking
        input_im_shape = image.shape
        #Convert to unsigned 8 bit so we can use opencv for morphology
        tmp_im=image.copy().astype('uint8')
        #Check for how many cores you are going to use
        if max_cores:
            #Get the full amount of cores
            tmp_max=os.cpu_count()
            tmp_divisors = []
            for i in range(1,tmp_max+1):
                if input_im_shape[0]%i==input_im_shape[1]%i==0:
                    tmp_divisors.append(i)
            #Get the greatest common divisor for possible number of cores to use
            num_cores = max(tmp_divisors)
        elif num_cores is not None:
            num_cores = num_cores
        #Start the calculations
        print('Using '+str(num_cores)+' cores for calculation...')
        #Use the gcd to use to divide your image into chunks
        print('Cutting the image into blocks...')
        chunk=util.view_as_blocks(tmp_im,\
            (int(input_im_shape[0]/num_cores),int(input_im_shape[1]/num_cores)))
        #Use joblib to do the calculations
        if process == 'Closing':
            #Perform Morphological closing
            print('Performing parallel closing...')
            processed=Parallel(n_jobs=num_cores)(delayed(MorphClose_General)(chunk[i,j])for i in range(num_cores) for j in range(num_cores))
        elif process == 'Opening':
            print('Performing parallel opening...')
            processed=Parallel(n_jobs=num_cores)(delayed(MorphOpen_General)(chunk[i,j])for i in range(num_cores) for j in range(num_cores))
        #Stitch together the full image from the processed chunks
        full=montage(processed)
        #Show the full processed image
        plt.imshow(full)
        plt.title('Parallel Processed Image')
        plt.show()
        #Return the image
        return full

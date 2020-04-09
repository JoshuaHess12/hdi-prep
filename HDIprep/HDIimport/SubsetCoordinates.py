#Function for subsetting a list of coordinate tuples
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import external modules
import numpy as np
import pandas as pd
import random
import scipy.sparse


#Define function
def SubsetCoordinates(coords,n,array_size):
        """Subset coordinates and return a list of tuples indicating
        the position of subsetting and where to find those tuples in the list. Also
        return a scipy coo matrix for boolean mask to use with subsetting.
        Note: Coordinates are 1-indexed and not python 0-indexed

        coords: list of 3D-tuples indicating the position of pixels
        n: integer indicating the size of subsampling
        """

        #Set random seed
        random.seed(1234)
        #Take subsample of integers for indexing
        idx = list(np.random.choice(a=len(coords), size=n,replace=False))
        #Use the indices to subsample coordinates
        sub_coords = [coords[c] for c in idx]
        #Create data with True values same length as sub_coords for scipy coo matrix
        data = np.ones(len(sub_coords),dtype=np.bool)
        #Create row data for scipy coo matrix (-1 index for 0-based python)
        row = np.array([sub_coords[c][1]-1 for c in range(len(sub_coords))])
        #Create row data for scipy coo matrix (-1 index for 0-based python)
        col = np.array([sub_coords[c][0]-1 for c in range(len(sub_coords))])
        #Create a subset mask
        sub_mask = scipy.sparse.coo_matrix((data, (row, col)), shape=array_size)

        #Return the objects
        return sub_mask, sub_coords

# HDIprep
Module for high-dimensional image reading and preparation.

**File type(s) supported**:
- ome.tif(f)
- tif(f)
- h(df)5
- nifti1

**Modules included**:
1) Image reading with file type(s) listed above
2) Image masking
3) Image subsampling
4) Dimension reduction using UMAP -- optimal dimension embedding selection
5) Image filtering -- median filtering to remove salt and pepper noise
6) Binary operations (opening, closing, etc.)
7) Image exporting to nifti1 format for registration with Elastix

## Implementation details
Function for preparing imaging data is HDIprep.


### Classes and function structure


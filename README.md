# HDIprep
High-dimensional image reading, compression, and preprocessing.

**File type(s) supported**:
- ome.tif(f)
- tif(f)
- h(df)5
- nifti1
- imzML

**Processing steps**:
1) Image reading with file type(s) listed above
2) Image masking
3) Image subsampling (applicable to dimension reduction)
4) Dimension reduction using UMAP -- steady-state embeddings
5) Image filtering -- median filtering to remove salt and pepper noise
6) Binary operations (opening, closing, etc.)
7) Image exporting to nifti1 format for registration with Elastix

## Implementation details
All image processing can be run using YAML files in conjunction with the function 'RunHDIprepYAML' in yaml_hdi_prep.py. YAML file inputs can be run from the command line, or the function can be called in Python.

### YAML input:
Options/ordered steps for image processing can all be set in a YAML file. Two input options need to be included in the YAML file:
1) ImportOptions and
2) ProcessingSteps

These two steps are main headers in the YAML format as follows:
```bash
#-----arguments for importing-----
ImportOptions:
  # input path to images
  list_of_paths:
    - "path/to/image"
  # flatten the array? (True for dimension reduction, False for histology images)
  flatten: True
  # subsample the image?
  subsample: True
  # method of subsampling
  method: grid
  #number of samples for subsampling if not grid subsampling
  #n: 0.15
  #grid spacing for subsampling
  grid_spacing: (5,5)
  #Use a mask? If yes, put path
  masks:
  #Save memory for very lare images? (Yes for whole tissue images)
  save_mem: True

#-----Arguments for processing-----
# set processing steps (in order) and arguments for each
ProcessingSteps:
  - RunUMAP:
      n_neighbors: 15
      metric: 'euclidean'
      random_state: 1221
      verbose: 0

  - SpatiallyMapUMAP

  - ExportNifti1:
      output_dir: "/Users/joshuahess/Desktop/Test"
      padding:
```
*Note: lists are indicated in YAML files by the '-' character*

#### YAML ImportOptions:
These options indicate import options inherited from class IntraModalityDataset and thus HDIimport:
Required

#### YAML ProcessingSteps:


### Command line usage -- recommended:
All image processing and exporting can be run from the command line by calling python, the command_hdi_prep.py code, and entering the path to a .yaml file that contains processing steps:
```bash
python command_hdi_prep.py --path_to_yaml /path/to/example.yaml
```



## Classes and function structure

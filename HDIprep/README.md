# HDIprep
Module for high-dimensional image reading and preparation.

**File type(s) supported**:
- ome.tif(f)
- tif(f)
- h(df)5
- nifti1

**Processing steps included**:
1) Image reading with file type(s) listed above
2) Image masking
3) Image subsampling (applicable to dimension reduction)
4) Dimension reduction using UMAP -- optimal dimension embedding selection
5) Image filtering -- median filtering to remove salt and pepper noise
6) Binary operations (opening, closing, etc.)
7) Image exporting to nifti1 format for registration with Elastix

## Implementation details
All image processing can be run using YAML files in conjunction with the function 'RunHDIprepYAML' in yaml_hdi_prep.py. YAML file inputs can be run from the command line, or the function can be called in python.

### YAML input:
Options/ordered steps for image processing can all be contained in YAML file format. Two main input options need to be included in the YAML file:
1) ImportOptions and
2) ProcessingSteps

These two steps are main headers in the YAML format as follows:
```bash
#-----arguments for importing-----
ImportOptions:
  list_of_paths:
    - "path/to/image"
    - "path/to/image(s)"
  etc.

#-----Arguments for processing-----
ProcessingSteps:
  - RunOptimalUMAP:
  - etc,
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


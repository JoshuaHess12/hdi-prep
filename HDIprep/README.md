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
All image processing can be run using the function 'RunHDIprepYAML' in yaml_hdi_prep.py. This can be run from the command line, or the function can be called in python.

**Command line usage -- recommended**:
All image processing and exporting can be run from the command line by calling python, the command_hdi_prep.py code, and entering the path to a .yaml file that contains processing steps:
```bash
python command_hdi_prep.py --path_to_yaml /path/to/example.yaml
```
**YAML input options**:


### Classes and function structure


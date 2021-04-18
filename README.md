# HDIprep
High-dimensional image reading, compression, and preprocessing workflow as part of the MIAAIM framework.

## Implementation Details
All steps  can be run using YAML files in conjunction with the function 'RunHDIprepYAML' from yaml_hdi_prep.py. YAML inputs can be run from command line, or can be called in Python. HDIprep is available as a containerized workflow for easy installation via Docker <insert link>.

### File Type(s) Supported
- OME-TIF
- TIF
- HDF5
- Nifti-1
- imzML

### Command Line Usage with Docker (recommended):
1. [Install]() nextflow and Docker. Check with `nextflow run hello` and `docker images` to make sure both are functional.
All image processing and exporting can be run from the command line by calling python, the command_hdi_prep.py code, and entering the path to a .yaml file that contains processing steps:
```bash
python command_hdi_prep.py --path_to_yaml /path/to/example.yaml
```
### YAML File Input:
Options/ordered steps for image processing are passed in a YAML file. Two input headers must be included in the YAML file:
1) ImportOptions and
2) ProcessingSteps

These two steps are set as main headers in YAML format as follows:
```bash
#-----arguments for importing images-----
ImportOptions:
  # input path to images
  list_of_paths:
    - "path/to/input-image.ext"
  # flatten the array? (True for dimension reduction, False for histology images)
  flatten: True
  # subsample the image?
  subsample: True
  # method of subsampling
  method: grid
  # grid spacing for subsampling
  grid_spacing: (5,5)
  # use a mask for only processing a subset of image? If yes, put path
  masks:
  # save memory for very large images? (e.g. for whole tissue images)
  save_mem: True

#-----arguments for processing images-----
# set processing steps (in order) and arguments for each
ProcessingSteps:
  # steady state UMAP compression
  - RunOptimalUMAP:
      n_neighbors: 15
      metric: 'euclidean'
      random_state: 1221
      verbose: 0
  # spatial reconstruction of UMAP embedding
  - SpatiallyMapUMAP
  # export nifti file for registering with Elastix
  - ExportNifti1:
      output_dir: "path/to/output-directory"
```
*Note: lists are indicated in YAML files by the '-' character. HDIprep will run the steps listed sequentially*

#### Input Parameters:
| YAML Step | Options |
| --- | --- |
| 1. ImportOptions |
| `--list_of_paths` | paths to input images (Ex. `./example.ome.tiff`) |
| `--flatten` | flatten pixels to array
                `True` if compressing images
                `False` if histology processing|
| `--subsample` | subsample image for compression (set `True` if compressing images) |
| `--method` | subsampling method (options: `grid`, `random`, or `pseudo_random`) |
| `--grid_spacing` | tuple representing x and y spacing for grid sampling (Ex. `(5,5)` |
| `--masks` | paths to masks to import in TIF format if compression portion of image |
| `--save_mem` | option to reduce memory footprint (for large images set `True`) |
| 2. ProcessingSteps |
| `--RunOptimalUMAP` | run steady-state image compression |
| `--RunUMAP` | run UMAP compression |
#### ProcessingSteps Parameters:

## Contributing to HDIprep

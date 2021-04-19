# hdi-prep
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
      # neighbors to use for UMAP
      n_neighbors: 15
      # spectral landmarks
      landmarks: 3000
      # metric to use for UMAP
      metric: 'euclidean'
      # reproducible results
      random_state: 1221
      # dimension range for steady state compression
      dim_range: (1,10)
      # export diagnostics for steady state compression
      export_diagnostics: True
      # output directory
      output_dir: "path/to/output-directory"
  # spatial reconstruction of UMAP embedding
  - SpatiallyMapUMAP
  # export nifti file for registering with Elastix
  - ExportNifti1:
      output_dir: "path/to/output-directory"
```
*Note: lists are indicated in YAML files by the '-' character. HDIprep will run the steps listed sequentially*

#### Input Parameters:
Options for import data and processing are listed below. Detailed descriptions of each function can be found within source code.
| YAML Step | Options |
| --- | --- |
| 1. ImportOptions |
| `--list_of_paths` | paths to input images (Ex. `./example.ome.tiff`) |
| `--flatten` | flatten pixels to array <br> <br> Options: <br>`True` if compressing images <br> <br> `False` if histology processing |
| `--subsample` | subsample image for compression  <br> <br> Options: <br> `True` if compressing images <br> <br> `False` if histology processing |
| `--method` | subsampling method <br> <br> Options: <br> `grid` for uniform grid sampling <br> <br> `random` for random coordinate sampling <br> <br> `pseudo_random` for random sampling initalized by uniform grids |
| `--grid_spacing` | tuple representing x and y spacing for grid sampling (Ex. `(5,5)`) |
| `--n` | fraction indicating sampling number (between 0-1) for random or pseudo_random sampling (Ex. `(5,5)`) |
| `--masks` | paths to TIF masks if compressing only a portion of image (Ex. `./mymask.tif`)|
| `--save_mem` | option to reduce memory footprint <br> <br> Options: <br> `True` if compressing very large images <br> <br> `False` if running for interactive Python session |
| 2. ProcessingSteps |
| `RunOptimalUMAP` | run steady-state image compression with UMAP <br> <br> Options: <br> `n_neighbors` nearest neighbors (Ex. `n_neighbors: 15`) <br> <br> `landmarks` number of spectral centroids (Ex. `landmarks: 3000`) <br> <br> `metric` metric to use for UMAP (Ex. `metric: "euclidean"`) <br> <br> `random_state` for reproducible results (Ex. `random_state: 0`) <br> <br> `dim_range` tuple indicating range of dimensionalities for iterative embedding (Ex. `dim_range: "(1,10)"`) <br> <br> `export_diagnostics` exports csv and image of steady state compression results (Ex. `export_diagnostics: True`) <br> <br> `output_dir`  directory for export diagnostic expoting (Ex. `output_dir: "./outdirectory")` <br> <br> `**kwargs` keyword arguments to be passed to [UMAP](https://umap-learn.readthedocs.io/en/latest/basic_usage.html)|
| `RunUMAP` | run UMAP compression <br> <br> Options: <br> `**kwargs` to be passed to UMAP (link above) |
| `RunOptimalParametricUMAP` | run steady-state image compression with neural network UMAP <br> <br> Options: <br> same input options as `RunOptimalUMAP` |
| `SpatiallyMapUMAP` | reconstruct image from pixel positions and UMAP embedding coordinates <br> <br> Options: <br> `method` reconstruction method to use (`rectangular` for large images with no mask, 'coordinate' for masked images and all data in imzML format (Ex. `method: "rectangular"`)) <br> <br> `save_mem` for large images (Ex. `save_mem: True`)) |
| `ApplyManualMask` | apply manual mask for histology processing <br> <br> Options: <br> mask is accessed from ImportOptions |
| `MedianFilter` | remove salt and pepper noise for histology processing <br> <br> Options: <br> `filter_size` size for disk used in filtering (Ex. `filter_size: 15`) <br> <br> `parallel` parallel processing option (Ex. `parallel: True`) |
| `Threshold` | create mask for processing with thresholding <br> <br> Options: <br> `type` thresholding type -- can be `"manual"` or automatic thresholding with `"otsu"` (Ex. `type: "otsu"`) <br> <br> `thresh_value` to use if manual thresholding (Ex. `thresh_value: 1.0`) <br> <br> `correction` factor to multiply threshold with for more stringent thresholding (Ex. `correction: 1.2`) |
| `Open` | morphological opening on mask <br> <br> Options: <br> `disk_size` disk size for opening (Ex. `disk_size: 10`) <br> <br> `parallel` parallel processing option (Ex. `parallel: True`) |
| `Close` | morphological closing on mask <br> <br> Options: <br> `disk_size` disk size for closing (Ex. `disk_size: 10`) <br> <br> `parallel` parallel processing option (Ex. `parallel: True`) |
| `Fill` | morphological filling on mask (fill holes) |
| `ApplyMask` | apply processed mask to image for final processing step |
| `ExportNifti1` | export processed image or compressed image in the Nifti-1 format for image registration with HDIreg workflow in MIAAIM <br> <br> Options: <br> `output_dir` output directory (Ex. `output_dir: "./outdirectory"`) <br> <br> `padding` border padding to add to image (useful for registration) (Ex. `padding: (50,50)`) |

## Contributing to hdi-prep
If you are interested in contributing to hdi-prep, access the contents to see the software organization. Code structure is documented for each module.

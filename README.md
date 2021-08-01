# hdi-prep
High-dimensional image reading, compression, and preprocessing workflow as part of the MIAAIM framework.

## Implementation Details
All steps can be run using YAML files in conjunction with the function 'RunHDIprepYAML' from yaml_hdi_prep.py. YAML inputs can be run from command line, or can be called in Python. The easiest way to use `hdi-prep` in the Python environment is through the pip installable [miaaim-python package](https://github.com/JoshuaHess12/miaaim-python). An API reference for Python usage is available at [miaaim.org](https://miaaim.org).

### File Type(s) Supported
- OME-TIF
- TIF
- HDF5
- NIfTI-1
- imzML

### Command Line Usage with Docker:
All image processing can be run with dependecies installed via Docker, the command_hdi_prep.py code, and entering the path to a .yaml file that contains processing steps as follows:
1. Install [Docker](https://www.docker.com) on your machine.
2. Check that Docker is installed with `docker images`
3. Pull the hdi-prep docker container `docker pull joshuahess/hdi-prep:latest` where latest is the version number.
4. Mount your data in the Docker container and enter shell with `docker run -it -v /path/to/data:/data joshuahess/hdi-prep:latest bash`
5. Run the pipeline with your new data using the following command:
```bash
python app/command_hdi_prep.py --path_to_yaml /data/yourfile.yaml --out_dir /data
```
### Usage without Docker:
If you are unable to install Docker on your machine, you can clone the `hdi-prep` repo and use it from the command line:
1. Clone the `hdi-prep` repo and install requirements in the `HDIprep.yml` file.
```bash
git clone https://github.com/JoshuaHess12/hdi-prep.git
cd hdi-prep
python command_hdi_prep.py --path_to_yaml path/to/your.yaml --out_dir path/to/output-directory
```

### YAML File Input:
Steps for image processing are passed sequentially in a YAML file. Two input headers must be included in the file:
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
      dim_range: (1,11)
      # export diagnostics for steady state compression
      export_diagnostics: True
      # output directory
      output_dir: "path/to/output-directory"
  # spatial reconstruction of UMAP embedding
  - SpatiallyMapUMAP
  # export NIfTI file for registering with Elastix
  - ExportNifti1:
      output_dir: "path/to/output-directory"
```
*Note: lists are indicated in YAML files by the '-' character. HDIprep will run the steps listed sequentially*

#### Input Parameters Command Line Usage:
Options for importing data and processing are listed below. Detailed descriptions of each function can be found within source code.
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
| `ExportNifti1` | export processed image or compressed image in the NIfTI-1 format for image registration with HDIreg workflow in MIAAIM <br> <br> Options: <br> `output_dir` output directory (Ex. `output_dir: "./outdirectory"`) <br> <br> `padding` border padding to add to image (useful for registration) (Ex. `padding: (50,50)`) <br> <br> `target_size` resulting resize shape after padding to match with corresponding image (Ex. `target_size: (1000,1500)`) |

## Contributing to hdi-prep
If you are interested in contributing to hdi-prep, access the contents to see the software organization. Code structure is documented for each module.

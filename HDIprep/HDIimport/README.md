# HDIimport
Module for high-dimensional image importing to python.

**File type(s) supported**:
- ome.tif(f)
- tif(f)
- h(df)5
- nifti

## Implementation details
Class object for importing data is HDIreader inside of hdi_reader.py.

***HDIreader arguments***:
* *path_to_data*: path to imaging data (Ex: path/mydata.extension)
* *path_to_markers*: path to marker list (Ex: path/mymarkers.csv or None)
* *flatten*: True to return a flattened pixel data table for dimension reduction
* *mask*: Path to tif mask to use for selecting a region to focus on in downstream preparation
* * ***kwargs*: - inherited from SubsetCoordinates utils function
  * *method*: "random" for uniform random coordinate sampling, "grid" for uniform grid spacing sampling
  * *n*: number of samples if method is "random" (Ex: 1000 for a count, 0.1 for percentage based sampling)
  * *grid_spacing*: tuple indicating xy grid size if method is "grid" (Ex: (2,2) for sampling every other pixel in an image)

*Note: If mask is used in addition to subsampling, subsamples are taken from within the masked region!*

### Classes and function structure
All class objects contain uppercase lettering, and exist inside of all lowercase .py files.

```bash
hdi_reader.py
├── HDIreader (class)
│   ├── imzml_reader.py
│   │   └── imzMLreader(class)
│   ├── nifti1_reader.py
│   │   └── NIFTI1reader (class)
│   └── cyt_reader.py (class)
│       ├── CYTreader(class)
│       ├── tif_reader.py
│       │   └── TIFreader(class)
│       └── h5_reader.py
│           └── H5reader(class)
utils.py (general functions)
```

*To add in custom file formats, integrate your image reader class with HDIreader class by following the structure of TIFreader/H5reader or imzMLreader. For best usage, incorporate coordinate subsampling as well by using SubsetCoordinates function that applies to both the imzMLreader and the CYTreader classes.*

**HDIreader (class) components**
```bash
HDIreader (class)
└── .hdi: base component that stores imzMLreader, CYTreader, or NIFTI1reader classes
```
 
**CYTreader (class) components**
```bash
CYTreader (class)
└── .data: base component that merges either TIFreader or H5reader classes with CYTreader attributes
    ├── .array_size: array size of the image (2D value)
    ├── .image_shape: image shape (2D or 3D value with channels included)
    ├── .channels: list of data channel names (If included. If not, a range of numbers equal to number of channels)
    ├── .filename: pathlib object storing the file name of the data
    ├── .image: numpy array storing the n-dimensional image (inherited from TIFreader/H5reader)
    ├── .pixel_table: pandas dataframe containing pixel-level data (rows are individual pixels, columns are channels)
    ├── .coordinates: list of 1-indexed 3D tuples (z=1) representing pixel locations
    └── .sub_coordinates: list of subsampled coordinates used to create pixel_table is subsampling is chosen
```

**imzMLreader (class) components**
```bash
imzMLreader (class)
└── .data: base component that merges pyimzML.ImzMLParser imzMLreader attributes
    ├── .array_size: array size of the image (2D value)
    ├── .image_shape: image shape (2D or 3D value with channels included)
    ├── .channels: list of data channel names (If included. If not, a range of numbers equal to number of channels)
    ├── .filename: pathlib object storing the file name of the data
    ├── .image: None -- currently not supported to create a full array from the imzML data (not currently needed)
    ├── .pixel_table: pandas dataframe containing pixel-level data (rows are individual pixels, columns are channels)
    ├── .coordinates: list of 1-indexed 3D tuples (z=1) representing pixel locations
    └── .sub_coordinates: list of subsampled coordinates used to create pixel_table is subsampling is chosen
```

**NIFTI1reader (class) components**
```bash
NIFTI1reader (class)
└── .data: base component that merges NiBabel and NIFTI1reader attributes
    ├── .array_size: array size of the image (2D value)
    ├── .image_shape: image shape (2D or 3D value with channels included)
    ├── .channels: list of data channel names (If included. If not, a range of numbers equal to number of channels)
    ├── .filename: pathlib object storing the file name of the data
    ├── .image: None -- currently not supported to create a full array from the imzML data (not currently needed)
    ├── .pixel_table: pandas dataframe containing pixel-level data (rows are individual pixels, columns are channels)
    ├── .coordinates: list of 1-indexed 3D tuples (z=1) representing pixel locations
    └── .sub_coordinates: list of subsampled coordinates used to create pixel_table is subsampling is chosen
```

**utils**
```bash
utils
├── ReadMarkers: csv marker reading
└── SubsetCoordinates: subsample imaging data with or without a mask using uniform grid or uniform random sampling
    └── FlattenZstack: flatten an xyc image to create a pandas data frame with per pixel information
```

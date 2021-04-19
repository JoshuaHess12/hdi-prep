# HDIimport
Module for high-dimensional and histology image importing.

## Details
Class object for importing data is HDIreader inside of hdi_reader.py.

| HDIreader | Options |
| --- | --- |
| `--path_to_data` | path to imaging data (Ex. `./example.ome.tiff`) |
| `--path_to_markers` | path to imaging data or `None` (Ex. `./examplemarkers.csv`) |
| `--flatten` | flatten pixels to array <br> <br> Options: <br>`True` if compressing images <br> <br> `False` if histology processing |
| `--mask` | path to TIF masks if compressing only a portion of image (Ex. `./mymask.tif`)|
| `--subsample` | subsample image for compression  <br> <br> Options: <br> `True` if compressing images <br> <br> `False` if histology processing |
| `--**kwargs` | inherited from `SubsetCoordinates` in utils.py |
| SubsetCoordinates | Options |
| `--method` | subsampling method <br> <br> Options: <br> `grid` for uniform grid sampling <br> <br> `random` for random coordinate sampling <br> <br> `pseudo_random` for random sampling initalized by uniform grids |
| `--grid_spacing` | tuple representing x and y spacing for grid sampling (Ex. `(5,5)`) |
| `--n` | fraction indicating sampling number (between 0-1) for random or pseudo_random sampling (Ex. `(5,5)`) |
*Note: If mask is used in addition to subsampling, subsamples are taken from within the masked region!*

## Classes and function structure
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
    ├── .processed_image: intialized as none. Gets modified in intramodality_dataset class through processing
    ├── .mask: scipy sparse coordinate matrix containing input mask (or none)
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
    ├── .processed_image: intialized as none.
    ├── .mask: scipy sparse coordinate matrix containing input mask (or none)
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
    ├── .image: numpy array storing the n-dimensional image
    ├── .processed_image: intialized as none. Gets modified in intramodality_dataset class through processing
    ├── .mask: scipy sparse coordinate matrix containing input mask (or none)
    ├── .pixel_table: pandas dataframe containing pixel-level data (rows are individual pixels, columns are channels)
    ├── .coordinates: list of 1-indexed 3D tuples (z=1) representing pixel locations
    └── .sub_coordinates: list of subsampled coordinates used to create pixel_table is subsampling is chosen
```

**utils**
```bash
utils
├── ReadMarkers: csv marker reading
└── SubsetCoordinates: subsample imaging data with or without a mask using uniform grid, random sampling, or pseudo random sampling
    └── FlattenZstack: flatten an xyc image to create a pandas data frame with per pixel information
```

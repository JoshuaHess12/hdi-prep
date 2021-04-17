FROM python:3.7

RUN pip install pathlib os h5py scikit-image numpy pandas random pyimzml nibabel scipy operator umap-learn uncertainties seaborn matplotlib ast scikit-learn

COPY . /app/

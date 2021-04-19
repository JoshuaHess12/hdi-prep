FROM python:3.8

RUN pip install scikit-image numpy pandas pyimzml nibabel scipy h5py pathlib umap-learn uncertainties seaborn matplotlib scikit-learn PyYAML

COPY . /app/

FROM tensorflow/tensorflow:2.2.0

RUN pip install scikit-image numpy pandas pyimzml nibabel scipy h5py pathlib umap-learn uncertainties seaborn matplotlib scikit-learn PyYAML

COPY . /app/
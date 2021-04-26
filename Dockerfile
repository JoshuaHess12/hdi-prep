FROM tensorflow/tensorflow:2.2.0

RUN apt update && apt -y upgrade

RUN apt-get -y install git

RUN pip install scikit-image numpy pandas pyimzml nibabel scipy h5py pathlib umap-learn uncertainties seaborn matplotlib scikit-learn PyYAML

RUN pip install -e git+https://github.com/JoshuaHess12/hdi-utils.git@main#egg=hdi-utils

COPY . /app/
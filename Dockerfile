FROM python:3.7

RUN pip install pathlib os h5py scikit-image numpy pandas random pyimzml nibabel scipy operator

COPY . /app/

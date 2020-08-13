import os
import h5py
import numpy as np

class Hdf5(object):
    def __init__(self, image_files):
        self.images = image_files
    
    def store_many_hdf5(self, images, labels = 'test', hdf5_dir = '/home/lch950721/hdf5/'):
        """ 
            Parameters :
            ----------------
            images  images array
            labels  labels array
        """
        num_images = len(self.images)

        # create a new HDF5 file
        file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "w")

        # create a dataset in the file
        dataset = file.create_dataset(
            "images", np.shape(images), h5py.h5t.STD_U8BE, data = images
        )
        meta_set = file.create_dataset(
            "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
        )
        file.close()
from os import listdir
import os
import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
import glob

if __name__ == '__main__':

    path = '/home/lch950721/Image/DIV2K_train_HR'
    files = listdir(path)
    
    image_files = sorted(glob.glob(path + '/*'))

    for file in files:
        with h5py.File('/home/lch950721/hdf5/test.h5', 'w') as hf:
            for i, img in enumerate(image_files):
                image = pil_image.open(image_files[i]).convert('RGB')
                image = np.array(image).astype(np.float32)
                hf.create_dataset(name='image'+str(i) ,data = image)
    
    # h5_file.close()
    #files = [f for f in listdir('/home/lch950721/Image') if isfile(join('/home/lch950721/Image/DIV2K_train_HR', f))]
    
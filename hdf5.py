from os import listdir
import os
import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
import glob

    # def saveFile(self, output_path, image_path):
    #     h5_file = h5py.File(output_path, 'w')
if __name__ == '__main__':

    path = '/home/lch950721/Image/DIV2K_train_HR'
    file_list = listdir(path)
    if os.path.isfile(path):
        image_files = sorted(glob.glob(path + '/*'))
        
        with h5py.File('/home/lch950721/hdf5/test.h5', 'w') as hf:
            for i, img in enumerate(image_files):
                image = pil_image.open(self.image_files[i]).convert('RGB')
                Xset = hf.create_dataset(
                    name = 'image'+str(i),
                    data = image
                )
    elif os.path.isdir(path):
            for files in file_list:
                print(files)
            # h5_file.create_group(files)    
    
    # h5_file.close()
    #files = [f for f in listdir('/home/lch950721/Image') if isfile(join('/home/lch950721/Image/DIV2K_train_HR', f))]
    
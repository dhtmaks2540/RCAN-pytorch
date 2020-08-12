# os 모듈 임포트
import os
# 운영체제에 등록되어 있는 모든 환경 변수 os 모듈의 environ이라는 속성으로 접근 가능
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# random 모듈
import random
# glob 모듈
# 파일들의 리스트를 뽑을 때 사용하는데, 파일의 경로명을 이용해서 사용
import glob
# numpy 모듈
# 다차원 배열과 이런 배열을 처리하는 다양한 함수와 툴 제공
import numpy as np
# Python Image Library
import PIL.Image as pil_image

# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.enable_eager_execution(config=config)

# Dataset 클래스
class Dataset(object):
    def __init__(self, images_dir, patch_size, scale, use_fast_loader=False):
        # golb.glob : 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환
        # image_dir 폴더 안의 모든 리스트 반환
        # sorted() : 이터러블로부터 새로운 정렬된 리스트를 만듬
        self.image_files = sorted(glob.glob(images_dir + '/*'))
        self.patch_size = patch_size
        self.scale = scale
        self.use_fast_loader = use_fast_loader

    # getitem 메소드
    def __getitem__(self, idx):
        # if self.use_fast_loader:
        #     hr = tf.read_file(self.image_files[idx])
        #     hr = tf.image.decode_jpeg(hr, channels=3)
        #     hr = pil_image.fromarray(hr.numpy())
        # pil_image.open을 통해 image_files로부터 인덱스에 맞게 이미지를 가져온다.
        # convert 메소드를 통해 RGB모드로 변환 
        hr = pil_image.open(self.image_files[idx]).convert('RGB')

        # randomly crop patch from training set
        # randint를 통해 0부터 hr이미지의 width, height에서 
        # patch_size * scale을 곱한 값을 뺀다.
        crop_x = random.randint(0, hr.width - self.patch_size * self.scale)
        crop_y = random.randint(0, hr.height - self.patch_size * self.scale)
        print('crop_x : ' + crop_x + ', crop_y : ' + crop_y)
        # crop(가로 시작점, 세로 시작점, 가로 범위, 세로 범위)
        hr = hr.crop((crop_x, crop_y, crop_x + self.patch_size * self.scale, crop_y + self.patch_size * self.scale))

        # degrade lr with Bicubic
        # patch_size, patch_size로 크기 조정하고 BICUBIC 보간법사용
        lr = hr.resize((self.patch_size, self.patch_size), resample=pil_image.BICUBIC)

        # hr과 lr의 이미지를 배열로 바꾸고 
        # astype을 통해 type을 데이터 타입을 변경(float32로)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)

        print('hr : ' + h + ', lr : ' + lr)

        # transpose를 통해 다차원의 텐서를 변형(0,1,2 차원을 2,0,1 차원으로)
        hr = np.transpose(hr, axes=[2, 0, 1])
        lr = np.transpose(lr, axes=[2, 0, 1])

        print('hr.transpose : ' + h + ', lr.transpose : ' + lr)

        # normalization(min - max 정규화)
        # 이미지데이터의 픽셀 정보는 0 ~ 255 사이의 값을 가진다
        # 이미지 픽셀 정보를 255로 나누어 0 ~ 1.0 사이의 값을 가지도록
        hr /= 255.0
        lr /= 255.0

        return lr, hr

    # len 메소드(image_files의 개수 반환)
    def __len__(self):
        return len(self.image_files)

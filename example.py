# 호출 당시 인자값을 줘서 동작을 다르게 하기위한 모듈
import argparse
# os 모듈 임포트
import os
# Python Image Library
import PIL.Image as pil_image
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
# 모델파일로부터 RCAN 클래스 임포트
from model import RCAN

cudnn.benchmark = True
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    print('Test... ')
    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser()
    # 입력받을 인자값 등록
    parser.add_argument('--arch', type=str, default='RCAN')
    parser.add_argument('--weights_path', type=str, default='/home/lch950721/Model/RCAN_loss_0.017220760196713463.pth')
    parser.add_argument('--image_path', type=str, default='/home/lch950721/Image/Urban100/img_078.png')
    parser.add_argument('--outputs_dir', type=str, default='/home/lch950721/Image/ImageResult')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_rg', type=int, default=10)
    parser.add_argument('--num_rcab', type=int, default=20)
    parser.add_argument('--reduction', type=int, default=16)
    # 입력받은 인자값을 opt에 저장
    opt = parser.parse_args()

    # outputs_dir이 존재하지 않으면 폴더만들기
    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    # model은 RCAN 클래스(매개변수로 opt)
    model = RCAN(opt)
    # state dict : 간단히 말해 각 계층을 매개변수 텐서로 매핑하는 Python 사전(dict) 객체
    state_dict = model.state_dict()
    # torch.load : 저장된 모델 불러오기(path, map_loaction : how to remap storage locations)
    for n, p in torch.load(opt.weights_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    # model을 device로 보냄
    model = model.to(device)
    # model eval 모드 설정
    model.eval()

    # filename을 image_path로부터 image를 얻어온 후 확장자를 떼고 넣어줍니다.
    filename = os.path.basename(opt.image_path).split('.')[0]
    # pil_image를 통해 image_path로부터 이미지를 열고 RGB 모드로 convert
    input = pil_image.open(opt.image_path).convert('RGB')
    # lr을 input을 이용해 resize 해주고 보간법으로 BICUBIC 사용
    lr = input.resize((input.width // opt.scale, input.height // opt.scale), pil_image.BICUBIC)
    # lr을 원래의 크기로 돌리기
    bicubic = lr.resize((input.width, input.height), pil_image.BICUBIC)
    # bicubic된 lr 이미지를 outputs_dir에 저장
    bicubic.save(os.path.join(opt.outputs_dir, '{}_x{}_bicubic.png'.format(filename, opt.scale)))
    # Tensor 형태로 만들고 0 위치에 1인 차원 추가 후 device로 보냄
    input = transforms.ToTensor()(lr).unsqueeze(0).to(device)
    # model에 input을 넣는데 no_grad를 통해 해당 블록을 기록 x 
    with torch.no_grad():
        pred = model(input)

    # pred : 255.0을 곱하는데 그 리턴값이 pred에 저장, clamp : 값을 특정범위로 묶는다.
    # squeeze : 1인 차원을 제거, permute : 차원을 교환
    output = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
    print(str(output))
    # fromarray를 통해 배열을 이미지로(모드는 RGB)
    output = pil_image.fromarray(output, mode='RGB')
    print(str(output))
    # 이미지를 outputs_dir에 저장
    output.save(os.path.join(opt.outputs_dir, '{}_x{}_{}.png'.format(filename, opt.scale, opt.arch)))


    print('Test end')

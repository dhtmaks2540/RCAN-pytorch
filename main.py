# 호출 당시 인자값을 줘서 동작을 다르게 하기위한 모듈
import argparse
import sys
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
# for문의 상태바를 알려준다.
from tqdm import tqdm
# 모델로부터 RCAN 클래스 가져오기
from model import RCAN
# dataset으로부터 Dataset 클래스 가져오기
from dataset import Dataset
# utils로부터 AverageMeter 클래스 가져오기
from utils import AverageMeter

# cuda benchmark mode true
cudnn.benchmark = True
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 인터프리터로 직접 실행했을 경우 아래 코드 실행
if __name__ == '__main__':
    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser()
    # 입력받을 인자값 등록
    parser.add_argument('--arch', type=str, default='RCAN')
    parser.add_argument('--images_dir', type=str, default='/home/lch950721/Image/DIV2K_train_HR/')
    parser.add_argument('--outputs_dir', type=str, default='/home/lch950721/Model/')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_rg', type=int, default=10)
    parser.add_argument('--num_rcab', type=int, default=20)
    parser.add_argument('--reduction', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--use_fast_loader', action='store_true')
    # 입력받은 인자값을 opt에 저장
    opt = parser.parse_args()

    # opt에 저장된 outputs_dir이 존재하지 않으면
    # outputs_dir 디렉토리를 생성하라
    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    # 랜덤생성에 사용되는 시드 설정(opt에 저장된 seed로)
    torch.manual_seed(opt.seed)

    # model(RCAN)에 opt를 넣어주고 device로 보낸다.
    model = RCAN(opt).to(device)

    # L1Loss : 각 원소별 차이의 절대값을 계산
    criterion = nn.L1Loss()

    # 최적화 함수로 optim.adam사용
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # dataset으로 opt에 저장된 image_dir, patch_size, scale, use_fast_loader를
    # 매개변수로 넣고 생성
    dataset = Dataset(opt.images_dir, opt.patch_size, opt.scale, opt.use_fast_loader)
    
    # dataloader로는 DataLoader 클래스에 데이터셋, batch_size, shuffle 주고 
    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            # num_workers로 opt에 저장된 threads 사용
                            # data 로딩을 위해 몇 개의 서브 프로세스를 사용할 것인지
                            num_workers=opt.threads,
                            # 고정된 영역의 데이터 복사본을 얻을 수 있다.
                            pin_memory=True,
                            # 제일 마지막 batch가 batch 수 보다 작을 경우 그냥
                            # 그 배치를 버리는 것
                            drop_last=True)

    min = sys.maxsize

    # opt에 저장된 num_epochs 만큼 epoch가 돌며
    for epoch in range(opt.num_epochs):
        # AverageMeter을 통해 loss 초기화
        epoch_losses = AverageMeter()
        # 전체 데이터셋의 개수에서 전체 데이터셋의 개수 % batch_size만큼을 빼고 tqdm으로 만들어준다.
        # with as : with 블럭이 시작되는 시점에 뒤를 실행하고 끝나면 as를 실행
        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            # set_description을 통해 프로그레스 바에 대한 묘사 추가
            # epoch + 1 , opt의 num_epochs(총 epoch) 를 보여줌
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, opt.num_epochs))
            # DataLoader : 학습에 쓰일 데이터 전체를 보관했다가 train 함수가 데이터를
            # 요구하면 사전에 저장된 batch size 만큼 return
            # dataloader안의 data만큼 for문을 돌며
            for data in dataloader:
                # inputs과 labels로 데이터 대입(lr - inputs, hr - labels)
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)
                # inputs을 가지고 RCAN 모델에 넣은 후 preds 변수에 대입
                preds = model(inputs)
                # preds와 labels를 가지고 오차함수 계산
                loss = criterion(preds, labels)
                # loss 업데이트(loss 스칼라값과 inputs의 개수)
                epoch_losses.update(loss.item(), len(inputs))
                # 기울기 0 설정
                optimizer.zero_grad()
                # 오차 함수를 가중치로 미분, 오차가 최소가 되는 방향을 구함
                loss.backward()
                # 가중치를 학습률만큼 갱신
                optimizer.step()
                # set_postfix를 통해 프로그레스바의 끝에 값 추가(loss의 평균)
                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                # 프로그레스바 update(inputs의 개수 만큼)
                _tqdm.update(len(inputs))

        # min이 losses.avg보다 크면 min 바꾸고 모델 저장
        if(min > epoch_losses.avg):
            min = epoch_losses.avg
            best = model.state_dict()

    # best model을 저장, outputs_dir 폴더로
    torch.save(best, os.path.join(opt.outputs_dir, '{}_loss_{}.pth'.format(opt.arch, min)))
        

# nn 패키지를 통해 신경망 사용
from torch import nn

# ChannelAttention 클래스(nn.Module 클래스를 상속)
# nn.Module 신경망 모듈을 상속받으면 파이토치 프레임워크에 있는 도구 쉽게 적용
class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        # super를 통해 파이토치의 nn.Module 클래스의 속성들을 가지고 초기화
        super(ChannelAttention, self).__init__()
        # Sequential : 순차적으로 실행하도록 담는 container
        self.module = nn.Sequential(
            # 여러 개의 입력 평면으로 구성된 입력 신호에 2D 적응형 최대 풀링을 적용(pooling layer)
            nn.AdaptiveAvgPool2d(1),
            # 컨볼루션(입력 채널수, 출력 채널수, 각 컨볼루션 계층의 필터크기(kerner_size)) (//  : 몫) -> downscale
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            # ReLU 활성화함수(inplace를 True로 줘서 input으로 들어온 것 자체를 수정 -> 메모리 usage 좋아짐)
            nn.ReLU(inplace=True),
            # upscale
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    # forward 메소드(매개변수로 들어온 x와 module을 통해 나온 값을 곱)
    def forward(self, x):
        # Channel Attention 시작 전의 값과 후의 값 곱하기 -> rescale
        return x * self.module(x)

# RCAB 클래스(nn.Module 클래스를 상속) - Residual Channel Attention Block
class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        # Sequential을 통해 순차적으로 실행
        self.module = nn.Sequential(
            # 컨볼루션(입력 채널 수, 출력 채널수, 각 컨볼루션 계층의 필터크기, 패딩값)
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            # 나온 값을 channelAttention 클래스에 넣어 인스턴스 생성
            ChannelAttention(num_features, reduction)
        )
    # forward 메소드(매개변수로 들어온 x와 module을 통해 나온 값을 더함)
    def forward(self, x):
        # RCAB 시작 전의 값과 후의 값을 더하기
        return x + self.module(x)

# RG 클래스(nn.Module 클래스를 상속) - Residual Group
class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):
        super(RG, self).__init__()
        # module은 매개변수로 들어온 num_features와 reduction을 통해 RCAB 인스턴스를 생성하는데 num_rcab만큼 생성해 리스트로 지정
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]
        # 컨볼루션(입력 채널 수, 출력 채널 수, 각 컨볼루션 계층의 필터 크기, 패딩값)
        # append를 통해 module의 마지막에 컨볼루션값 대입
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        # sequential을 통해 module을 순차적으로 실행(*은 파라미터를 받는것)
        self.module = nn.Sequential(*self.module)
    # forward 메소드(매개변수로 들어온 x와 module을 통해 나온 값을 더함)
    def forward(self, x):
        # RG 시작 전의 값과 후의 값을 더하기(SSC)
        return x + self.module(x)

# RCAN 클래스(nn.Module 클래스를 상속)
class RCAN(nn.Module):
    def __init__(self, args):
        super(RCAN, self).__init__()
        # 매개변수로 들어온 args를 통해 값들을 넣어준다.
        scale = args.scale 
        num_features = args.num_features
        num_rg = args.num_rg
        num_rcab = args.num_rcab
        reduction = args.reduction
        # extract the shallow filter
        self.sf = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        # num_features와 num_crab, reduction을 가지고
        # RG 인스턴스를 생성한 후 num_rg만큼 생성해 리스트로 지정
        self.rgs = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])
        # 컨볼루션 실행
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        # Sequential을 통해 컨볼루션과 PixelShuffle() - upscale 모듈
        self.upscale = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),
            # Rearrange elements -> useful for implementing efficient sub-pixel convolution
            # lr 이미지크기 hr 이미지크기로
            nn.PixelShuffle(scale)
        )
        # 컨볼루션 실행(reconstruct)
        self.conv2 = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)
    # forward 메소드
    def forward(self, x):
        # 매개변수 x를 가지고 컨볼루션 실행
        x = self.sf(x)
        # residual에 x값 대입(RCAN 시작 전의 값)
        residual = x
        # 컨볼루션 수행하고 나온 x를 가지고 RG 생성
        x = self.rgs(x)
        # RG을 수행하고 나온 x를 가지고 컨볼루션 실행
        x = self.conv1(x)
        # conv1를 수행하고 나온 x에 residual값(sf를 수행하고 나온 x값)을 더함(LSC) -> bypassed abundant low - frequency information
        x += residual
        # 더해줘서 나온 x에 upscale 메소드 실행
        x = self.upscale(x)
        # upscale를 수행하고 나온 x에 conv2 메소드 실행
        x = self.conv2(x)
        return x

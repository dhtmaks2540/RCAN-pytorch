# AverageMeter 클래스(object 클래스를 상속)
class AverageMeter(object):
    # 생성자에서 reset 메소드를 통해 인스턴스 초기화
    def __init__(self):
        self.reset()

    # reset 메소드(val, avg, sum, count 값을 0으로 초기화)
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # update 메소드(매개변수를 통해 들어오는 val값과 1의 값을 통해 값들을 업데이트)
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

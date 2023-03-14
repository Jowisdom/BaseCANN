# RingCANN
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

class RingCANN():

    def __init__(self):
        self.CAN_RANGE = 360
        self.CAN_SIZE = 36
        self.unit = self.CAN_RANGE / self.CAN_SIZE
        self.STD = 3
        self.MIN = 0
        self.MAX = 35
        self.x = np.linspace(self.MIN,self.MAX,self.CAN_SIZE)
        self.cann = np.zeros_like(self.x)
        self.max_neuron = 0
        self.max_neuron_repre = 0
        self.alpha = 0.7 # 选择相信后续的观察值的比例
    # 局部刺激和抑制
    # key:要刺激的位置或索引
    # 返回x索引
    # 返回ey+shift_iy: 刺激后的值
    # def activity(self,key):
    #     ey = ss.norm.pdf(self.x, key, self.STD)
    #     iy = -ss.norm.pdf(self.x, key, self.STD*3)
    #     half_dist = int(self.CAN_SIZE/2)
    #     shift_iy = np.roll(iy,half_dist)
    #     self.cann = ey+shift_iy+self.cann+iy
    def activity(self,key):
        half_dist = int(self.CAN_SIZE / 2)
        ey = ss.norm.pdf(self.x, half_dist , self.STD)
        iy = -ss.norm.pdf(self.x, half_dist , self.STD*3)
        shift_ey = np.roll(ey,key-half_dist)
        shift_iy = np.roll(iy,key)
        self.cann = (1-self.alpha) * self.cann+ self.alpha *(shift_ey+shift_iy)

    def normalize(self):
        # The minimal value of the CAN is set to zero then its area is set to unity
        zeros = np.zeros_like(self.cann)
        self.cann = np.where(self.cann <= zeros, zeros, self.cann)
        s = np.sum(self.cann)
        self.cann = self.cann / s


    # transform bump
    def shitf_bump(self,int_dist):
        self.cann = np.roll(self.cann,int_dist)

    # find_best
    def find_max_bump(self):
        max_value = max(self.cann)
        max_indices = np.where(self.cann == max_value)
        self.max_neuron = max_indices[0][0]
        self.max_neuron_repre = self.max_neuron* self.unit



if __name__ == '__main__':
    rcann = RingCANN()

    rcann.activity(23)
    plt.plot(rcann.x, rcann.cann, lw=3)
    rcann.normalize()
    plt.plot(rcann.x, rcann.cann, lw=3)
    rcann.shitf_bump(10)
    plt.plot(rcann.x, rcann.cann, lw=3)
    plt.legend(['ext-23', 'norm-after', 'shift-10'])
    rcann.find_max_bump()
    print(rcann.max_neuron)
    print(rcann.max_neuron_repre)
    plt.show()
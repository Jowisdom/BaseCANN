# RingCANN
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

# DonutCANN
import math
# import generate_path


class CANN(object):
    def __init__(self, arr_size, per_step, minVoltageLevel=0, maxVoltageLevel=1.0, injectEnergyValue=1):
        self.net_size = arr_size
        self.net_dim = arr_size.ndim + 1
        self.per_step = per_step  # 相邻神经元的单位长度
        self.minVoltageLevel = minVoltageLevel
        self.maxVoltageLevel = maxVoltageLevel
        self.injectEnergyValue = injectEnergyValue
        self.matCANN = np.zeros(arr_size) * self.minVoltageLevel
        self.index_best_neuron = np.zeros(self.net_dim)
        self.alpha = 0.7  # 系数，更相信观测值还是路径积分的值（更相信旧值还是新值）

    # 找到当前能量最大的位置
    def find_best_neuron(self):
        # print(np.max(self.matCANN))
        temp_index_best_neuron = np.where(self.matCANN == np.max(self.matCANN))
        self.index_best_neuron = np.array(temp_index_best_neuron).T.reshape(2)
        self.index_best_neuron = np.flip(self.index_best_neuron, 0)

    def global_inhibition(self):
        zeros = np.zeros_like(self.matCANN)
        self.matCANN = np.where(self.matCANN <= zeros, zeros, self.matCANN)
        s = np.sum(self.matCANN)
        self.matCANN = self.matCANN / s
        # print(f'All energy in network is {self.matCANN.sum()}')


class TwoDimCANN(CANN):
    def __init__(self, arr_size, per_step, minVoltageLevel=0.1, maxVoltageLevel=1, injectEnergyValue=1):
        super().__init__(arr_size, per_step, minVoltageLevel, maxVoltageLevel, injectEnergyValue)
        self.x, self.y = np.mgrid[0:self.net_size[0], 0:self.net_size[1]]
        self.pos = np.dstack((self.x, self.y))

    def localised_excitation(self, np_inject_position):
        # self.find_best_neuron()
        # x,y = np.mgrid[0:self.net_size[0],0:self.net_size[1]]
        # pos = np.dstack((x, y))
        # 生成居中的权重，参见 https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
        ey = ss.multivariate_normal([int(self.net_size[0] / 2), int(self.net_size[1] / 2)],
                                    [[self.net_size[0], 0], [0, self.net_size[0]]])
        iy = ss.multivariate_normal([int(self.net_size[0] / 2), int(self.net_size[1] / 2)],
                                    [[3 * self.net_size[0], 0], [0, 3 * self.net_size[0]]])
        ey = ey.pdf(self.pos)
        iy = -iy.pdf(self.pos)
        # 将居中的权重移动到激活区域
        shift_ey = np.roll(ey, [int(np_inject_position[0] - int(self.net_size[0] / 2)), 0], 1)
        shift_ey = np.roll(shift_ey, [0, int(np_inject_position[1] - int(self.net_size[1] / 2))], 0)
        shift_iy = np.roll(iy, [int(np_inject_position[0]), 0], 1)
        shift_iy = np.roll(shift_iy, [0, int(np_inject_position[1])], 0)
        self.matCANN = (1 - self.alpha) * self.matCANN + self.alpha * (shift_ey + shift_iy)

        # plt.matshow(self.matCANN)

    # 注入能量后动态调整
    def excitation_and_dynamic_adjust(self, np_position):
        self.localised_excitation(np_position)
        self.global_inhibition()
        pass

    # 找到注入能量的位置,
    # 输入arr_transform为位移
    def transfer_bump(self, arr_transform):
        self.matCANN = np.roll(self.matCANN, [arr_transform[0], 0], 1)
        self.matCANN = np.roll(self.matCANN, [0, arr_transform[1]], 0)
        pass


if __name__ == '__main__':
    ############################ Simple Test ############################
    NET_SIZE = 200
    SPACING = 1.0
    ORIENTATION = 0.0
    NEURON_STEP = SPACING / NET_SIZE
    cann = TwoDimCANN(np.array([NET_SIZE, NET_SIZE]), NEURON_STEP)

    cann.excitation_and_dynamic_adjust(np.array([25, 55]))
    cann.find_best_neuron()
    print(cann.index_best_neuron)
    plt.matshow(cann.matCANN)
    cann.excitation_and_dynamic_adjust(np.array([10, 10]))
    cann.find_best_neuron()
    print(cann.index_best_neuron)
    plt.matshow(cann.matCANN)
    cann.transfer_bump(np.array((-20, 1)))
    cann.find_best_neuron()
    print(cann.index_best_neuron)
    plt.matshow(cann.matCANN)

    # print(cann.matCANN)
    plt.show()
    ######################################################################
    ####################### 动态转移 #######################################
    # cann = TwoDimCANN(np.array([20, 20]), 1)
    # cann.dynamic_adjust(np.array([0,0]))
    # cann.find_best_neuron()
    # print(cann.index_best_neuron)
    # N =300
    # i = 0
    # while i < N:
    #     plt.cla()
    #     plt.imshow(cann.matCANN, cmap=plt.get_cmap('plasma'))
    #
    #     a = np.random.randint(0, 6)
    #     b = np.random.randint(0, 6)
    #     cann.transfer_bump(np.array([a, b]))
    #     i += 1
    #     # str = './phase_CAN_fig/fig_{}.png'.format(i)
    #     # plt.savefig(str, dpi=300, format='png', bbox_inches='tight', pad_inches=0.1)
    #     plt.pause(0.01)
    #
    # plt.colorbar()
    # plt.close()
    # plt.show()
    #########################################################################

import numpy as np
import matplotlib.pyplot as plt

data = np.load('feature/20190215.npy')

def compare_snr():
    fig = plt.figure()
    ax1 = fig.add_subplot('211')
    ax2 = fig.add_subplot('212')
    # ax2 = ax1.twinx()
    # 2----wake   1----light sleep   0----deep sleep

    time = range(len(data))
    ax1.plot(time, data[:, 1], 'g', label="snr_max")
    ax2.plot(time, data[:, -1], 'r', label="stage")
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax2.set_xlabel('time line')
    plt.show()


if __name__ == '__main__':
    compare_snr()
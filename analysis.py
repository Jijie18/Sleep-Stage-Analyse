import numpy as np
import matplotlib.pyplot as plt
import load_mat

global data


def compare_truth(data, fit_res):
    fig = plt.figure()
    ax1 = fig.add_subplot('211')
    ax2 = fig.add_subplot('212')
    # ax2 = ax1.twinx()
    # 2----wake   1----light sleep   0----deep sleep

    time = range(len(data))
    ax1.plot(time, fit_res, 'g', label="stage_predict")
    ax2.plot(time, data[:, -1], 'r', label="stage_truth")
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax2.set_xlabel('time line')
    plt.show()


def compare_speed():
    fig = plt.figure()
    ax1 = fig.add_subplot('211')
    ax2 = fig.add_subplot('212')
    # ax2 = ax1.twinx()
    # 2----wake   1----light sleep   0----deep sleep

    time = range(len(data))
    ax1.plot(time, data[:, 0], 'g', label="speed_max")
    ax2.plot(time, data[:, -1], 'r', label="stage")
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax2.set_xlabel('time line')
    plt.show()


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


def compare_other(col, col_label):
    fig = plt.figure()
    ax1 = fig.add_subplot('211')
    ax2 = fig.add_subplot('212')
    # ax2 = ax1.twinx()
    # 2----wake   1----light sleep   0----deep sleep

    time = range(len(data))
    ax1.plot(time, data[:, col], 'g', label=col_label)
    ax2.plot(time, data[:, -1], 'r', label="stage")
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax2.set_xlabel('time line')
    plt.show()


if __name__ == '__main__':
    data = load_mat.get_feature_sample("G:/资料/Radar_Sleep_pointCloud/20190211", col_name="11th Feb")
    # compare_speed()
    # compare_snr()
    # compare_other(2, 'speed_var')
    # compare_other(3, 'snr_var')
    # compare_other(4, 'x_var')
    # compare_other(5, 'y_var')
    # compare_other(6, 'speed_mean')
    # compare_other(7, 'snr_mean')
    compare_other(8, 'strong_rate')

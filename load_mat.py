from scipy.io import loadmat
import os
import numpy as np
import xlrd

# ==========get ground truth from xiao_mi ring=========
f = xlrd.open_workbook("feature/Xiaomi_ring_result.xlsx")

# the path store point cloud data
path = "G:/资料/Radar_Sleep_pointCloud"
dir = os.listdir(path)
NUM_OF_FRAME = 1200


def get_feature_sample(path, ground_truth=None, col_name=None):
    if not col_name and not ground_truth:
        raise Exception("must provide one ground truth")
    if not ground_truth:
        ground_truth = f.sheet_by_name(col_name).col_values(1)
    files = os.listdir(path)
    features = np.zeros((len(files), 10))
    index = 0
    # ==========get each minutes data.=========
    for file in files:
        newpath = path + "/" + file
        data = loadmat(newpath).get("savedata1")

        # all of those information stored in data_all format [distance_x, distance_y, speed, snr]
        all_data = []
        dataFrame = []  # stores the frame data between two 999
        hasInfo = False
        for j in data:
            if j[0] == 999:
                if hasInfo:
                    hasInfo = False
                    res = process_frame(dataFrame)
                    all_data.append(res)
            else:  # the state has been hasInfo, just add info to it
                dataFrame.append(j)
                hasInfo = True
        features[index][-1] = ground_truth[index]
        if not all_data:
            index = index + 1
            continue
        all_data_array = np.asarray(all_data)
        for i, feature in enumerate(all_data_array.max(0)[0:2]):
            features[index][i] = feature  # 2 features
        for i, feature in enumerate(all_data_array.var(0)[0:4]):
            features[index][2 + i] = feature  # 4 features
        sum_minu = all_data_array.sum(0)
        for i, feature in enumerate(sum_minu[0:2]):
            features[index][6 + i] = feature / NUM_OF_FRAME  # 3 features
        features[index][8] = sum_minu[-1] / NUM_OF_FRAME  # 3 features
        index = index + 1
    return features


def checkSize():
    for k in dir:
        path_fur = path + "/" + k
        files = os.listdir(path_fur)
        print(path_fur, len(files), files[0], files[-1])


# checkSize()
# pass

def process_frame(dataSeg: list):
    """
    :param dataSeg:
    :return: res[Speed_max, Snr_max, x_mean, y_mean, 活动强烈等级]
    """
    dataSeg_array = np.abs(dataSeg)
    dataSeg_bak = np.asarray(dataSeg)
    res = []
    for d in dataSeg_array.max(0)[2:]:
        res.append(d)
    res.append(sum(dataSeg_bak[:, 0])/len(dataSeg)) # get x mean position
    res.append(sum(dataSeg_bak[:, 1])/len(dataSeg)) # get y mean position
    # res.append(len(dataSeg))  # 活动部位数
    res.append(0)
    if res[0] > 1 or res[1] > 100:
        res[-1] += 1
    if res[0] > 2 or res[1] > 400:
        res[-1] += 1
    dataSeg.clear()
    return res


if __name__ == '__main__':
    for colIndex, k in enumerate(dir):  # k is a folder which contains all data in one night
        path_fur = path + "/" + k
        ground_truth = f.sheet_by_index(len(dir) - 1 - colIndex).col_values(1)

        features = get_feature_sample(path_fur, ground_truth=ground_truth)

        np.save("feature/" + k, features)
        print("C", k)

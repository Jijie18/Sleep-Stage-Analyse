from scipy.io import loadmat
import os
import numpy as np
import xlrd

# ==========get ground truth from xiao_mi ring=========
f = xlrd.open_workbook("feature/Xiaomi_ring_result.xlsx")
ground_truth = f.sheet_by_index(0).col_values(1)

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
    features = np.zeros((len(files), 12))
    index = 0
    # ==========get each minutes data.=========
    for file in files:
        newpath = path + "/" + file
        data = loadmat(newpath).get("savedata1")

        # all of those information stored in data_all format [distance_x, distance_y, speed, snr]
        all_data = []
        dataSeg = []  # stores the segment data between two 999
        hasInfo = False

        for j in data:
            if j[0] == 999:
                if hasInfo:
                    hasInfo = False
                    res = process_segment(dataSeg)
                    all_data.append(res)
            else:  # the state has been hasInfo, just add info to it
                dataSeg.append(j)
                hasInfo = True
        features[index][-1] = ground_truth[index]
        features[index][7] = features[index - 1][0]
        features[index][8] = features[index - 1][1]
        features[index][9] = features[index - 2][0]
        features[index][10] = features[index - 2][1]
        if not all_data:
            index = index + 1
            continue
        all_data_array = np.asarray(all_data)
        for i, feature in enumerate(all_data_array.max(0)[0:2]):
            features[index][i] = feature  # 2 features
        for i, feature in enumerate(all_data_array.var(0)[0:2]):
            features[index][2 + i] = feature  # 2 features
        for i, feature in enumerate(all_data_array.sum(0)):
            features[index][4 + i] = feature / NUM_OF_FRAME  # 3 features

        index = index + 1
    features[0][7] = features[- 1][0]
    features[0][8] = features[- 1][1]
    features[0][9] = features[-2][0]
    features[0][10] = features[-2][1]
    features[1][9] = features[0][0]
    features[1][10] = features[0][1]
    return features


def checkSize():
    for k in dir:
        path_fur = path + "/" + k
        files = os.listdir(path_fur)
        print(path_fur, len(files), files[0], files[-1])


# checkSize()
# pass

def process_segment(dataSeg: list):
    """
    :param dataSeg:
    :return: res[Speed_max, Snr_max, 活动强烈等级]
    """
    dataSeg_array = np.abs(dataSeg)
    res = []
    for d in dataSeg_array.max(0)[2:]:
        res.append(d)
    # res.append(len(dataSeg))  # 活动部位数
    for d in dataSeg_array:
        if d[-1] > 100 or d[-2] > 1:
            res.append(1)  # 活动是否足够强烈
            dataSeg.clear()
            return res
    res.append(0)
    dataSeg.clear()
    return res


if __name__ == '__main__':
    for colIndex, k in enumerate(dir):  # k is a folder which contains all data in one night
        path_fur = path + "/" + k
        ground_truth = f.sheet_by_index(len(dir) - 1 - colIndex).col_values(1)

        features = get_feature_sample(path_fur, ground_truth=ground_truth)

        np.save("feature/" + k, features)
        print("C", k)

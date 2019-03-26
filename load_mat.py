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
    :return: res[X_mean, Y_mean, Speed_mean, Snr_mean, 活动部位数, 活动强烈等级]
    """
    dataSeg_array = np.array(dataSeg)
    res = []
    for d in dataSeg_array.mean(0):
        res.append(d)
    res.append(len(dataSeg))  # 活动部位数
    for d in dataSeg:
        if d[-1] > 10:
            res.append(1)  # 活动是否足够强烈
            dataSeg.clear()
            return res
    res.append(0)
    dataSeg.clear()
    return res


if __name__ == '__main__':
    for colIndex, k in enumerate(dir):  # k is a folder which contains all data in one night
        path_fur = path + "/" + k
        files = os.listdir(path_fur)
        features = np.zeros((len(files), 11))
        index = 0
        ground_truth = f.sheet_by_index(len(dir)-1-colIndex).col_values(1)
        # ==========get each minutes data.=========
        for file in files:
            newpath = path_fur + "/" + file
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
            if not all_data:
                index = index + 1
                continue
            all_data_array = np.asarray(all_data)
            for i, feature in enumerate(all_data_array.mean(0)[0:4]):
                features[index][i] = feature
            for i, feature in enumerate(all_data_array.var(0)[0:4]):
                features[index][4+i] = feature
            for i, feature in enumerate(all_data_array.sum(0)[-2:]):
                features[index][8+i] = feature
            index = index + 1
        # features[0][9] = feature[- 1][4]

        np.save("feature/" + k, features)
        print("C", k)

from scipy.io import loadmat
import os
import numpy as np
path = "D:/chuangxin/Radar_Sleep_pointCloud"
dir = os.listdir(path)
for k in dir:
    path_fur = path+"/"+k
    files = os.listdir(path_fur)
    feature = np.zeros((len(files),5))
    index = 0
    for i in files:
        newpath = path_fur+"/"+i
        data = loadmat(newpath).get("savedata1")
        distance_x = []
        distance_y = []
        speed = []
        snr = []
        for j in data[:]:
            if(j[0]!=999):
                distance_x.append(j[0])
                distance_y.append(j[1])
                speed.append(j[2])
                snr.append(j[3])
        distance_x_tmp = np.array(distance_x)
        distance_y_tmp = np.array(distance_y)
        speed_tmp = np.array(speed)
        snr_tmp = np.array(snr)
        feature[index][0] = distance_x_tmp.var()
        feature[index][1] = distance_y_tmp.var()
        feature[index][2] = speed_tmp.mean()
        feature[index][3] = speed_tmp.var()
        feature[index][4] = snr_tmp.mean()
        index = index+1

    np.save(k, feature)
    print("C")



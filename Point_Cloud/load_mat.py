from scipy.io import loadmat
import os
import numpy as np


# the path store point cloud data
path = "G:/资料/Radar_Sleep_pointCloud"
dir = os.listdir(path)

if __name__ == '__main__':
    for k in dir:
        path_fur = path+"/"+k
        files = os.listdir(path_fur)
        feature = np.zeros((len(files),10))
        index = 0
        # ==========get each minutes data.=========
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
            try:
                feature[index][5] = speed_tmp.min()
                feature[index][6] = speed_tmp.max()
                feature[index][7] = snr_tmp.min()
                feature[index][8] = snr_tmp.max()
            except:
                pass
            feature[index][9] = feature[index-1][4]
            index = index+1
        #
        # feature[0][5] = feature[- 1][5]
        # feature[0][6] = feature[- 1][6]
        # feature[0][7] = feature[- 1][7]
        # feature[0][8] = feature[- 1][8]
        feature[0][9] = feature[- 1][4]

        np.save("feature/"+k, feature)
        print("C", k)



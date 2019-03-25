import numpy as np
import random
import sys
import xlrd
from sklearn.svm import SVC


class SVM:
    def __init__(self, x, y, epochs=500, learning_rate=0.01):
        self.x = np.c_[np.ones((x.shape[0])), x]
        self.y = y
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.w = np.random.uniform(size=np.shape(self.x)[1], )

    def get_lost(self, x, y):
        loss = max(0, 1 - y * np.dot(x, self.w))
        return loss

    def cal_sgd(self, x, y, w):
        if y * np.dot(x, self.w) < 1:
            w = w + self.learning_rate * y * x
        else:
            w = w
        return w

    def train(self):
        for epochs in range(self.epochs):
            randomize = np.arange(len(self.x))
            np.random.shuffle(randomize)
            x = self.x[randomize]
            y = np.array(self.y)[randomize]
            loss = 0
            for xi, yi in zip(x, y):
                loss += self.get_lost(xi, yi)
                self.w = self.cal_sgd(xi, yi, self.w)
            # print("epochs:{0} loss:{1}".format(epochs,loss))

    def predict(self, x):
        x_test = np.c_[np.ones((x.shape[0])), x]
        return np.dot(x_test, self.w)


if __name__ == '__main__':
    data = np.load('feature/20190215.npy')
    data = np.r_[data, np.load('feature/20190214.npy')]
    data = np.r_[data, np.load('feature/20190213.npy')]
    data = np.r_[data, np.load('feature/20190212.npy')]
    data = np.r_[data, np.load('feature/20190211.npy')]
    data = np.r_[data, np.load('feature/20190210.npy')]
    data = np.r_[data, np.load('feature/20190209.npy')]
    print("open workspace")
    f = xlrd.open_workbook("feature/Xiaomi_ring_result.xlsx")
    print("open complete")
    ground_truth = f.sheet_by_index(0).col_values(1)
    ground_truth = np.r_[ground_truth, f.sheet_by_index(1).col_values(1)]
    ground_truth = np.r_[ground_truth, f.sheet_by_index(2).col_values(1)]
    ground_truth = np.r_[ground_truth, f.sheet_by_index(3).col_values(1)]
    ground_truth = np.r_[ground_truth, f.sheet_by_index(4).col_values(1)]
    ground_truth = np.r_[ground_truth, f.sheet_by_index(5).col_values(1)]
    ground_truth = np.r_[ground_truth, f.sheet_by_index(6).col_values(1)].astype(np.uint8)

    # 2----wake   1----light sleep   0----deep sleep

    clf = SVC(decision_function_shape='ovr', kernel="rbf")
    # a.一对多法（one-versus-rest, 简称1-v-r SVMs）。
    # 训练时依次把某个类别的样本归为一类, 其他剩余的样本归为另一类，
    # 这样k个类别的样本就构造出了k个SVM。分类时将未知样本分类为具有最大分类函数值的那类。

    clf.fit(data, ground_truth)
    prer = clf.predict(data)
    print(sum([1 if _== 0 else 0 for _ in (prer-ground_truth)])/len(data))

    test_case = np.load('feature/20190130.npy')
    ground_truth_test = f.sheet_by_name("30th Jan").col_values(1)
    test_prer = clf.predict(test_case)
    print(sum([1 if _ == 0 else 0 for _ in (test_prer - ground_truth_test)]) / len(test_case))
    exit(0)


    ground_truth_wake = []
    ground_truth_ls = []
    ground_truth_ds = []
    for i in ground_truth:
        if i == 2:
            ground_truth_wake.append(1)
            ground_truth_ls.append(-1)
            ground_truth_ds.append(-1)
        elif i == 1:
            ground_truth_wake.append(-1)
            ground_truth_ls.append(1)
            ground_truth_ds.append(-1)
        else:
            ground_truth_wake.append(-1)
            ground_truth_ls.append(-1)
            ground_truth_ds.append(1)

    print("I don;t want yo connitune")
    svm_wake = SVM(data, ground_truth_wake)
    svm_ls = SVM(data, ground_truth_ls)
    svm_ds = SVM(data, ground_truth_ds)

    svm_wake.train()
    svm_ls.train()
    svm_ds.train()

    test_case = np.load('feature/20190130.npy')
    ground_truth_test = f.sheet_by_name("30th Jan").col_values(1)
    test_result_wake = svm_wake.predict(test_case)
    test_result_ls = svm_ls.predict(test_case)
    test_result_ds = svm_ds.predict(test_case)

    predict_result = []
    for i in range(len(ground_truth_test)):
        max_num = max(test_result_wake[i], test_result_ls[i], test_result_ds[i])
        if max_num == test_result_wake[i]:
            predict_result.append(2)
        elif max_num == test_result_ls[i]:
            predict_result.append(1)
        else:
            predict_result.append(0)

    correct_num = 0
    for i in range(len(predict_result)):
        if predict_result[i] == ground_truth_test[i]:
            correct_num = correct_num + 1
    print(correct_num / len(predict_result))
    exit(0)

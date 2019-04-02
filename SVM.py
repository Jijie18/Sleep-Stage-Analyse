import numpy as np
from sklearn.svm import SVC
import analysis

global clf


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


def validate(test_case):
    ground_truth_test = test_case[:, -1]
    test_case = test_case[:, 0:-1]
    test_prer = clf.predict(test_case)
    print("validate correct rate", sum([1 if _ == 0 else 0 for _ in (test_prer - ground_truth_test)]) / len(test_case))
    corr_res = correct(test_prer, window=15)
    print("correct train result(correct rate)",
          sum([1 if _ == 0 else 0 for _ in (corr_res - ground_truth_test)]) / len(test_case), '\n')

    return test_prer, corr_res


def correct(prer, window=5):
    """
    :return: correct predict result, look the 5 data point around to correct
    """
    correct_res = prer.copy()
    count = [0, 0, 0]
    length = len(prer)
    for i, stage in enumerate(prer):
        j = i - (window >> 1)
        end = i + (window >> 1)
        while j <= end and j < length:
            if j < 0:
                j += 1
                continue
            count[int(prer[j])] += 1
            j += 1
        if count[2] >= count[1]:
            if count[2] >= count[0]:
                correct_res[i] = 2
            else:
                correct_res[i] = 0
        elif count[1] > count[0]:
            correct_res[i] = 1
        else:
            correct_res[i] = 0
        count[0] = 0
        count[1] = 0
        count[2] = 0
    return correct_res


if __name__ == '__main__':
    data = np.load('feature/20190215.npy')
    # data = np.r_[data, np.load('feature/20190214.npy')]
    # data = np.r_[data, np.load('feature/20190213.npy')]
    data = np.r_[data, np.load('feature/20190212.npy')]
    data = np.r_[data, np.load('feature/20190211.npy')]
    data = np.r_[data, np.load('feature/20190210.npy')]
    data = np.r_[data, np.load('feature/20190209.npy')]
    data = np.r_[data, np.load('feature/20190207.npy')]
    # data = np.r_[data, np.load('feature/20190204.npy')]
    # data = np.r_[data, np.load('feature/20190203.npy')]
    data = np.r_[data, np.load('feature/20190131.npy')]
    data = np.r_[data, np.load('feature/20190130.npy')]
    data = np.r_[data, np.load('feature/20190128.npy')]

    test_data0 = np.load('feature/20190214.npy')
    test_data1 = np.load('feature/20190213.npy')
    data = np.r_[data, test_data0]
    data = np.r_[data, test_data1]

    test_data2 = np.load('feature/20190204.npy')
    test_data3 = np.load('feature/20190203.npy')

    np.random.shuffle(data)
    train_data_num = int(len(data) * 0.7)
    test_data = data[train_data_num:]
    data = data[0:train_data_num]

    # 2----wake   1----light sleep   0----deep sleep
    ground_truth = data[:, -1]
    input_data = data[:, 0:-1]

    clf = SVC(C=1, decision_function_shape='ovr', gamma='auto', kernel="rbf")
    # a.一对多法（one-versus-rest, 简称1-v-r SVMs）。
    # 训练时依次把某个类别的样本归为一类, 其他剩余的样本归为另一类，
    # 这样k个类别的样本就构造出了k个SVM。分类时将未知样本分类为具有最大分类函数值的那类。

    clf.fit(input_data, ground_truth)
    prer = clf.predict(input_data)
    print("train result", sum([1 if _ == 0 else 0 for _ in (prer - ground_truth)]) / len(input_data))

    validate(test_data)
    print("test data result(whole night):\n")
    test_res0, corr0 = validate(test_data0)
    test_res1, corr1 = validate(test_data1)
    test_res2, corr2 = validate(test_data2)
    test_res3, corr3 = validate(test_data3)
    analysis.compare_truth(test_data0, test_res0)
    analysis.compare_truth(test_data0, corr0)

    analysis.compare_truth(test_data1, test_res1)
    analysis.compare_truth(test_data1, corr1)

    analysis.compare_truth(test_data2, test_res2)
    analysis.compare_truth(test_data2, corr2)

    analysis.compare_truth(test_data3, test_res3)
    analysis.compare_truth(test_data3, corr3)

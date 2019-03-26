import numpy as np
from sklearn.svm import SVC

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


def validate(path):
    test_case = np.load(path)
    ground_truth_test = test_case[:, -1]
    test_case = test_case[:, 0:-1]
    test_prer = clf.predict(test_case)
    print(path, sum([1 if _ == 0 else 0 for _ in (test_prer - ground_truth_test)]) / len(test_case))


if __name__ == '__main__':
    data = np.load('feature/20190215.npy')
    data = np.r_[data, np.load('feature/20190214.npy')]
    data = np.r_[data, np.load('feature/20190213.npy')]
    data = np.r_[data, np.load('feature/20190212.npy')]
    data = np.r_[data, np.load('feature/20190211.npy')]
    data = np.r_[data, np.load('feature/20190210.npy')]
    data = np.r_[data, np.load('feature/20190209.npy')]
    data = np.r_[data, np.load('feature/20190207.npy')]
    data = np.r_[data, np.load('feature/20190204.npy')]
    data = np.r_[data, np.load('feature/20190203.npy')]
    data = np.r_[data, np.load('feature/20190131.npy')]
    data = np.r_[data, np.load('feature/20190130.npy')]
    # data = np.r_[data, np.load('feature/20190128.npy')]

    # 2----wake   1----light sleep   0----deep sleep
    ground_truth = data[:, -1]
    input_data = data[:, 0:-1]

    clf = SVC(decision_function_shape='ovr', gamma='auto', kernel="rbf")
    # a.一对多法（one-versus-rest, 简称1-v-r SVMs）。
    # 训练时依次把某个类别的样本归为一类, 其他剩余的样本归为另一类，
    # 这样k个类别的样本就构造出了k个SVM。分类时将未知样本分类为具有最大分类函数值的那类。

    clf.fit(input_data, ground_truth)
    prer = clf.predict(input_data)
    print(sum([1 if _== 0 else 0 for _ in (prer-ground_truth)])/len(input_data))

    validate('feature/20190128.npy')
    validate('feature/20190130.npy')
    validate('feature/20190131.npy')
    validate('feature/20190203.npy')
    validate('feature/20190204.npy')
    validate('feature/20190207.npy')
    validate('feature/20190209.npy')
import pandas as pd
from ai.util import Storage
import numpy as np
from ai.perceptron import Perceptron
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ai.adaline import Adaline
class Iris:
    iris : object
    def __init__(self):
        self.storage = Storage()
        self.neural_network = None

    @property
    def iris(self) -> object: return self._iris
    @iris.setter
    def iris(self, iris): self._iris = iris

    def initialize(self):
        self.iris = pd.read_csv('https://archive.ics.uci.edu/ml/'
                        'machine-learning-databases/iris/iris.data', header=None)
        print(self.iris.head())
        print(self.iris.columns)
        """
        setosa, versicolor, virginica의 세가지 붓꽃 종(species)
        feature
        꽃받침 길이(Sepal Length)
        꽃받침 폭(Sepal Width)
        꽃잎 길이(Petal Length)
        꽃잎 폭(Petal Width)
        """
        # setosa 와 versicolor 선택
        temp = self.iris.iloc[0:100, 4].values
        this = self.storage
        this.train_Y = np.where(temp == 'Iris-setosa', -1, 1)
        # 꽃받침 길이와 꽃잎 폭 선택
        this.train_X = self.iris.iloc[0:100, [0,2]].values
        self.neural_network = Perceptron(eta=0.1, n_iter=10)

    def show_scatter(self):
        this = self.storage
        X = this.train_X
        plt.scatter(X[:50, 0], X[:50, 1],
                    color='red', marker='o', label='setosa')
        plt.scatter(X[50:100, 0], X[50:100, 1],
                    color='blue', marker='x', label='versicolor')
        plt.xlabel('Sepal Length')
        plt.ylabel('Petal Length')
        plt.legend(loc='upper left')
        plt.show()


    def show_errors(self):
        this = self.storage
        X = this.train_X
        y = this.train_Y
        self.neural_network.fit(X, y)
        plt.plot(range(1, len(self.neural_network.errors_) + 1),
                 self.neural_network.errors_, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Number of Errors')
        plt.show()

    def show_decision_tree(self):
        this = self.storage
        X = this.train_X
        y = this.train_Y
        nn = self.neural_network
        nn.fit(X, y)
        colors = ('red', 'blue', 'rightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        x1_min, x1_max = X[:, 0].min() - 1,X[:, 0].max() + 1,
        x2_min, x2_max = X[:, 1].min() - 1,X[:, 1].max() + 1,
        resolution = 0.2
        """
        numpy 모듈의 arange 함수는 반열린 구간 [start, stop] 에서
        step 의 크기만큼 일정하게 떨어져 있는 숫자들을 array 형태로 반환해 주는 함수
        
        meshgrid 명령은 사각형 영역을 구성하는 
        가로축의 점들과 세로축의 점을 나타내는 두 벡터를 인수로 받아서
        이 사각형 영역을 이루는 조합을 출력한다
        결과는 그리드 포인트의 x 값만을 표시하는 행렬과
        y값 만을 표시하는 행렬 두 개로 불리하여 출력한다.
        """
        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution),
        )
        Z = nn.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0],
                        y=X[y == cl, 1],
                        alpha=0.8,
                        c=colors[idx], label=cl, edgecolors='black')

        plt.xlabel('Sepal Length')
        plt.ylabel('Petal Length')
        plt.legend(loc='upper left')
        plt.show()

    def show_adaline(self):
        this = self.storage
        X = this.train_X
        y = this.train_Y
        X_std = np.copy(X)
        X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
        X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
        self.neural_network = Adaline(eta=0.01, n_iter= 50, random_state=1)
        self.neural_network.fit(X_std, y)
        self.show_decision_tree()








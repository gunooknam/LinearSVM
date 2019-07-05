import argparse

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
from sklearn.datasets.samples_generator import make_blobs

from model import LinearSVM

def train(X, Y, model, args):
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    N = len(Y)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(args.epoch):
        perm = torch.randperm(N) #
        sum_loss = 0

        for i in range(0, N, args.batchsize):
            # 0에서 N까지 args.batchsize 간격만큼
            x = X[perm[i : i+args.batchsize]]
            y = Y[perm[i : i+args.batchsize]]
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            optimizer.zero_grad()
            output = model(x)
            # output = ( x*w ) 이것과 y의  값이 최대가 되야 한다.
            # output.t()*y 값이 너무 커서 음수가 되면 0으로 clamp 된다.
            # 작으면 이게 loss이다.
            loss = torch.mean(torch.clamp(1 - output.t()*y, min=0)) # hinge loss
            loss += args.c * torch.mean(model.fc.weight ** 2) # l2 penalty  + 1/N 시그마 w^2 args.c는 상수값이다. -> 이것이 크면 패널티가 크다.
            loss.backward()
            optimizer.step()

            sum_loss += loss.data.cpu().numpy()

        print("Epoch:{:4d}\tloss:{}".format(epoch, sum_loss / N))

def visualize(X, Y, model):
    print(model.fc.weight)          # model.fc.weight는 1,2 행렬 => [ [0.5520, -1.4039] ]
    W = model.fc.weight[0].data.cpu().numpy()
    b = model.fc.bias[0].data.cpu().numpy()
    # numpy 형식으로 visualize

    delta = 0.01
    x = np.arange(X[:, 0].min(), X[:,0].max(), delta) # x1 축 0.01간격으로
    y = np.arange(X[:, 1].min(), X[:,1].max(), delta) # x2 축 0.01간격으로
    '''
    xx, yy = meshgrid(x,y)
    y 7
      6              >>   1 7   2 7  3 7  4 7      1 2 3 4             7 7 7 7
      5                   1 6   2 6  3 6  4 6 >>   1 2 3 4             6 6 6 6
         1 2 3 4          1 5   2 5  3 5  4 5      1 2 3 4             5 5 5 5
            x                                         xx                 yy
    '''
    x, y = np.meshgrid(x, y)
    xy = list(map(np.ravel, [x, y]))

    z = (W.dot(xy) + b).reshape(x.shape)
    z[np.where(z > 1.)] = 4 # 색깔 설정
    z[np.where((z > 0.) & (z <= 1.))] = 3
    z[np.where((z > -1.) & (z <= 0.))] = 2
    z[np.where(z <= -1.)] = 1

    plt.figure(figsize=(10, 10))
    plt.xlim([X[:, 0].min() + delta, X[:, 0].max() - delta])
    plt.ylim([X[:, 1].min() + delta, X[:, 1].max() - delta])
    plt.contourf(x, y, z, alpha=0.8, cmap="Greys")
    plt.scatter(x=X[:, 0], y=X[:, 1], c="black", s=10)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--constant", type=float, default= 0)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size',type=int, default=1)
    parser.add_argument('--epoch', type=int, default=10)
    args = parser.parse_args()

    X, Y = make_blobs(n_samples=500,
                      n_features=2,
                      centers=2,
                      random_state=0,
                      cluster_std=0.4)
    print(X.shape, Y.shape)  # (100, 500) (100,) 이다.

    X = (X - X.mean()) / X.std()  # normalize

    Y[np.where(Y == 0)] = -1  # 클래스는 0과 1 두개이고 0인 클래스를 -1로 하겠당

    model = LinearSVM()  # 모델 불러오고
    if torch.cuda.is_available():
        model.cuda()

    train(X, Y, model, args)
    visualize(X, Y, model)
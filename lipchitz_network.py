import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Sequential, ReLU
from torch.optim import SGD


class OneLipschitzModule(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        # self.q = Parameter(torch.ones(in_features))
        self.q = Parameter(torch.ones(in_features, dtype=dtype))

    def forward(self, input: Tensor) -> Tensor:
        entries = torch.sqrt(((torch.abs(self.weight.T @ self.weight)) * self.q).sum(axis=0) / self.q)

        T = torch.diag(1 / entries)

        W = self.weight @ T

        return F.linear(input, W, self.bias)


if __name__ == '__main__':
    torch.manual_seed(42)

    X = torch.arange(-5, 5, 0.1).view(-1, 1)
    func = 1 * X
    Y = func + 0.4 * torch.randn(X.size())

    module1 = OneLipschitzModule(1, 32)
    module2 = OneLipschitzModule(32, 64)
    module3 = OneLipschitzModule(64, 1)

    net = Sequential(module1, ReLU(), module2, ReLU(), module3)

    criterion = nn.MSELoss()
    optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(1000):  # loop over the dataset multiple times

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        # print statistics
        if epoch >= 0:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {epoch + 1:5d}] loss: {loss.item():.3f}')

    print('Finished Training')

    with torch.no_grad():

        plt.scatter(X, Y, c='r')

        X = torch.arange(-10, 10, 0.1).view(-1, 1)
        y_pred = net(X)

        plt.plot(X, y_pred, c='b')

        plt.show()

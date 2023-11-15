from typing import Iterable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Sequential, ReLU
from torch.optim import Adam


class SequentialLipschitz(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1, hidden_sizes: Iterable[int] = None,
                 activation: torch.nn.Module = ReLU, scalable_lipschitz=False, device=None, dtype=None):
        super().__init__()
        self.output_size = output_size
        self.activation = activation
        self.hidden_layers = hidden_sizes
        self.input_size = input_size

        net = []

        self.scaling_params = []

        for i, hidden in enumerate(hidden_sizes):

            if scalable_lipschitz:
                name = f"lipschitz_scale_{i}"

                scaling = Parameter(torch.ones(1))

                self.scaling_params.append(scaling)

                self.register_parameter(name, scaling)
            else:
                scaling = None

            net.append(OneLipschitzModule(input_size, hidden, device=device, dtype=dtype, scaling=scaling))
            net.append(activation())

            input_size = hidden

        if scalable_lipschitz:
            name = f"lipschitz_scale_output"

            scaling = Parameter(torch.ones(1))
            self.scaling_params.append(scaling)

            self.register_parameter(name, scaling)
        else:
            scaling = None

        net.append(OneLipschitzModule(input_size, output_size, device=device, dtype=dtype, scaling=scaling))

        self.net = Sequential(*net)

    def get_lipschitz(self):
        if len(self.scaling_params) == 0:
            return 1

        return torch.prod(torch.abs(torch.stack(self.scaling_params)))

    def forward(self, x):
        return self.net(x)


class OneLipschitzModule(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, scaling=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        if scaling is not None:
            self.scaling = scaling
        else:
            self.scaling = torch.tensor(1., dtype=dtype, device=device)

        self.q = Parameter(torch.ones(in_features, dtype=dtype, device=device))

    def forward(self, input: Tensor) -> Tensor:
        entries = torch.sqrt(((torch.abs(self.weight.T @ self.weight)) * self.q).sum(axis=0) / self.q)

        T = torch.diag(1 / entries)

        W = self.weight @ T

        return F.linear(input, W, self.bias) * torch.abs(self.scaling)


if __name__ == '__main__':
    torch.manual_seed(42)

    X = torch.arange(-5, 5, 0.1).view(-1, 1)
    func = 2 * X
    Y = func + 0.4 * torch.randn(X.size())

    module1 = OneLipschitzModule(1, 32)
    module2 = OneLipschitzModule(32, 64)
    module3 = OneLipschitzModule(64, 1)

    # net = Sequential(module1, ReLU(), module2, ReLU(), module3)
    net = SequentialLipschitz(1, 1, [32, 64], scalable_lipschitz=True)

    criterion = nn.MSELoss()
    optimizer = Adam(net.parameters(), lr=0.01)

    for epoch in range(1000):  # loop over the dataset multiple times

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(X)
        loss = criterion(outputs, Y) + net.get_lipschitz()
        loss.backward()
        optimizer.step()

        print(net.get_lipschitz())

        # print statistics
        if epoch >= 0:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {epoch + 1:5d}] loss: {loss.item():.3f}')

    print('Finished Training')

    with torch.no_grad():

        plt.scatter(X, Y, c='r')

        dt = 0.01
        X = torch.arange(-10, 10, dt).view(-1, 1)
        y_pred = net(X)

        y_pred = torch.clamp_min(y_pred, 0)

        L = torch.max(torch.abs((X[1:] - X[:-1]) / dt))
        print(L)

        plt.plot(X, y_pred, c='b')

        plt.show()

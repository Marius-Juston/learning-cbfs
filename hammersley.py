import matplotlib.pyplot as plt
import numpy as np
from skopt.sampler import Hammersly

if __name__ == '__main__':
    b = Hammersly()

    a = [(-10., 10.), (-5., 5.)]

    samples = b.generate(a, 1000)
    samples = np.array(samples)

    plt.scatter(samples[:, 0], samples[:, 1])
    plt.show()

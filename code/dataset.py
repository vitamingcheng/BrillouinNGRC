"""
Dataset:

All data is 2D-array, even it is a one-dimension question.

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class Dataset(object):
    def __init__(self):
        self.data = None
        self.dimension = None
        self.length = None
        self.name = None


class Test1D(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "SantaFe"
        self.get_data()

    def get_data(self):
        self.data = np.arange(10000).reshape(1, -1, order='F')
        self.dimension, self.length = self.data.shape


class Test3D(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "Lorenz63"
        self.get_data()

    def get_data(self):
        self.data = np.array([np.arange(10000), np.arange(10000), np.arange(10000)])
        self.dimension, self.length = self.data.shape


class SantaFe(Dataset):
    """
    SantaFe Laser Time Series
    """
    def __init__(self):
        super().__init__()
        self.name = "SantaFe"
        self.get_data()

    def get_data(self):
        self.data = np.loadtxt("./dataset/SantaFe.txt").reshape(1, -1)
        self.dimension, self.length = self.data.shape


class Equalisation(Dataset):
    """
    Channel Equalisation Task Dataset
    input:
    -> length_each_symbol:
    -> is_normed:

    Note: The first 7 and last 2 symbol is useless
    """
    def __init__(self, length_each_symbol, snr):
        super().__init__()
        self.dimension = 1
        self.length = 4 * length_each_symbol
        self.symbol = None
        self.snr = snr
        self.name = "Equalisation"
        self.get_symbol()
        self.get_data()

    def get_symbol(self):
        a = np.ones(int(self.length / 4)) * (-3)
        b = np.ones(int(self.length / 4)) * (-1)
        c = np.ones(int(self.length / 4)) * 1
        d = np.ones(int(self.length / 4)) * 3
        e = np.concatenate([a, b, c, d])
        np.random.shuffle(e)
        self.symbol = e.reshape(1, -1)

    def get_data(self):
        s = self.symbol
        q = np.zeros_like(s)
        q[:, 7:-2] = 0.08 * s[:, 9:] - 0.12 * s[:, 8:-1] + s[:, 7:-2] + 0.18 * s[:, 6:-3] - 0.1 * s[:, 5:-4] + \
            0.091 * s[:, 4:-5] - 0.05 * s[:, 3:-6] + 0.04 * s[:, 2:-7] + 0.03 * s[:, 1:-8] + 0.01 * s[:, :-9]
        self.data = q + 0.036 * q ** 2 - 0.011 * q ** 3 + \
            np.exp(-self.snr) * np.random.normal(loc=0, scale=1, size=np.shape(q))


class Lorenz63(Dataset):
    def __init__(self, dt, warmup, train, infer):
        super().__init__()
        self.name = "Lorenz63"
        self.dt = dt
        self.warmup = warmup
        self.train = train
        self.infer = infer
        self.get_data()

    @staticmethod
    def lorenz63(sigma=10, rho=28, beta=8/3):
        def f(t, y):
            dy0 = sigma * (y[1] - y[0])
            dy1 = y[0] * (rho - y[2]) - y[1]
            dy2 = y[0] * y[1] - beta * y[2]

            return [dy0, dy1, dy2]
        return f

    @staticmethod
    def ode_euler(f, y0, t):
        dimension = len(y0)
        length = len(t)
        y = np.zeros([dimension, length])
        y[:, 0] = y0
        for i in range(length - 1):
            for j in range(dimension):
                y[j, i + 1] = y[j, i] + f(t[i], y[:, i])[j] * (t[i + 1] - t[i])
        return y

    def get_data(self):
        dt = self.dt
        warmup = self.warmup
        train = self.train
        infer = self.infer
        total = warmup + train + infer
        warmup_pts = round(warmup / dt)
        train_pts = round(train / dt)
        infer_pts = round(infer / dt)
        total_pts = warmup_pts + train_pts + infer_pts
        t_eval = np.linspace(0, total, total_pts + 1)

        y0 = [17.67715816276679, 12.931379185960404, 43.91404334248268]
        self.data = solve_ivp(self.lorenz63(), (0, total), y0, t_eval=t_eval, method="RK23").y
        self.dimension, self.length = self.data.shape


if __name__ == "__main__":
    eq = Equalisation(100, 30)
    sf = SantaFe()
    lorenz = Lorenz63(0.025, 5, 10, 120)
    t3 = Test3D()
    print(t3.dimension)
    print(t3.length)
















































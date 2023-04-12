from dataset import SantaFe, Equalisation, Lorenz63, Test1D, Test3D
from ngrc import NGRCSolver

theDataSet = SantaFe()
settings = {
    "normed": False,
    "time_lag": 2,
    "start": 1000,
    "length": 4000,
    "ratio": 0.5,
    "ridge_param": 1e-6,
    "wout_solver": "new1"
}

s1 = NGRCSolver(Lorenz63(0.025, 5, 10, 120), settings)
s1.run()












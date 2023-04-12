from dataset import Equalisation
from ngrc import NGRCSolver

theDataSet = Equalisation(25000, 10)
settings = {
    "normed": False,
    "time_lag": 9,
    "start": 1000,
    "length": 80000,
    "ratio": 0.5,
    "ridge_param": 1e-10,
    "wout_solver": "new1"
}

s1 = NGRCSolver(Equalisation(25000, 30), settings)
s1.run()


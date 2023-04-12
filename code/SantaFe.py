from dataset import SantaFe, Equalisation, Lorenz63
from ngrc import NGRCSolver
"""
0.039326255691157844
"""
theDataSet = SantaFe()
settings = {
    "normed": True,
    "time_lag": 8,
    "start": 1000,
    "length": 4000,
    "ratio": 0.8,
    "ridge_param": 1e-6,
    "wout_solver": "old2"
}

s1 = NGRCSolver(SantaFe(), settings)
s1.run()
# s2 = NGRCSolver(Equalisation(2500, 30), settings)
# s2.print_information()
# s3 = NGRCSolver(Lorenz63(0.025, 5, 10, 120), settings)
# s3.print_information()

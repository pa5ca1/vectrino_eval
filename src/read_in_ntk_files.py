from __future__ import print_function, division
from nortek.files import VectrinoFile
from nortek.controls import PdControl
import matplotlib.pyplot as plt

def test_vectrino_file():
    vec = VectrinoFile("../sample_data/VectrinoData.144.23.Vectrino Profiler.00006.ntk")
    print(vec.keys())
    u = vec["velocity"]["data"][0,:,0]
    print(vec["velocityHeader"])
    print(vec["hardwareConfiguration"])
    plt.plot(u)
    plt.show()

test_vectrino_file()
import numpy as np
import os
from utils.datasets import *
from models import *

def saveParams( path, model, fName="weights.dat" ):
    if not os.path.exists(path):
        os.makedirs(path)
    params = np.empty(0)
    Dict = model.state_dict()
    for name in Dict:
        param = Dict[name].numpy()
        param = param.reshape(param.size)
        params = np.concatenate((params, param))
    params.tofile(path+"/"+fName)

if __name__ == "__main__":

    path = "checkpoints/bestFinetune88_60.weights"

    model = ROBO()
    model.load_state_dict(torch.load(path))

    saveParams("checkpoints/",model)
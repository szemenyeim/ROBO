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
        if "num_batches" in name:
            continue
        param = param.reshape(param.size)
        params = np.concatenate((params, param))
    params.tofile(path+"/"+fName)

if __name__ == "__main__":

    path = "checkpoints/bestFinetuneHR93_32.weights"

    model = ROBO(bn=False,inch=3,halfRes=True)
    model.load_state_dict(torch.load(path))

    saveParams("checkpoints/",model,fName="weightsHR93.dat")
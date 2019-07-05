import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
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

    path = "checkpoints/bestFinetuneHR93_31.weights"

    model = ROBO(bn=False,inch=3,halfRes=True)
    model.load_state_dict(torch.load(path, map_location={'cuda:0': 'cpu'}))

    saveParams("checkpoints/",model,fName="weightsHR.dat")

    path = "checkpoints/bestFinetune2C94_38.weights"

    model = ROBO(bn=False,inch=2,halfRes=False)
    model.load_state_dict(torch.load(path, map_location={'cuda:0': 'cpu'}))

    saveParams("checkpoints/",model,fName="weights2C.dat")

    path = "checkpoints/bestFinetuneBN97_78.weights"

    model = ROBO(bn=True,inch=3,halfRes=False)
    model.load_state_dict(torch.load(path, map_location={'cuda:0': 'cpu'}))

    saveParams("checkpoints/",model,fName="weightsBN.dat")

    path = "checkpoints/bestFinetune93_39.weights"

    model = ROBO(bn=False,inch=3,halfRes=False)
    model.load_state_dict(torch.load(path, map_location={'cuda:0': 'cpu'}))

    saveParams("checkpoints/",model,fName="weights.dat")
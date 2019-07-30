import os
import time
import random
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split


def getData(path, trainDataRatio, random_seed=123):
    data = pd.read_csv(path, header=None)
    numData = data.shape[1]
    data.rename(columns={numData-1:'is_malicious'}, inplace=True)
    trainData, testData = train_test_split(data, train_size=trainDataRatio, random_state=random_seed)
    return (trainData, testData)
    

def project_operator(Theta):
    projected_Theta = np.copy(Theta)
    projected_Theta[np.where(projected_Theta <= 0.0000000001)[0]] = 0
    projected_Theta[np.where(projected_Theta >= 0.9999999999)[0]] = 1
    return projected_Theta


# write progress info to disk
def output_log(path, message):
    if os.path.exists(path):
        with open(path, 'a+') as fid:
            fid.write(time.strftime("%H:%M:%S") + " " + message)
    else:
        with open(path, 'w') as fid:
            fid.write(time.strftime("%H:%M:%S") + " " + message)

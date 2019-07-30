import os
import sys
import time
import pickle
import operator
import argparse
from model import *
import pandas as pd
import multiprocessing
import numpy.linalg as lin
from multiprocessing import Pool
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegressionCV

RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


## Instead of selecting malicious nodes uniformly at random
## we use other greedy method to choose malicious nodes
def selectMaliciousNodes(G, numMaliciousNodes):
    degree_sequence = dict(sorted([(n, d) for n, d in G.degree().items()], key=lambda x: x[1], reverse=True))
    selectedNodes = []
    for n in range(numMaliciousNodes):
        selected_n = max(degree_sequence.iteritems(), key=operator.itemgetter(1))[0]
        degree_sequence.pop(selected_n)
        selectedNodes.append(selected_n)
        for neighbor in G.neighbors(selected_n):
            if neighbor in degree_sequence:
                degree_sequence[neighbor] -= 1
    return selectedNodes


### This function is used to simulate signal on networks, which is then
### used to estimate maliciousness probabilites so as to make decisions 
### about which subset of nodes to remove
def simulateSignalInGraph(G, malicious_ratio, dataPath, trainSize, std_):
    standard_deviation = std_
    n = len(G.nodes())
    numMaliciousNodes = np.int(np.floor(n * malicious_ratio))

    ## here trainData is used to train a global classifier
    ## to generate the maliciousness probabilities
    trainData, testData = getData(dataPath, trainSize)

    ## we divided testData into two parts: the first is for 
    ## selecting a subset of malicious nodes, and the other is
    ## for testing the performance of this subset
    testData_selectSubset, testData_eval = train_test_split(testData, train_size=0.8)

    ## A normally trained classifier is always needed
    clf = LogisticRegressionCV(cv=10, penalty='l2')
    clf.fit(trainData.iloc[:, :-1], trainData.iloc[:, -1])

    clf_eval = LogisticRegressionCV(cv=10, penalty='l2')
    clf_eval.fit(pd.concat([trainData, testData_selectSubset]).iloc[:, :-1], pd.concat([trainData, testData_selectSubset]).iloc[:, -1])

    malicious_nodes = list(np.random.choice(G.nodes(), numMaliciousNodes, replace=False))
    # malicious_nodes = selectMaliciousNodes(G, numMaliciousNodes)

    ## the number of malicious emails needed is equal to
    ## the number of malicious nodes
    malicious_data = testData_selectSubset[testData_selectSubset['is_malicious'] == 1].sample(numMaliciousNodes)
    malicious_data = malicious_data.iloc[:, :-1]
    malicious_data.index = malicious_nodes

    malicious_data_evaluation = testData_eval[testData_eval['is_malicious'] == 1].sample(numMaliciousNodes)
    malicious_data_evaluation = malicious_data_evaluation.iloc[:, :-1]
    malicious_data_evaluation.index = malicious_nodes

    for node in malicious_nodes:
        signals = clf.predict_proba(malicious_data.loc[node].values.reshape(1, -1)).squeeze()[1]
        G.node[node]['detectionSignals'] = signals
        evaluationSignals = clf_eval.predict_proba(malicious_data.loc[node].values.reshape(1, -1)).squeeze()[1]
        noise = np.random.normal(scale=standard_deviation)
        if evaluationSignals + noise > 0.9999:
            evaluationSignals = 0.9999
        elif evaluationSignals + noise < 0.0001:
            evaluationSignals = 0.0001
        else:
            evaluationSignals += noise
        G.node[node]['evaluationSignals'] = evaluationSignals

    benign_nodes = list(set(G.nodes()) - set(malicious_nodes))
    numBenignNodes = len(benign_nodes)
    benign_data = testData_selectSubset[testData_selectSubset['is_malicious'] == 0].sample(numBenignNodes, replace=True)
    benign_data = benign_data.iloc[:, :-1]
    benign_data.index = benign_nodes

    benign_data_evaluation = testData_eval[testData_eval['is_malicious'] == 0].sample(numBenignNodes, replace=True)
    benign_data_evaluation = benign_data_evaluation.iloc[:, :-1]
    benign_data_evaluation.index = benign_nodes

    for node in benign_nodes:
        signals = clf.predict_proba(benign_data.loc[node].values.reshape(1, -1)).squeeze()[1]
        G.node[node]['detectionSignals'] = signals
        evaluationSignals = clf_eval.predict_proba(benign_data.loc[node].values.reshape(1, -1)).squeeze()[1] 
        noise = np.random.normal(scale=standard_deviation)
        if evaluationSignals + noise > 0.9999:
            evaluationSignals = 0.9999
        elif evaluationSignals + noise < 0.0001:
            evaluationSignals = 0.0001
        else:
            evaluationSignals += noise
        G.node[node]['evaluationSignals'] = evaluationSignals

    return (G, malicious_nodes, benign_nodes)



## Auxiliary functions to do parallel computing ###
###################################################################################



def dispatchFunc_smart(arg_list):
    std_ = 0
    G, malicious_ratio, dataPath, trainSize, optTH, rho, weightFactors, Algo = arg_list
    G, malicious_nodes, benign_nodes = simulateSignalInGraph(G, malicious_ratio, dataPath, trainSize, std_)
    allNodes = G.nodes()
    averageDetectionSignals = [np.mean(G.node[i]['detectionSignals']) for i in allNodes]
    if Algo.__name__ == 'LESS':
        our_ret = Algo(averageDetectionSignals, G, rho, weightFactors)
    elif Algo.__name__ == 'MINT':
        our_ret = Algo(averageDetectionSignals, G, weightFactors)
    elif Algo.__name__ == 'simple_baseline':
        our_ret = Algo(averageDetectionSignals, G, optTH, weightFactors)
    else:
        raise ValueError("Unknown algorithm\n")
    return our_ret 


def generateGraphData(numExp, graph_type, batch, dataPath, trainSize):
    n = 128
    ret = []
    if graph_type == 'Facebook':
        G0 = nx.read_edgelist('data/facebook_combined.txt', nodetype=int)

    for ii in range(numExp):
        if graph_type == 'BA':
            if batch == 1:
                G = nx.barabasi_albert_graph(n, 3)    # 1: exponent =  2.7167
            elif batch == 2:
                G = nx.barabasi_albert_graph(n, 4)    # 2: exponent =  2.2789
            elif batch == 3:
                G = nx.barabasi_albert_graph(n, 5)    # 3: exponent =  2.0374
            else:
                raise ValueError("Unknown batch number!\n")
        elif graph_type == 'Small-World':
            if batch == 1:
                G = nx.watts_strogatz_graph(n, 10, 0.2)   # 1 local clustering = 0.3667
            elif batch == 2:
                G = nx.watts_strogatz_graph(n, 15, 0.2)     # 2 local clustering = 0.3893
            elif batch == 3:
                G = nx.watts_strogatz_graph(n, 20, 0.2)   # 3 local clustering = 0.4070
            else:
                raise ValueError("Unknown batch number!\n")
        elif graph_type == 'Facebook':
            G = G0.subgraph(np.random.choice(G0.nodes(), 500, replace=False))
            mapping = {item: idx for idx, item in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
        else:
            raise ValueError("Unknown graph type!\n")  
        ret.append((G, malicious_ratio, dataPath, trainSize))

    return ret

###################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_type', type=str)
    parser.add_argument('batch', type=int)
    parser.add_argument('numExp', type=int)
    args = parser.parse_args()
    graph_type, batch, numExp = args.graph_type, args.batch, args.numExp

    numCPU = multiprocessing.cpu_count()
    malicious_ratio = 0.1
    trainSize = 0.3
    dataPath = 'data/spambase.data' if graph_type in ['BA', 'Small-World'] else 'data/hate_speech.csv'
    rho = 15
    FPR_weight = 0.5
    optTH = selectThreshold(FPR_weight, dataPath)
    
    weightFactor_dict = {
            '1_2_7': [0.1, 0.2, 0.7],
            '2_7_1': [0.2, 0.7, 0.1],
            '7_2_1': [0.7, 0.2, 0.1],
            'equalAlpha': [1/3, 1/3, 1/3]
            }

    ### Generate graph data ###
    ###################################################################################
    graph_args = generateGraphData(numExp, graph_type, batch,  dataPath, trainSize)

    ###################################################################################
    for key, value in weightFactor_dict.items():
        ret = {'MINT': [], 'baseline': [], 'LESS': []}
        pool = Pool(processes=numCPU-1)
        all_args = []

        for ii in range(numExp):
            G, malicious_ratio, dataPath, trainSize = graph_args[ii]
            all_args.append([G, malicious_ratio, dataPath, trainSize, optTH, rho, value, MINT])
      
        ###################################################################################
        ## our SDP relaxation (MINT) ###
        MINT_ret = pool.map(dispatchFunc_smart, all_args)
        ret['MINT'] = [item[1] for item in MINT_ret]

        ###################################################################################
        ### simple baseline ###
        for ii in range(numExp):
            all_args[ii][-1] = simple_baseline
        baseline_ret = pool.map(dispatchFunc_smart, all_args)
        ret['baseline'] = [item[1] for item in baseline_ret]

        ###################################################################################
        ## LESS ###
        if graph_type != 'Facebook':
           for ii in range(numExp):
               all_args[ii][-1] = LESS
           LESS_ret = pool.map(dispatchFunc_smart, all_args)
           ret['LESS'] = [item[1] for item in LESS_ret]

        fidName = os.path.join('result', '%s_boxplot_data_%s_%i.p' % (graph_type, key, batch))
        with open(fidName, 'wb') as fid:
            pickle.dump(ret, fid)

        ###################################################################################
        pool.close()
        pool.join()



          

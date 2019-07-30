import pickle
import numpy as np
from utils import *
import cvxpy as cvx
import networkx as nx
import numpy.linalg as lin
import scipy.linalg as scilin
from sklearn.linear_model import LogisticRegressionCV



##########################################################################################


def cal_removal_loss(subset, G, weights):
    n = len(G)
    remainNodes = list(set(G.nodes()) - set(subset))
    evaluationSignals = [np.mean(G.node[idx]['evaluationSignals']) for idx in range(n)]

    removal_loss = 0
    totalLoss = [0, 0, 0]

    ## loss for the first term (false-positives)
    for n in subset:
        totalLoss[0] += (1 - evaluationSignals[n])

    ## loss for the third term (false-negatives)
    for n in remainNodes:
        totalLoss[2] += evaluationSignals[n]

    ## loss for the second term
    for n in subset:
        neighborhood = G.neighbors(n) 
        for neigh in neighborhood:
            if neigh not in remainNodes:
                continue
            totalLoss[1] += (1 - evaluationSignals[n]) * (1 - evaluationSignals[neigh])

    removal_loss = np.sum([totalLoss[idx] * weights[idx] for idx in range(len(totalLoss))])
    return removal_loss


##########################################################################################


def rounding(X, augmented=True):
    # Cholesky decomposition is for positive definite matrices
    # So we use LDL decomposition here
    m = X.shape[0]
    epsilon = 1e-8
    lu, d, perm = scilin.ldl(X)
    d[d < epsilon] = 0
    assert(np.all(d >= 0))
    d_sqroot = np.diag(np.sqrt(np.diag(d)))
    V = d_sqroot.dot(lu.T)

    # sample  standard gaussian vectors
    numSample = 10000
    x = 0
    for ii in range(numSample):
        z = np.random.multivariate_normal(np.zeros(m), np.identity(m))
        x += np.sign(V.T.dot(z))
    x /= numSample
    x = np.sign(x)

    if augmented:
        if x[-1] == -1:
            x_opt = -1 * x[:-1]
        else:
            x_opt = x[:-1]
        return x_opt
    else:
        x_opt = x
        return x_opt



##########################################################################################

def MINT(signal, G, weightFactors):
    n = len(G.nodes())
    all_one = np.asmatrix(np.ones(n)).reshape(n, 1)
    A = nx.adjacency_matrix(G).todense()

    B = np.asmatrix(np.diag(1 - np.asarray(signal)))
    m = np.asmatrix(signal).reshape(n, 1)
    estimated_cov = np.asmatrix(np.identity(n) * 0.01)
    P = np.asmatrix(np.ones((n, n))) - all_one * m.T - m * all_one.T + estimated_cov + m * m.T
    P = np.multiply(A, P)
    M = m * all_one.T - estimated_cov - m * m.T
    M = np.multiply(A, M)

    alpha_1 = weightFactors[0]
    alpha_2 = weightFactors[1]
    alpha_3 = weightFactors[2]

    ## 
    Q = (alpha_3/2)*(M + M.T) - (alpha_2/2) * (P + P.T)
    b = (alpha_1/2) * B * all_one 

    Q_hat = np.bmat([[Q, b], [b.T, np.matrix(0)]])
    X_opt = cvx.Variable((n+1, n+1), symmetric=True)

    obj = cvx.trace(Q_hat * X_opt)
    constraints = [X_opt >> 0,
                   cvx.reshape(cvx.diag(X_opt), (n+1, 1)) == 1]
    prob = cvx.Problem(cvx.Minimize(obj), constraints)
    prob.solve(solver=cvx.SCS, verbose=True)

    x_opt = rounding(X_opt.value)
    our_subset = np.where(x_opt > 0.0)[0]
    MINT_loss = cal_removal_loss(our_subset, G, weightFactors)
    print(prob.status)
    return (our_subset, MINT_loss)

##########################################################################################




### LESS algorithm ###
def LESS(signal, G,  rho, weightFactors):

    epsilon = 1e-5
    n = len(G.nodes())
    ## LESS
    current_obj = -np.inf
    current_sol = None
    inci = nx.incidence_matrix(G, oriented=True).T
    for t in range(1, n+1):
        x_opt = cvx.Variable(n)
        y = signal
        v = inci * x_opt
        obj = x_opt.T * y / np.sqrt(t)
        constraints = [x_opt >= 0,
                       x_opt <= 1,
                       cvx.sum(x_opt) <= t,
                       cvx.norm(cvx.pos(v), 1) <= rho]
        prob = cvx.Problem(cvx.Maximize(obj), constraints)
        try:
            prob.solve(solver=cvx.CVXOPT)
        except:
            print("Cannot solve! because: ", prob.status)
            continue

        if obj.value > current_obj:
            current_obj = obj.value
            current_sol = x_opt.value
    
    if current_sol is not None:
        LESS_subset = np.where(current_sol > epsilon)[0].tolist()
        LESS_sol    = (LESS_subset, cal_removal_loss(LESS_subset, G, weightFactors))
    else:
        LESS_sol = (None, rho)
    return LESS_sol


##########################################################################################


###  threshold-based simple baseline ###
def simple_baseline(signal, G, threshold, weightFactors):
    numNodes = len(G.nodes())
    signal = np.asarray(signal).squeeze()
    our_subset = [i for i in range(numNodes) if signal[i] >= threshold] 
    our_sol = (our_subset, cal_removal_loss(our_subset, G, weightFactors))
    return our_sol


##########################################################################################


def selectThreshold(alpha, dataPath):
    trainData, _ = getData(dataPath, 0.5)
    trainData, valData = train_test_split(trainData, train_size=0.7)
    clf = LogisticRegressionCV(cv=10, penalty='l2')
    clf.fit(trainData.iloc[:, :-1], trainData.iloc[:, -1])
    val_score = clf.score(valData.iloc[:, :-1], valData.iloc[:, -1])
    print("Validation accuracy:  %.6f" % val_score)
    ## find optimal threshold on validation data
    y_true = valData.iloc[:, -1]
    y_positive_idx = set(np.where(y_true == 1)[0])
    y_negative_idx = set(np.where(y_true == 0)[0])
    numPositive = len(y_positive_idx)
    numNegative = len(y_negative_idx)
    y_pred = clf.predict_proba(valData.iloc[:, :-1])[:, 1]

    ret = []
    for th in np.linspace(0.1, 0.9, 9):
        FPR = len(set(np.where(y_pred > th)[0]) & y_negative_idx) / numNegative
        FNR = len(set(np.where(y_pred < th)[0]) & y_positive_idx) / numPositive
        cost = alpha * FPR + (1 - alpha) * FNR
        print("threshold: %.2f  cost: %.2f" % (th, cost))
        ret.append((th, cost))
    ## return optimal th
    optTh = min(ret, key=lambda x: x[1])[0]
    return optTh



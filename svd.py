import numpy as np
from math import sqrt
from collaborative import build_corrMatrix
from collaborative import collaborative_basic
from metrics import get_metrics
import time


def build_svd_matrices(a):
    at = np.transpose(a)
    a_at = np.matmul(a, at)
    nb_users = a.shape[0]
    nb_movies = a.shape[1]
    at_a = np.matmul(at, a)
    del a
    del at
    eigval_u, eigvec_u = np.linalg.eigh(a_at)
    eigval_v, eigvec_v = np.linalg.eigh(at_a)
    positive_eig_u = []
    positive_eig_v = []
    for val in eigval_u.tolist():
        if(val > 0):
            positive_eig_u.append(val)
    for val in eigval_v.tolist():
        if(val > 0):
            positive_eig_v.append(val)
    positive_eig_u.reverse()
    positive_eig_v.reverse()
    root_eig_u = [sqrt(val) for val in positive_eig_u]
    root_eig_u = np.array(root_eig_u)
    sigma = np.diag(root_eig_u)
    size_sigma = sigma.shape[0]
    ut = np.zeros(shape = (size_sigma, nb_users))
    vt = np.zeros(shape = (size_sigma, nb_movies))
    i = 0
    for val in positive_eig_u:
        ut[i] = eigvec_u[eigval_u.tolist().index(val)]
        i = i + 1
    i = 0
    for val in positive_eig_v:
        vt[i] = eigvec_v[eigval_v.tolist().index(val)]
        i = i + 1
    u = np.transpose(ut)
    del ut
    return u, vt, sigma
    

def top_90_energy(u, vt, sigma):
    size_sigma = sigma.shape[0]
    total_sum = 0
    required_eigvalues = np.zeros(size_sigma)
    for i in range(size_sigma):
        total_sum += sigma[i][i] * sigma[i][i]
    current_sum = 0
    for i in range(size_sigma):
        current_sum += sigma[i][i] * sigma[i][i]
        required_eigvalues[i] = sigma[i][i]
        if (current_sum/total_sum) >= 0.9:
            i = i + 1
            break
    required_eigvalues = required_eigvalues[required_eigvalues > 0]
    new_sigma = np.diag(required_eigvalues)
    new_u = np.transpose(np.transpose(u)[:new_sigma.shape[0]])
    new_vt = vt[:new_sigma.shape[0]]
    return new_u, new_vt, new_sigma


def main():
    
    K = 50
    trainData = np.load('train.npy')
    testData = np.load('test.npy')
    
    # UNCOMMENT BELOW AND REMOVE FURTHER BELOW FOR FASTER PERFORMANCE ON MULTIPLE RUNS
    '''
    try:
        svd_corrMatrix = np.load('svd_correlation_matrix.npy')
        svd_90_corrMatrix = np.load('svd_90_correlation_matrix.npy')
    except FileNotFoundError:
        u, vt, sigma = build_svd_matrices(trainData)
        new_u, new_vt, new_sigma = top_90_energy(u, vt, sigma)
        users_in_svd_space = np.matmul(u, sigma)
        users_in_svd_90_space = np.matmul(new_u, new_sigma)
        svd_corrMatrix = build_corrMatrix(users_in_svd_space, 'svd_correlation_matrix.npy')
        svd_90_corrMatrix = build_corrMatrix(users_in_svd_90_space, 'svd_90_correlation_matrix.npy')
    '''
    
    t0 = time.clock()
    
    # REMOVE BELOW FOR FASTER PERFORMANCE ON MULTIPLE RUNS, AND UNCOMMENT ABOVE
    u, vt, sigma = build_svd_matrices(trainData)
    users_in_svd_space = np.matmul(u, sigma)
    svd_corrMatrix = build_corrMatrix(users_in_svd_space, 'svd_correlation_matrix.npy')
    
    result_svd = collaborative_basic(trainData, testData, svd_corrMatrix, K)
    RMSE_svd, SRC_svd, precisionTopK_svd = get_metrics(result_svd, testData)
    del result_svd
    del svd_corrMatrix
    t1 = time.clock()
    print('SVD:     RMSE = {}; SRC = {}; Precision on top K = {}; time taken = {}'.format(RMSE_svd, SRC_svd, precisionTopK_svd, t1-t0))

    t2 = time.clock()
    
    # REMOVE BELOW FOR FASTER PERFORMANCE ON MULTIPLE RUNS, AND UNCOMMENT ABOVE
    new_u, new_vt, new_sigma = top_90_energy(u, vt, sigma)
    users_in_svd_90_space = np.matmul(new_u, new_sigma)
    svd_90_corrMatrix = build_corrMatrix(users_in_svd_90_space, 'svd_90_correlation_matrix.npy')
    
    result_svd_90 = collaborative_basic(trainData, testData, svd_90_corrMatrix, K)
    RMSE_svd_90, SRC_svd_90, precisionTopK_svd_90 = get_metrics(result_svd_90, testData)
    del result_svd_90
    del svd_90_corrMatrix
    t3 = time.clock()
    print('SVD 90%: RMSE = {}; SRC = {}; Precision on top K = {}; time taken = {}'.format(RMSE_svd_90, SRC_svd_90, precisionTopK_svd_90, t3-t2))
    

if __name__ == '__main__':
    main()

import numpy as np
import time
from metrics import get_metrics

def build_corrMatrix(data, file_name):
    numUsers = len(data)
    corrMatrix = np.corrcoef(data)[:numUsers+1, :numUsers+1]
    np.save(file_name, corrMatrix)
    return corrMatrix


def collaborative_basic_old(trainData, corrMatrix, K):
    numUsers = len(trainData)
    numItems = len(trainData[0])
    
    # compute reconstructed matrix
    reconstructedTrain = np.zeros((numUsers, numItems))
    for userToPredict in range(numUsers):
        closestKUsers = (-corrMatrix[userToPredict]).argsort()[:K] # most similar K users
        corrSum = 0
        for closeUser in closestKUsers:
            reconstructedTrain[userToPredict] = np.add( reconstructedTrain[userToPredict], (corrMatrix[userToPredict, closeUser] * trainData[closeUser]))
            corrSum += corrMatrix[userToPredict, closeUser]
        reconstructedTrain[userToPredict] /= corrSum
    return reconstructedTrain


def collaborative_basic(trainData, testData, corrMatrix, K):
    numUsers = len(trainData)
    numItems = len(trainData[0])
    
    reconstructedTrain = np.zeros((numUsers, numItems))
    for userToPredict in range(numUsers):
        closestKUsers = (-corrMatrix[userToPredict]).argsort()[:K]
        for item in range(numItems):
            if testData[userToPredict, item] == 0:
                continue
            corrSum = 0
            for closeUser in closestKUsers:
                if trainData[closeUser, item] != 0:
                    reconstructedTrain[userToPredict, item] += corrMatrix[userToPredict, closeUser] * trainData[closeUser, item]
                    corrSum += corrMatrix[userToPredict, closeUser]
            if corrSum != 0:
                reconstructedTrain[userToPredict, item] /= corrSum
    return reconstructedTrain


def collaborative_baseline_old(trainData, corrMatrix, K):
    numUsers = len(trainData)
    numItems = len(trainData[0])
    globalMean = 0
    
    # rating deviation for each user/item
    ratingDeviationUsers, ratingDeviationItems = np.zeros(numUsers), np.zeros(numItems)
    
    # number of ratings per user/item
    numUserRatings, numItemRatings = np.zeros(numUsers), np.zeros(numItems)
    
    for user in range(numUsers):
        for item in range(numItems):
            if trainData[user, item] == 0:
                continue
            else:
                ratingDeviationUsers[user] += trainData[user, item]
                ratingDeviationItems[item] += trainData[user, item]
                globalMean += trainData[user, item]
                numUserRatings[user] += 1
                numItemRatings[item] += 1
    
    # handle cases where a user/item has not rated/been rated (to avoid divide-by-zero)
    for user in range(numUsers):
        if numUserRatings[user] == 0:
            numUserRatings[user] = 1
    for user in range(numItems):
        if numItemRatings[user] == 0:
            numItemRatings[user] = 1
    
    # calculate global mean and rating deviations
    globalMean /= np.sum(numUserRatings)
    ratingDeviationUsers = np.divide(ratingDeviationUsers, numUserRatings) # avg rating of any user
    ratingDeviationUsers -= globalMean # subtract global mean
    ratingDeviationItems = np.divide(ratingDeviationItems, numItemRatings) # avg rating of any item
    ratingDeviationItems -= globalMean # subtract global mean
    
    # calculate baselines for each user,item pair
    baseline = np.zeros((numUsers, numItems))
    for user in range(numUsers):
        for item in range(numItems):
            baseline[user, item] = globalMean + ratingDeviationUsers[user] + ratingDeviationItems[item]
    
    # compute reconstructed matrix
    reconstructedTrain = np.zeros((numUsers, numItems))
    for userToPredict in range(numUsers):
        closestKUsers = (-corrMatrix[userToPredict]).argsort()[:K] # top K users
        corrSum = 0
        for closeUser in closestKUsers:
            temp = np.subtract(trainData[closeUser], baseline[closeUser])
            reconstructedTrain[userToPredict] = np.add( reconstructedTrain[userToPredict], (corrMatrix[userToPredict, closeUser] * temp))
            corrSum += corrMatrix[userToPredict, closeUser]
        reconstructedTrain[userToPredict] /= corrSum
    reconstructedTrain = np.array( np.matrix(reconstructedTrain) + np.matrix(baseline) )
    return reconstructedTrain


def collaborative_baseline(trainData, testData, corrMatrix, K):
    numUsers = len(trainData)
    numItems = len(trainData[0])
    globalMean = 0
    
    # rating deviation for each user/item
    ratingDeviationUsers, ratingDeviationItems = np.zeros(numUsers), np.zeros(numItems)
    
    # number of ratings per user/item
    numUserRatings, numItemRatings = np.zeros(numUsers), np.zeros(numItems)
    
    for user in range(numUsers):
        for item in range(numItems):
            if trainData[user, item] == 0:
                continue
            else:
                ratingDeviationUsers[user] += trainData[user, item]
                ratingDeviationItems[item] += trainData[user, item]
                globalMean += trainData[user, item]
                numUserRatings[user] += 1
                numItemRatings[item] += 1
    
    # handle cases where a user/item has not rated/been rated (to avoid divide-by-zero)
    for user in range(numUsers):
        if numUserRatings[user] == 0:
            numUserRatings[user] = 1
    for item in range(numItems):
        if numItemRatings[item] == 0:
            numItemRatings[item] = 1
    
    # calculate global mean and rating deviations
    globalMean /= np.sum(numUserRatings)
    ratingDeviationUsers = np.divide(ratingDeviationUsers, numUserRatings) # avg rating of any user
    ratingDeviationUsers -= globalMean # subtract global mean
    ratingDeviationItems = np.divide(ratingDeviationItems, numItemRatings) # avg rating of any item
    ratingDeviationItems -= globalMean # subtract global mean
    
    # calculate baselines for each user,item pair
    baseline = np.zeros((numUsers, numItems))
    for user in range(numUsers):
        for item in range(numItems):
            baseline[user, item] = globalMean + ratingDeviationUsers[user] + ratingDeviationItems[item]
    
    # compute reconstructed matrix
    reconstructedTrain = np.zeros((numUsers, numItems))
    for userToPredict in range(numUsers):
        closestKUsers = (-corrMatrix[userToPredict]).argsort()[:K]
        for item in range(numItems):
            if testData[userToPredict, item] == 0:
                continue
            corrSum = 0
            for closeUser in closestKUsers:
                if trainData[closeUser, item] != 0:
                    reconstructedTrain[userToPredict, item] += corrMatrix[userToPredict, closeUser] * (trainData[closeUser, item] - baseline[closeUser, item])
                    corrSum += corrMatrix[userToPredict, closeUser]
            if corrSum != 0:
                reconstructedTrain[userToPredict, item] /= corrSum
                reconstructedTrain[userToPredict, item] += baseline[userToPredict, item]
    return reconstructedTrain


def main():
    K = 50
    trainData = np.load('train.npy')
    testData = np.load('test.npy')
    
    # UNCOMMENT BELOW AND REMOVE FURTHER BELOW FOR FASTER PERFORMANCE ON MULTIPLE RUNS
    '''
    try:
        corrMatrix = np.load('correlation_matrix.npy')
    except FileNotFoundError:
        corrMatrix = build_corrMatrix(trainData, 'correlation_matrix.npy')
    '''
    
    t0 = time.clock()
    
    # REMOVE BELOW FOR FASTER PERFORMANCE ON MULTIPLE RUNS, AND UNCOMMENT ABOVE
    corrMatrix = build_corrMatrix(trainData, 'correlation_matrix.npy')
    
    reconstructedTrainBasic = collaborative_basic(trainData, testData, corrMatrix, K)
    RMSEbasic, SRCbasic, precisionTopKbasic = get_metrics(reconstructedTrainBasic, testData)
    t1 = time.clock()
    print('basic:    RMSE = {}; SRC = {}; Precision on top K = {}; time taken = {}'.format(RMSEbasic, SRCbasic, precisionTopKbasic, t1-t0))
    
    # REMOVE BELOW FOR FASTER PERFORMANCE ON MULTIPLE RUNS, AND UNCOMMENT ABOVE
    corrMatrix = build_corrMatrix(trainData, 'correlation_matrix.npy')
    
    reconstructedTrainBaseline = collaborative_baseline(trainData, testData, corrMatrix, K)
    RMSEbaseline, SRCbaseline, precisionTopKbaseline = get_metrics(reconstructedTrainBaseline, testData)
    t2 = time.clock()
    print('baseline: RMSE = {}; SRC = {}; Precision on top K = {}; time taken = {}'.format(RMSEbaseline, SRCbaseline, precisionTopKbaseline, t2-t1))


if __name__ == '__main__':
    main()

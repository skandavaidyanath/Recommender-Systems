# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 17:03:28 2018

@author: Skanda
"""

import pandas as pd
import numpy as np


def convert(data, min_user_index, max_user_index, nb_movies):
    new_data = []
    for id_users in range(min_user_index, max_user_index + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

      
def main():
    training_set = pd.read_csv('train.csv')
    test_set = pd.read_csv('test.csv')
    training_set = np.array(training_set, dtype='int')
    test_set = np.array(test_set, dtype='int')
    nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
    nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))
    min_user_index = 1
    max_user_index = nb_users
    training_set = convert(training_set, min_user_index, max_user_index, nb_movies)
    test_set = convert(test_set, min_user_index, max_user_index, nb_movies)
    training_data = np.array([np.array(x) for x in training_set])
    test_data = np.array([np.array(x) for x in test_set])
    np.save('train.npy',training_data)
    np.save('test.npy', test_data)
    
if __name__ == '__main__':
    main()
    
    
    
    
    
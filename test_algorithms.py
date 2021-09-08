from CreateDataset import get_coordinates_from_file, get_solution_from_file, get_sequence_length, get_sequence, create_record, get_tour_error, create_new_tour, get_tour_length
from random import seed, randint
from csv import writer
from sklearn.model_selection import train_test_split
from scipy.spatial import distance_matrix
from itertools import permutations

import numpy as np
import pandas as pd
import os
import time
import argparse

""" Create the CSV dataset file
"""
def create_csv_file(FOLDER_NAME, timestamp, N_RECORDS=80, N_SEQUENCE=5):
    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)
    else:
        if os.path.exists(f'{FOLDER_NAME}/{timestamp}_NRECORDS{N_RECORDS}_NSEQUENCE{N_SEQUENCE}.csv'):
            os.remove(f'{FOLDER_NAME}/{timestamp}_NRECORDS{N_RECORDS}_NSEQUENCE{N_SEQUENCE}.csv')
    columns = ['Sequence Error', 'Tour Error','Random Forest', 'Accuracy Random Forest', 'Naive Bayes', 'Accuracy Naive Bayes', 'Ada Boost', 'Accuracy Ada Boost', 'Decision Tree', 'Accuracy Decision Tree', 'Logistic Regression', 'Accuracy Logistic Regression', 'K-Nearest Neighbors', 'Accuracy KNN']
    with open(f'{FOLDER_NAME}/{timestamp}_NRECORDS{N_RECORDS}_NSEQUENCE{N_SEQUENCE}.csv', 'w', newline='') as csv_file:
        # creating a csv writer object 
        csvwriter = writer(csv_file, delimiter=';')
        # writing the fields 
        csvwriter.writerow(columns)

""" Add a record in the CSV dataset file
"""
def add_record_to_csv(FOLDER_NAME, timestamp, record, N_RECORDS=80, N_SEQUENCE=5):
    new_record = []
    for r in record:
        new_r = str(r)
        new_record.append(new_r.replace(".", ","))
    with open(f'{FOLDER_NAME}/{timestamp}_NRECORDS{N_RECORDS}_NSEQUENCE{N_SEQUENCE}.csv', 'a', newline='') as csv_file:
        # Pass this file object to csv.writer() and get a writer object
        writer_object = writer(csv_file, delimiter=';')
    
        # Pass the list as an argument into the writerow()
        writer_object.writerow(new_record)

def get_worst_permutation(tour, distance_matrix, random_node_index, tour_sequence):
    error_dict = {}
    permutation_tuple = list(permutations(tour_sequence))
    permutation_list = [list(i) for i in permutation_tuple]
    permutation_list.pop(0)
    for perm in permutation_list:
        new_tour = create_new_tour(tour, random_node_index, perm)
        perm_tour_error = get_tour_error(get_tour_length(tour, distance_matrix), get_tour_length(new_tour, distance_matrix))
        error_dict[tuple(perm)] = perm_tour_error
    error_dict_sorted = {k: v for k, v in sorted(error_dict.items(), key=lambda item: item[1], reverse = True)}
    print(error_dict_sorted)
    return list(error_dict_sorted.keys())[0], list(error_dict_sorted.values())[0], list(error_dict_sorted.keys())[len(error_dict_sorted)//2], list(error_dict_sorted.values())[len(error_dict_sorted)//2]

def dataset_setup(N_RECORDS=80, N_SEQUENCES=5):
    DATASET_FOLDER = './Dataset'
    DATASET_NAME = f'Dataset_NRECORDS{N_RECORDS}_NSEQUENCE{N_SEQUENCES}'
    dataset_file = pd.read_csv(f'{DATASET_FOLDER}/{DATASET_NAME}.csv')

    X = dataset_file.iloc[:, 0:15].values
    y = dataset_file[['Good']]

    # Training and Test Division
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    #print(len(X_train), len(y_train))
    #print(len(X_test), len(y_test))

    return X_train, X_test, y_train, y_test

def RandomForestAlgoritm(N_RECORDS=80, N_SEQUENCES=5):
    from sklearn.ensemble import  RandomForestClassifier
    rfm = RandomForestClassifier(n_estimators=15, oob_score=False, n_jobs=-1, 
                           random_state=101, max_features=None, min_samples_leaf=30)
    X_train, X_test, y_train, y_test = dataset_setup(N_RECORDS, N_SEQUENCES)
    rfm.fit(X_train, y_train.values.ravel())
    # print(f'\nRandom Forest:\n')
    # print("TRAIN SET", rfm.score(X_train, y_train))
    # print("TEST  SET", rfm.score(X_test, y_test))
    # prediction = rfm.predict(record)
    # accuracy = rfm.predict_proba(record)
    # #print(f'Prediction: {prediction}\n')
    # print(f'Accuracy: {accuracy}\n')
    return rfm

def NaiveBayesAlgorithm(N_RECORDS=80, N_SEQUENCES=5):
    #  Random Forest Algor
    # Naive Bayes
    from sklearn.naive_bayes import GaussianNB
    nb= GaussianNB()
    X_train, X_test, y_train, y_test = dataset_setup()
    nb.fit(X_train, y_train.values.ravel())
    # print(f'\nNaive Bayes:\n')
    # print("TRAIN SET", nb.score(X_train, y_train))
    # print("TEST  SET", nb.score(X_test, y_test))
    # prediction = nb.predict(record)
    # accuracy = nb.predict_proba(record)
    # print(f'Prediction: {prediction}\n')
    # print(f'Accuracy: {accuracy}\n')
    return nb

def AdaBoostAlgorithm(N_RECORDS=80, N_SEQUENCES=5):
    from sklearn.ensemble import AdaBoostClassifier
    ada = AdaBoostClassifier(n_estimators=100, random_state=0)
    X_train, X_test, y_train, y_test = dataset_setup()
    ada.fit(X_train, y_train.values.ravel())
    # print(f'\nAdaBoost:\n')
    # print("TRAIN SET", ada.score(X_train, y_train))
    # print("TEST  SET", ada.score(X_test, y_test))
    # prediction = ada.predict(record)
    # accuracy = ada.predict_proba(record)
    # print(f'Prediction: {prediction}\n')
    # print(f'Accuracy: {accuracy}\n')
    return ada

def DecisionTreeAlgorithm(N_RECORDS=80, N_SEQUENCES=5):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = dataset_setup()
    clf.fit(X_train, y_train)
    # print(f'\nDecision Tree:\n')
    # print("TRAIN SET", clf.score(X_train, y_train))
    # print("TEST  SET", clf.score(X_test, y_test))
    # prediction = clf.predict(record)
    # accuracy = clf.predict_proba(record)
    # print(f'Prediction: {prediction}\n')
    # print(f'Accuracy: {accuracy}\n')
    return clf

def LogisticRegressionAlgorithm(N_RECORDS=80, N_SEQUENCES=5):
    from sklearn.linear_model import LogisticRegression
    lr=LogisticRegression(max_iter=1000)
    X_train, X_test, y_train, y_test = dataset_setup()
    lr.fit(X_train, y_train.values.ravel())
    # print(f'\nLogistic Regression:\n')
    # print("TRAIN SET", lr.score(X_train, y_train))
    # print("TEST  SET", lr.score(X_test, y_test))
    # prediction = lr.predict(record)
    # accuracy = lr.predict_proba(record)
    # print(f'Prediction: {prediction}\n')
    # print(f'Accuracy: {accuracy}\n')
    return lr

def KNNAlgorithm(N_RECORDS=80, N_SEQUENCES=5):
    from sklearn.neighbors import  KNeighborsClassifier
    knn=KNeighborsClassifier(n_neighbors=5)
    X_train, X_test, y_train, y_test = dataset_setup()
    knn.fit(X_train, y_train.values.ravel())
    # print(f'\nKNN:\n')
    # print("TRAIN SET", knn.score(X_train, y_train))
    # print("TEST  SET", knn.score(X_test, y_test))
    # prediction = knn.predict(record)
    # accuracy = knn.predict_proba(record)
    # print(f'Prediction: {prediction}\n')
    # print(f'Accuracy: {accuracy}\n')
    return knn

def test_algorithms(N_RECORDS, N_SEQUENCE):
    seed(100)
    TSP_TEST_FOLDER = './TSPRandomGraph-Test/Results'
    timestamp = time.strftime('%Y%m%d-%H.%M')
    create_csv_file(TSP_TEST_FOLDER, timestamp, N_RECORDS, N_SEQUENCE)
    for i in range(1, 11):
        file_name = f'problem_{i}'
        print(f'{file_name}\n')
        coords, dist_mat = get_coordinates_from_file(TSP_TEST_FOLDER, file_name)
        tour_length, tour = get_solution_from_file(TSP_TEST_FOLDER, file_name)
        random_node_index = randint(0,len(tour))
        solution_sequence = get_sequence(tour, random_node_index, 5)
        solution_sequence_length = get_sequence_length(solution_sequence, dist_mat)
        solution_record = [create_record(solution_sequence, coords, dist_mat, None)]
        solution_result = [0, 0]
        worst_perm, worst_perm_error, mid_perm, mid_perm_error = get_worst_permutation(tour, dist_mat, random_node_index, solution_sequence)
        worst_perm_length, mid_perm_length = get_sequence_length(worst_perm, dist_mat), get_sequence_length(mid_perm, dist_mat)
        mid_sequence_error = get_tour_error(solution_sequence_length, mid_perm_length)
        worst_sequence_error = get_tour_error(solution_sequence_length, worst_perm_length)
        worst_perm_record = [create_record(worst_perm, coords, dist_mat, None)]
        mid_perm_record = [create_record(mid_perm, coords, dist_mat, None)]
        mid_result = [mid_sequence_error, mid_perm_error]
        worst_result = [worst_sequence_error, worst_perm_error]
        print(f'Random NODE IDX: {random_node_index}\nSOL. Sequence: {solution_sequence},\nMID SEQ: {mid_perm}\t-> Error: {mid_sequence_error}, New Tour Error:{mid_perm_error}\nWORST SEQ: {worst_perm}\t-> Error: {worst_sequence_error}, New Tour Error:{worst_perm_error}\n')

        rfa = RandomForestAlgoritm()
        print('RANDOM FOREST:')
        print(f'Predict SOL  . SEQ: {rfa.predict(solution_record)} \tAccuracy: {rfa.predict_proba(solution_record)}')
        print(f'Predict MID  . SEQ: {rfa.predict(mid_perm_record)} \tAccuracy: {rfa.predict_proba(mid_perm_record)}')
        print(f'Predict WORST. SEQ: {rfa.predict(worst_perm_record)} \tAccuracy: {rfa.predict_proba(worst_perm_record)}\n')
        solution_result.append(rfa.predict(solution_record)[0])
        solution_result.append(max(rfa.predict_proba(solution_record)[0]))
        mid_result.append(rfa.predict(mid_perm_record)[0])
        mid_result.append(max(rfa.predict_proba(mid_perm_record)[0]))
        worst_result.append(rfa.predict(worst_perm_record)[0])
        worst_result.append(max(rfa.predict_proba(worst_perm_record)[0]))
        
        nba = NaiveBayesAlgorithm()
        print('NAIVE BAYES:')
        print(f'Predict SOL  . SEQ: {nba.predict(solution_record)} \tAccuracy: {nba.predict_proba(solution_record)}')
        print(f'Predict MID  . SEQ: {nba.predict(mid_perm_record)} \tAccuracy: {nba.predict_proba(mid_perm_record)}')
        print(f'Predict WORST. SEQ: {nba.predict(worst_perm_record)} \tAccuracy: {nba.predict_proba(worst_perm_record)}\n')
        solution_result.append(nba.predict(solution_record)[0])
        solution_result.append(max(nba.predict_proba(solution_record)[0]))
        mid_result.append(nba.predict(mid_perm_record)[0])
        mid_result.append(max(nba.predict_proba(mid_perm_record)[0]))
        worst_result.append(nba.predict(worst_perm_record)[0])
        worst_result.append(max(nba.predict_proba(worst_perm_record)[0]))

        ada = AdaBoostAlgorithm()
        print('ADA BOOST:')
        print(f'Predict SOL  . SEQ: {ada.predict(solution_record)} \tAccuracy: {ada.predict_proba(solution_record)}')
        print(f'Predict MID  . SEQ: {ada.predict(mid_perm_record)} \tAccuracy: {ada.predict_proba(mid_perm_record)}')
        print(f'Predict WORST. SEQ: {ada.predict(worst_perm_record)} \tAccuracy: {ada.predict_proba(worst_perm_record)}\n')
        solution_result.append(ada.predict(solution_record)[0])
        solution_result.append(max(ada.predict_proba(solution_record)[0]))
        mid_result.append(ada.predict(mid_perm_record)[0])
        mid_result.append(max(ada.predict_proba(mid_perm_record)[0]))
        worst_result.append(ada.predict(worst_perm_record)[0])
        worst_result.append(max(ada.predict_proba(worst_perm_record)[0]))

        dta = DecisionTreeAlgorithm()
        print('DECISION TREE:')
        print(f'Predict SOL  . SEQ: {dta.predict(solution_record)} \tAccuracy: {dta.predict_proba(solution_record)}')
        print(f'Predict MID  . SEQ: {dta.predict(mid_perm_record)} \tAccuracy: {dta.predict_proba(mid_perm_record)}')
        print(f'Predict WORST. SEQ: {dta.predict(worst_perm_record)} \tAccuracy: {dta.predict_proba(worst_perm_record)}\n')
        solution_result.append(dta.predict(solution_record)[0])
        solution_result.append(max(dta.predict_proba(solution_record)[0]))
        mid_result.append(dta.predict(mid_perm_record)[0])
        mid_result.append(max(dta.predict_proba(mid_perm_record)[0]))
        worst_result.append(dta.predict(worst_perm_record)[0])
        worst_result.append(max(dta.predict_proba(worst_perm_record)[0]))

        lra = LogisticRegressionAlgorithm()
        print('LOGISTIC REGRESSION:')
        print(f'Predict SOL  . SEQ: {lra.predict(solution_record)} \tAccuracy: {lra.predict_proba(solution_record)}')
        print(f'Predict MID  . SEQ: {lra.predict(mid_perm_record)} \tAccuracy: {lra.predict_proba(mid_perm_record)}')
        print(f'Predict WORST. SEQ: {lra.predict(worst_perm_record)} \tAccuracy: {lra.predict_proba(worst_perm_record)}\n')
        solution_result.append(lra.predict(solution_record)[0])
        solution_result.append(max(lra.predict_proba(solution_record)[0]))
        mid_result.append(lra.predict(mid_perm_record)[0])
        mid_result.append(max(lra.predict_proba(mid_perm_record)[0]))
        worst_result.append(lra.predict(worst_perm_record)[0])
        worst_result.append(max(lra.predict_proba(worst_perm_record)[0]))

        knn = KNNAlgorithm()
        print('K-NEAREST NEIGHBORS:')
        print(f'Predict SOL  . SEQ: {knn.predict(solution_record)} \tAccuracy: {knn.predict_proba(solution_record)}')
        print(f'Predict MID  . SEQ: {knn.predict(mid_perm_record)} \tAccuracy: {knn.predict_proba(mid_perm_record)}')
        print(f'Predict WORST. SEQ: {knn.predict(worst_perm_record)} \tAccuracy: {knn.predict_proba(worst_perm_record)}\n')
        solution_result.append(knn.predict(solution_record)[0])
        solution_result.append(max(knn.predict_proba(solution_record)[0]))
        mid_result.append(knn.predict(mid_perm_record)[0])
        mid_result.append(max(knn.predict_proba(mid_perm_record)[0]))
        worst_result.append(knn.predict(worst_perm_record)[0])
        worst_result.append(max(knn.predict_proba(worst_perm_record)[0]))

        add_record_to_csv(TSP_TEST_FOLDER, timestamp, solution_result)
        add_record_to_csv(TSP_TEST_FOLDER, timestamp, mid_result)
        add_record_to_csv(TSP_TEST_FOLDER, timestamp, worst_result)

    print(f'-----------------------------------\n')

def main():
    N_RECORDS = 80
    N_SEQUENCE = 5
    parser = argparse.ArgumentParser(description='Test Configuration')
    parser.add_argument('--r', dest='records', type=int, help='Records for each graph of the dataset to use')
    parser.add_argument('--s', dest='sequence_nodes', type=int, help='Nodes in each sequence')
    args = parser.parse_args()
    # Set the Configuration Parameter
    if args.record is not None:
        N_RECORDS = args.record
    if args.sequence_nodes is not None:
        N_SEQUENCE = args.sequence_node
    test_algorithms(N_RECORDS, N_SEQUENCE)

if __name__ == "__main__":
    main()    
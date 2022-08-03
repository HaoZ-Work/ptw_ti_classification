'''
This is used for training and saving the model.



'''
import pandas as pd

import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


NUM_FEA = 24


def load_data(data_folder:str, label_path:str) ->pd.DataFrame:
    '''
    Load the data to a DataFrame

    :param data_folder:  The folder containing csv data
    :param label_path:  The path of labels
    :return: meta_pd : meta data of current data set containing 3 columns: 'id','label','path'
    '''
    labels = pd.read_excel(label_path)
    labels = labels.rename(columns={"Unnamed: 0": "id"})

    samples_path = glob(data_folder+"*.csv")
    samples_id_path_dic = {x.split("/")[-1].split(".")[0]: x for x in samples_path}

    meta_pd = pd.DataFrame(columns=["id", "label", "path"])
    meta_pd["id"] = samples_id_path_dic.keys()
    meta_pd["path"] = meta_pd["id"].map(samples_id_path_dic.get)

    # from OK and NOK to 1 and  respectively
    for i in meta_pd.iterrows():
        if labels[labels["id"] == i[1]["id"]]["LINE"].values == ["OK"]:
            i[1]["label"] = 1
        else:
            i[1]["label"] = 0
        # print(labels[ labels["id"]==i[1]['id'] ]['LINE'].values)

    return meta_pd

def convert_meta_to_np(meta_data:pd.DataFrame,padding_method:str,max_len:int,fft:bool)-> np.array:
    '''
    Create X,y in np.array() from meta data. padding methon and max_len of feature can be selected manually.
    :param meta_data: meta_pd
    :param padding_method: how to pad the data. e.g. 'mean','zero'
    :param max_len: the maximum length of features
    :param fft: True for using fft preprocessing approach
    :return: X: np.array(num_sample,max_len,num_features), y:np.array(num_sample,1)
    '''
    # create an empty tensor container for loading csv file into one tensor.
    X = np.empty((1, max_len, NUM_FEA))
    y = np.array(0)

    for sample in meta_data["id"].to_list():
        current_path = meta_data[meta_data["id"] == sample]["path"].to_list()[0]
        current_sample = pd.read_csv(current_path).to_numpy()


        y = np.append(y, meta_data[meta_data["id"] == sample].label.values)
        # normalize the loaded data
        # print(current_sample.shape)


        if fft==True:
            current_sample = np.abs(np.fft.rfft(current_sample, axis=0))

        trs = preprocessing.Normalizer().fit(current_sample)
        current_sample = trs.transform(current_sample)

        # standardize the loaded data
        trs = preprocessing.StandardScaler().fit(current_sample)
        current_sample = trs.transform(current_sample)

        # The max length of time channel is 715.
        # If the current data is shorter than 715,
        # then pad the data to 715 with mean value of corresponding feature
        if current_sample.shape[0] != max_len:
            # print(current_sample.shape[0])
            mean = current_sample.mean(axis=0).reshape(1, NUM_FEA)
            for i in range(max_len - current_sample.shape[0]):
                # print(mean.shape)
                # print(current_sample.shape)
                current_sample = np.concatenate((current_sample, mean))
            # print(current_sample.shape)



        X = np.concatenate((X, current_sample.reshape((1, max_len, NUM_FEA))), axis=0)

    # drop the empty tensor
    X = X[1:, :, :]
    y = y[1:].astype("int")

    return X,y

def data_split(X: np.array,y:np.array,val_size:float)-> list:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=val_size, random_state=42, stratify=y,shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.3, random_state=42, stratify=y_test,shuffle=True
    )

    return X_train,y_train, X_val,y_val,X_test,y_test


def ml_classifier(X_train:np.array,y_train:np.array, X_val,y_val,X_test,y_test,clf:str, matrix:bool)->object:
    '''
    Train a certain classifier with help of sklearn.

    :param X_train: training data in shape (n,len,num_feature)
    :param y_train:  labels in (n,1)
    :param X_val:
    :param y_val:
    :param X_test:
    :param y_test:
    :param clf: the name of classifier from sklearn,e.g. 'rf','mlp'
    :param matrix: True for showing all matrix from training step,val set and test step, false for showing nothing.
    :return:  a trained classifier
    '''
    if clf == 'rf':
        clf = RandomForestClassifier(max_depth=10, random_state=42)

    elif clf == 'mlp':
        clf = MLPClassifier(
            hidden_layer_sizes=(3000, 2000, 1000, 500),
            early_stopping=True,
            max_iter=1000,
            learning_rate="adaptive",
        )
    # elif clf == 'other':
    #     clf = ()
    else:
        raise Exception(f"The {clf} classifier has not been implemented.")

    clf.fit(X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]), y_train)

    y_train_pred = clf.predict(
        X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    )
    y_val_pred = clf.predict(X_val.reshape(X_val.shape[0], X_val.shape[1] * X_val.shape[2]))

    y_test_pred = clf.predict(
        X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    )
    if matrix == True:
        print(classification_report(y_train, y_train_pred))
        print(classification_report(y_val, y_val_pred))
        print(classification_report(y_test, y_test_pred))
    return clf

def test_step(clf:object,X:np.array,y:np.array):
    '''

    :param clf: a trainined classifier from sklearn
    :param X: test set x data (n,len,num_feature)
    :param y:  test set label data (n,1)
    :return:
    '''
    y_test_pred = clf.predict(
        X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    )
    print(classification_report(y, y_test_pred))


def main():
    print('Loading data...')
    meta_pd_01 = load_data("data/Versuchreihe_09_2020/machine_data/","data/Versuchreihe_09_2020/Label.xlsx")
    meta_pd_02 = load_data("data/Versuchsreihe_01_2022/renamed_machine_data/","data/Versuchsreihe_01_2022/Label.xlsx")

    X_01,y_01 = convert_meta_to_np(meta_pd_01,'mean',715,True)
    X_02, y_02 = convert_meta_to_np(meta_pd_02, 'mean', 715,True)
    X_01_train,y_01_train, X_01_val,y_01_val,X_01_test,y_01_test = data_split(X_01,y_01,0.3)

    print("Training classifier...")
    ml_clf  = ml_classifier(X_01_train,y_01_train, X_01_val,y_01_val,X_01_test,y_01_test,'mlp',True)


    print("Testing on test set")
    test_step(ml_clf,X_02,y_02)


if __name__ == '__main__':
    main()
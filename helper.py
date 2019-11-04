import torch
import random
import copy
import random
import math
import os
import pickle
import numpy as np
import time
import glob
import torch.nn as nn

import pandas as pd


def ab_ret(x,y):# x=a-b, y=a/b
    b=x/(y-1)
    a= b*y
    return (a,b)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def dict_to_concat_data(data_dict):

    all_data_dict = {'input':[],'label':[]}

    for i in range(len(data_dict['a'])):
        inp = np.concatenate((data_dict['transcript'][i],data_dict['a'][i],[data_dict['view'][i]],data_dict['u'][i]))#data_dict['view'][i]
        label = data_dict['rating'][i]
        all_data_dict['input'].append(inp)
        all_data_dict['label'].append(label)

    all_data_dict['input'] = np.array(all_data_dict['input'])
    all_data_dict['label'] = np.array(all_data_dict['label'])

    return all_data_dict

def sample_indices(N):
    train_len = int(0.8 * N)
    train_indices = np.random.choice( N,size = train_len,replace=False)
    dev_indices = np.random.choice( train_indices,size = int(0.1*train_len),replace=False)
    test_indices = list(set(range(N)).difference(train_indices))
    return train_indices, dev_indices, test_indices

def cvt(ind_list, span=11):
    return np.array([range(i*span,(i+1)*span) for i in ind_list]).flatten()

def make_minibatch(list_index, minibatch_size=10):
    while True:
        np.random.shuffle(list_index)
        yield list_index[:minibatch_size]


def counterfactual_loss(cf_outputs,labels,epsilon=0.1,span=11):
    n = len(cf_outputs)

    labels = labels.repeat_interleave(span, dim=0)
    op = (torch.abs(cf_outputs - labels)-epsilon)
    #print(op)
    return op


def calc_acc(model_output, target):
    m= nn.Sigmoid()
    x= (m(model_output)>= 0.5).float()
    # print(model_output.shape)
    print(x)
    y = torch.eq(x, target).float()
    return x,y

import pickle

class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))

def cast(s,num):

    slen = len(s)
    to_add = num-slen
    if to_add%2 == 0:
        ad = int(to_add/2)
        return ad*' ' + s + ad*' '
    else:
        ad = int(to_add/2)
        return ad*' ' + s + (to_add - ad)*' '



def convert_dict_to_categorical(df):

    # convert dataframe from one hot encoded dict to categorical for gender rance
    df_for_metric = pd.DataFrame({})
    df_for_metric['view'] = df['view']
    
    protected_df = pd.DataFrame(data = df['a'], index=None, 
                                columns = ['female', 'male', 'gender_other', 'white', 'black_or_african_american',
                                                               'asian','race_other'] )
    
    gender_df = protected_df[['male','female','gender_other']]
    race_df = protected_df[['white', 'black_or_african_american','asian','race_other']]
    

    gender_race = pd.DataFrame({})
    gender_race['gender'] = gender_df.idxmax(1).tolist()
    gender_race['race'] =  race_df.idxmax(1).tolist()
    
    df_for_metric['gender'] = gender_race['gender']
    df_for_metric['race'] = gender_race['race']

    
    rating_df = pd.DataFrame(data = df['rating'], index=None, columns = ['beautiful', 'confusing',
                              'courageous', 'fascinating', 'funny', 'informative', 'ingenious',
                              'inspiring', 'jaw-dropping', 'longwinded', 'obnoxious', 'ok',
                              'persuasive', 'unconvincing'] )
    
    df_for_metric[['beautiful', 'confusing',
                              'courageous', 'fascinating', 'funny', 'informative', 'ingenious',
                              'inspiring', 'jaw-dropping', 'longwinded', 'obnoxious', 'ok',
                              'persuasive', 'unconvincing']] = rating_df[['beautiful', 'confusing',
                              'courageous', 'fascinating', 'funny', 'informative', 'ingenious',
                              'inspiring', 'jaw-dropping', 'longwinded', 'obnoxious', 'ok',
                              'persuasive', 'unconvincing']]


    # for transcripts use 50 columns for doc2vec50
    array = df['transcript']
    temp_transcript = pd.DataFrame(data=array[:,:],    # values
                                   index=None,    # 1st column as index
                                   columns=['t_'+str(i) for i in range(1,array.shape[1]+1)])

    for col in temp_transcript.columns:
        df_for_metric[col] = temp_transcript[col]
        
    return df_for_metric

def find_std_dev(pred_df, true_df):
    # input: dataframes created by convert_dict_to_categorical containing 14 rating category columns
    temp_pred = pred_df[['gender','race', 'beautiful', 'confusing',
                              'courageous', 'fascinating', 'funny', 'informative', 'ingenious',
                              'inspiring', 'jaw-dropping', 'longwinded', 'obnoxious', 'ok',
                              'persuasive', 'unconvincing']]
    temp_true = true_df[['gender','race', 'beautiful', 'confusing',
                              'courageous', 'fascinating', 'funny', 'informative', 'ingenious',
                              'inspiring', 'jaw-dropping', 'longwinded', 'obnoxious', 'ok',
                              'persuasive', 'unconvincing']]
    
    pred_prob_mat = temp_pred.groupby(['gender','race']).mean()
    truth_prob_mat = temp_true.groupby(['gender','race']).mean()

    pred_std, truth_std = pred_prob_mat.std().values, truth_prob_mat.std().values
    pred_mean, truth_mean = pred_prob_mat.mean().values, truth_prob_mat.mean().values
    


    return pred_std, truth_std, pred_mean, truth_mean, pred_prob_mat,truth_prob_mat








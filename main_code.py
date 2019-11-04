#!/usr/bin/env python
# coding: utf-8

# In[1]:


from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import DisparateImpactRemover, LFR
from aif360.algorithms.postprocessing import RejectOptionClassification
import pandas as pd
import os
from helper import *
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# # Load Data and Split

# In[2]:


data_frame_true = convert_dict_to_categorical(load_pickle('mydoc2vec_200.pkl'))
print(data_frame_true.groupby('gender').count())


data_frame_train, data_frame_test = train_test_split(data_frame_true,test_size=0.2)
print('train shape: ',data_frame_train.shape[0])
print('test shape: ',data_frame_test.shape[0])


# In[3]:


rating_names = ['beautiful', 'confusing', 'courageous', 'fascinating', 'funny', 'informative', 'ingenious', 'inspiring', 'jaw-dropping', 'longwinded', 'obnoxious', 'ok', 'persuasive', 'unconvincing']
protected_attribute_maps = [{0.0: 'male', 1.0: 'female',2.0:'gender_other'}]
default_mappings = {
    'label_maps': [{1.0: 1, 0.0: 0}],
    'protected_attribute_maps': [{0.0: 'white', 1.0: 'black_or_african_american',2.0:'asian',3.0:'race_other'},
                                 {0.0: 'male', 1.0: 'female',2.0:'gender_other'}]
}
prot_attr_dict = {'race':{0.0: 'white', 1.0: 'black_or_african_american',2.0:'asian',3.0:'race_other'},
                                 'gender':{0.0: 'male', 1.0: 'female',2.0:'gender_other'}}

privileged_classes=[lambda x: x == 'white',lambda x: x == 'male']
protected_attribute_names=['race', 'gender']
unpriv_list = [[{'race':1},{'race':2},{'race':3}],[{'gender':1},{'gender':2}]]
priv_list = [[{'race':0}],[{'gender':0}]]


# # Create data frame for each rating category

# In[4]:


train_df_list = [pd.DataFrame({}) for i in range(14)]
test_df_list = [pd.DataFrame({}) for i in range(14)]

for col in data_frame_true.columns:
    if col in rating_names:
        ind =  rating_names.index(col)
        train_df_list[ind][[col]] = data_frame_train[[col]]
        test_df_list[ind][[col]] = data_frame_test[[col]]
    else:
        for i in range(14):
            train_df_list[i][[col]] = data_frame_train[[col]]
            test_df_list[i][[col]] = data_frame_test[[col]]


# In[5]:


for (u,p) in zip(unpriv_list,priv_list):

    di_orig_list, di_pred_orig_list, di_pred_trans_list = [], [], []

    unpriv_label = '+'.join(['-'.join([prot_attr_dict[key][u_el[key]] for key in u_el]) for u_el in u])
    priv_label = '+'.join(['-'.join([prot_attr_dict[key][p_el[key]] for key in p_el]) for p_el in p])


    print('-------------------------------------------------------------------')
    print('unpriv_label:-->',unpriv_label)
    print('-------------------------------------------------------------------')
    print('priv_label  :-->',priv_label)
    print('-------------------------------------------------------------------')

    for i,label in enumerate(rating_names):
        print(label)

        scaler = MinMaxScaler(copy=False)

        train_dataset  = StandardDataset(df=train_df_list[i], label_name=label, favorable_classes=[1.0,1.0],
                            protected_attribute_names=protected_attribute_names, privileged_classes=privileged_classes) 
        test_dataset  = StandardDataset(df=test_df_list[i], label_name=label, favorable_classes=[1.0,1.0],
                            protected_attribute_names=protected_attribute_names, privileged_classes=privileged_classes) 
        train_dataset.features = scaler.fit_transform(train_dataset.features)
        test_dataset.features = scaler.fit_transform(test_dataset.features)
        
        
        
        index = [test_dataset.feature_names.index(x) for x in protected_attribute_names]           
        
        
        
        
        #Metric of Original Data
        train_dataset_metric = BinaryLabelDatasetMetric(train_dataset, unprivileged_groups=u, privileged_groups=p)
        test_dataset_metric = BinaryLabelDatasetMetric(test_dataset, unprivileged_groups=u, privileged_groups=p)
        di_orig_list.append(test_dataset_metric.disparate_impact())
        
        
       


    xaxis = np.arange(14)
    width = 0.2
    baseline = 1   
    fig, ax = plt.subplots()
    rects1 = ax.bar(xaxis, np.array(di_orig_list)- baseline, width,bottom=baseline, label='Original')
    # rects2 = ax.bar(xaxis + width, np.array(di_pred_orig_list)- baseline, width,bottom=baseline, label='Predicted Original')
    # rects3 = ax.bar(xaxis + width+width, np.array(di_pred_trans_list)- baseline, width,bottom=baseline, label='Predicted Transformed')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Disparate Impact')
    ax.set_title('Unfairness between '+priv_label+' and '+unpriv_label)
    ax.set_xticks(xaxis)
    ax.set_xticklabels(rating_names,rotation=60)
    #ax.legend()
    plt.tight_layout()
    plt.savefig('./Plots/'+priv_label+' '+unpriv_label+'.pdf')
    plt.show()





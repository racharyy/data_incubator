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
        
        
        #Metric for predicted orginal data        
        X_tr = np.delete(train_dataset.features,index,axis=1)
        X_te = np.delete(test_dataset.features,index,axis=1)
        y_tr = train_dataset.labels.ravel()
        

        clf = LogisticRegression(class_weight='balanced',solver='liblinear')
        clf.fit(X_tr,y_tr)

        test_orig_pred = test_dataset.copy()
        test_orig_pred.labels = clf.predict(X_te)

        test_orig_pred_metric = BinaryLabelDatasetMetric(test_orig_pred, unprivileged_groups=u, privileged_groups=p)        
        di_pred_orig_list.append(test_orig_pred_metric.disparate_impact())
        
        
        #Metric for predicted transformed data
        di = DisparateImpactRemover(repair_level=0.5)
        train_repd  = di.fit_transform(train_dataset)
        test_repd =  di.fit_transform(test_dataset)

        
        X_tr = np.delete(train_repd.features,index,axis=1)
        X_te = np.delete(test_repd.features,index,axis=1)
        y_tr = train_repd.labels.ravel()
        

        clf = LogisticRegression(class_weight='balanced',solver='liblinear')
        clf.fit(X_tr,y_tr)

        test_repd_pred = test_repd.copy()
        test_repd_pred.labels = clf.predict(X_te)

        test_repd_pred_metric = BinaryLabelDatasetMetric(test_repd_pred, unprivileged_groups=u, privileged_groups=p)
        di_pred_trans_list.append(test_repd_pred_metric.disparate_impact())


    xaxis = np.arange(14)
    width = 0.2
    baseline = 1   
    fig, ax = plt.subplots()
    #rects1 = ax.bar(xaxis, np.array(di_orig_list)- baseline, width,bottom=baseline, label='Original')
    rects2 = ax.bar(xaxis + width, np.array(di_pred_orig_list)- baseline, width,bottom=baseline, label='Predicted Original')
    rects3 = ax.bar(xaxis + width+width, np.array(di_pred_trans_list)- baseline, width,bottom=baseline, label='Predicted Transformed')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Group Probability before running classifier')
    ax.set_title('Group Fairness for different Label')
    ax.set_xticks(xaxis)
    ax.set_xticklabels(rating_names,rotation=60)
    ax.legend()
    plt.tight_layout()
    plt.savefig('./Plots/'+priv_label+' '+unpriv_label+'.pdf')
    plt.show()


# In[ ]:


# get_ipython().system('conda install tensorflow')


# # In[ ]:


# y


# In[6]:


from aif360.algorithms.inprocessing import PrejudiceRemover
from sklearn.preprocessing import StandardScaler


# In[ ]:


from collections import defaultdict

def test(dataset, model, thresh_arr):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0
    
    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                     + metric.true_negative_rate()) / 2)
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())
    
    return metric_arrs


# # In Processing

# In[ ]:


# model = PrejudiceRemover(sensitive_attr='race', eta=25.0)
# pr_orig_scaler = StandardScaler()
# for i,label in enumerate(rating_names):
#     print(label)
#     dataset = dataset_orig_panel19_train.copy()
#     dataset.features = pr_orig_scaler.fit_transform(dataset.features)

#     pr_orig_panel19 = model.fit(dataset)
iter = 0   
for (u,p) in zip(unpriv_list[1:],priv_list[1:]):

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
        #train_dataset_metric = BinaryLabelDatasetMetric(train_dataset, unprivileged_groups=u, privileged_groups=p)
        test_dataset_metric = BinaryLabelDatasetMetric(test_dataset, unprivileged_groups=u, privileged_groups=p)
        md = test_dataset_metric.disparate_impact()#.mean_difference()
        di_orig_list.append(md)
        
        
        #Metric for predicted orginal data        
        X_tr = np.delete(train_dataset.features,index,axis=1)
        X_te = np.delete(test_dataset.features,index,axis=1)
        y_tr = train_dataset.labels.ravel()
        

        clf = LogisticRegression(class_weight='balanced',solver='liblinear')
        clf.fit(X_tr,y_tr)

        test_orig_pred = test_dataset.copy()
        test_orig_pred.labels = clf.predict(X_te)

        test_orig_pred_metric = BinaryLabelDatasetMetric(test_orig_pred, unprivileged_groups=u, privileged_groups=p)        
        md = test_orig_pred_metric.disparate_impact()#.mean_difference()
        di_pred_orig_list.append(md)
        
        
        #Metric for predicted transformed data
        train_inproc_dataset = train_dataset.copy()
        train_inproc_dataset.features = pr_orig_scaler.fit_transform(train_inproc_dataset.features)
        if iter==0:
            sens_attr ='race'
        else:
            sens_attr ='gender'
        model = PrejudiceRemover(eta=50.0)
        pr_orig_scaler = StandardScaler()
        pr_fit = model.fit(train_inproc_dataset)
        

        test_inproc_pred = pr_fit.predict(test_dataset)

        test_repd_pred_metric = BinaryLabelDatasetMetric(test_inproc_pred, unprivileged_groups=u, privileged_groups=p)
        md = test_repd_pred_metric.disparate_impact()#.mean_difference()
        di_pred_trans_list.append(md)

    iter=iter+1
    xaxis = np.arange(14)
    width = 0.2
    baseline = 1   
    fig, ax = plt.subplots()
    rects1 = ax.bar(xaxis, np.array(di_orig_list)- baseline, width,bottom=baseline, label='Original')
    rects2 = ax.bar(xaxis + width, np.array(di_pred_orig_list)- baseline, width,bottom=baseline, label='Predicted Original')
    rects3 = ax.bar(xaxis + width+width, np.array(di_pred_trans_list)- baseline, width,bottom=baseline, label='Predicted Transformed')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Disparate Impact')
    #ax.set_title('Group Fairness for different Label')
    ax.set_xticks(xaxis)
    ax.set_xticklabels(rating_names,rotation=60)
    ax.legend()
    plt.tight_layout()
    plt.savefig('../Plots/'+priv_label+' '+unpriv_label+'_pr.pdf')
    plt.show()


# # Post Processing

# In[ ]:


for (u,p) in zip(unpriv_list[1:],priv_list[1:]):

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
        #train_dataset_metric = BinaryLabelDatasetMetric(train_dataset, unprivileged_groups=u, privileged_groups=p)
        test_dataset_metric = BinaryLabelDatasetMetric(test_dataset, unprivileged_groups=u, privileged_groups=p)
        md = test_dataset_metric.disparate_impact()#.mean_difference()
        di_orig_list.append(md)
        
        
        #Metric for predicted orginal data        
        X_tr = np.delete(train_dataset.features,index,axis=1)
        X_te = np.delete(test_dataset.features,index,axis=1)
        y_tr = train_dataset.labels.ravel()
        

        clf = LogisticRegression(class_weight='balanced',solver='liblinear')
        clf.fit(X_tr,y_tr)

        test_orig_pred = test_dataset.copy()
        test_orig_pred.labels = clf.predict(X_te)

        test_orig_pred_metric = BinaryLabelDatasetMetric(test_orig_pred, unprivileged_groups=u, privileged_groups=p)        
        md = test_orig_pred_metric.disparate_impact()#.mean_difference()
        di_pred_orig_list.append(md)
        
        
        #Metric for predicted transformed data
        train_postproc_dataset = train_dataset.copy()
        train_postproc_dataset.scores = clf.predict_proba(X_tr)
       
        post_proc = RejectOptionClassification(unprivileged_groups=u,privileged_groups=p)
        post_proc.fit(train_dataset,train_postproc_dataset)
        

        test_postproc_pred = post_proc.predict(test_dataset)

        test_repd_pred_metric = BinaryLabelDatasetMetric(test_postproc_pred, unprivileged_groups=u, privileged_groups=p)
        md = test_repd_pred_metric.disparate_impact()#.mean_difference()
        di_pred_trans_list.append(md)

    iter=iter+1
    xaxis = np.arange(14)
    width = 0.2
    baseline = 1   
    fig, ax = plt.subplots()
    rects1 = ax.bar(xaxis, np.array(di_orig_list)- baseline, width,bottom=baseline, label='Original')
    rects2 = ax.bar(xaxis + width, np.array(di_pred_orig_list)- baseline, width,bottom=baseline, label='Predicted Original')
    rects3 = ax.bar(xaxis + width+width, np.array(di_pred_trans_list)- baseline, width,bottom=baseline, label='Predicted Transformed')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Disparate Impact')
    #ax.set_title('Group Fairness for different Label')
    ax.set_xticks(xaxis)
    ax.set_xticklabels(rating_names,rotation=60)
    ax.legend()
    plt.tight_layout()
    plt.savefig('../Plots/'+priv_label+' '+unpriv_label+'_pr.pdf')
    plt.show()


# In[ ]:





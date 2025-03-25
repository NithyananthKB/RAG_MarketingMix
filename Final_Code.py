#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[4]:


df = pd.read_csv('C:/Users/A6191/OneDrive - Axtria/Documents/Study/ML Assignment/train_test_dataset.csv')


# In[5]:


#Shape of the dataframe (rows, columns)
df.shape


# In[6]:


#Viewing a sample of dataframe
df.head()


# In[7]:


df.head().T


# In[8]:


#Column information
df.info()


# In[9]:


#Checking if data have any duplicates
df.duplicated().sum()


# In[10]:


#Data description
df.describe(include = 'all').T


# In[11]:


nums = ['dependent_var','total_representative_visits_flags','sample_drop','repVisit_sampleDrop_interaction',
        'saving_cards_dropped','vouchers_dropped','seminar_as_attendee','total_seminar_as_speaker',
        'physician_hospital_affiliation','physician_in_group_practice','total_prescriptions_for_indication1',
        'total_prescriptions_for_indication2','total_prescriptions_for_indication3','total_patient_with_commercial_insurance_plan',
        'total_patient_with_medicare_insurance_plan','total_patient_with_medicaid_insurance_plan',
        'brand_web_impressions_flag','brand_ehr_impressions_flag','brand_enews_impressions_flag','brand_mobile_impressions_flag',
        'brand_organic_web_visits',
        'brand_paidsearch_visits','competitor_prescription_bucket','new_prescriptions_bucket',
        'physician_value_tier','urban_population_perc_in_physician_locality_flag',
        'population_with_health_insurance_in_last10q_bucket','physician_gender','physician_tenure_bucket','Age Group',
        'spl_nephrology','spl_urology','spl_other']


# In[12]:


for col in nums:
    plt.figure(figsize = (8,4))
    plt.hist(df[col])
    plt.title(col)
    plt.xlim(df[col].min(),df[col].quantile(1))
    plt.show()


# In[13]:


#sns.pairplot(df[nums])
#plt.show()


# In[14]:


#Correlation plot
cor = df[nums].corr()
plt.figure(figsize=(25,18))
sns.heatmap(cor,annot = True, cmap = 'coolwarm')
plt.show()


# Brand impressions are highly correlated. so replacing the 4 impressions with one. 
# Competitor and total prescription 1 is also highly correlated

# In[15]:


y = df['dependent_var']
nums_2 = ['total_representative_visits_flags','repVisit_sampleDrop_interaction',
        'saving_cards_dropped','vouchers_dropped','seminar_as_attendee','total_seminar_as_speaker',
        'physician_hospital_affiliation','total_prescriptions_for_indication1',
        'total_patient_with_commercial_insurance_plan',
        'total_patient_with_medicare_insurance_plan','total_patient_with_medicaid_insurance_plan',
        'brand_web_impressions_flag',
        'brand_paidsearch_visits','new_prescriptions_bucket',
        'physician_value_tier','urban_population_perc_in_physician_locality_flag',
        'population_with_health_insurance_in_last10q_bucket','physician_gender','physician_tenure_bucket',
        'spl_urology','spl_other']

df_num = df[nums_2]


# In[16]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

X = df_num.iloc[:,:]
calc_vif(X).sort_values(by = ["VIF"],axis = 0, ascending = [False])


# spl nephrology and spl other have a correlation of 0.71.
# paid search and organic search have high correlation.
# age and tenure are similar.

# In[17]:


#Anova for feature selection

from sklearn.feature_selection import f_classif

fvalue,pvalue = f_classif(df_num,y)

for i in range(len(nums_2)):
    print(nums_2[i],pvalue[i])


# physician tenure bucket has a pvalue more than 0.5

# In[18]:


# Variation in data - using variation (can use standard deviation or even skewness/kurtosis values)
df.var()


# In[19]:


# Final features
final_features = nums

len(final_features)


# In[20]:


df_fin = df[final_features]
df_fin.shape


# In[21]:


df_fin.head()


# In[22]:


x = df_fin.drop(['dependent_var'],axis = 1)
y = df_fin['dependent_var']


# In[23]:


x.head()


# In[24]:


y.head()


# In[25]:


from collections import Counter


# In[26]:


print('orginal dataset {}'.format(Counter(y)))


# In[27]:


print(x.shape)


# In[28]:


print(y.shape)


# In[29]:


from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification


# In[30]:


# # Making an instance of SMOTE class 
# # For oversampling of minority class
# smote = SMOTE()
  
# # Fit predictor (x variable)
# # and target (y variable) using fit_resample()
# xoversmote, yoversmote = smote.fit_resample(x, y)


# In[31]:


# print('orginal dataset {}'.format(Counter(y)))
# print('orginal dataset {}'.format(Counter(yoversmote)))


# In[32]:


# xoversmote.count()


# In[33]:


# xoversmote.describe()


# In[34]:


# yoversmote.count()


# In[35]:


# x.count()


# In[36]:


#check= pd.concat([y_oversmote, x_oversmote])


# In[37]:


#check.describe()


# In[38]:


# a=x_oversmote
# b=y_oversmote
# b.info()


# In[39]:


# print(x_oversmote.shape)
# print(y_oversmote.shape)
# print(x.shape)
# print(y.shape)


# # Train Test Split

# In[40]:


#Train and test split
from sklearn.model_selection import train_test_split
x_train,xtest,y_train,ytest = train_test_split(x,y,test_size=0.2,random_state=5,stratify=y)


# In[41]:


print(x.shape)
print(x_train.shape)
print(xtest.shape)


# In[42]:


# Making an instance of SMOTE class 
# For oversampling of minority class
smote = SMOTE()
  
# Fit predictor (x variable)
# and target (y variable) using fit_resample()
xtrain,ytrain = smote.fit_resample(x_train, y_train)


# In[43]:


print(x.shape)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)


# In[44]:


print('orginal dataset {}'.format(Counter(y_train)))
print('orginal dataset {}'.format(Counter(ytrain)))


# In[45]:


#train-test-validation split

# Let's say we want to split the data in 80:10:10 for train:valid:test dataset
#train_size=0.8

# In the first step we will split the data in training and remaining dataset
#X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8)

# Now since we want the valid and test size to be equal (10% each of overall data). 
# we have to define valid_size=0.5 (that is 50% of remaining data)
#test_size = 0.5
#X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)


# In[46]:


# print(X_train.shape), print(y_train.shape)
# print(X_valid.shape), print(y_valid.shape)
# print(X_test.shape), print(y_test.shape)


# In[47]:


#Scaling data
from sklearn.preprocessing import  StandardScaler
scalar = StandardScaler()

non_binary_numeric_features = ['total_representative_visits_flags',
        'saving_cards_dropped','vouchers_dropped','total_prescriptions_for_indication1',
        'total_patient_with_commercial_insurance_plan',
        'total_patient_with_medicare_insurance_plan','total_patient_with_medicaid_insurance_plan',
        'brand_paidsearch_visits','new_prescriptions_bucket',
        'physician_value_tier',
        'population_with_health_insurance_in_last10q_bucket','physician_tenure_bucket']

xtrain[non_binary_numeric_features] = scalar.fit_transform(xtrain[non_binary_numeric_features])
xtest[non_binary_numeric_features] = scalar.transform(xtest[non_binary_numeric_features])


# In[ ]:





# In[48]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier


# model = RandomForestClassifier()
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(model, xtrain, ytrain, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

# model = RandomForestClassifier()
# # fit the model on the whole dataset
# model.fit(xtrain, ytrain)
# # make predictions on test data
# yhat = model.predict(xtest)
# print(yhat)

# print("Precision -",precision_score(ytest, yhat))
# print("Recall -",recall_score(ytest, yhat))
# print("F1 Score -",f1_score(ytest, yhat))
# print("Accuracy -",f1_score(ytest, yhat))

# # Random forest

# #Random Forerst with grid search and random search
# 
# from sklearn.ensemble import RandomForestClassifier
# 
# rf = RandomForestClassifier(random_state = 42)

# # Look at parameters used by our current forest
# print('Parameters currently in use:\n')
# print(rf.get_params())

# from sklearn.model_selection import RandomizedSearchCV
# 
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# random_grid

# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestClassifier()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# # Fit the random search model
# rf_random.fit(xtrain, ytrain)

# rf_random.best_params_

# optimised_random_forest_random_search = rf_random.best_estimator_

# y_pred = optimised_random_forest_random_search.predict(xtest)

# print("Precision -",precision_score(ytest, y_pred))
# print("Recall -",recall_score(ytest, y_pred))
# print("F1 Score -",f1_score(ytest, y_pred))
# print("Accuracy -",f1_score(ytest, y_pred))

# #Grid Search
# 
# from sklearn.model_selection import GridSearchCV
# # Create the parameter grid based on the results of random search 
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100, 110],
#     'max_features': [2, 3],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 200, 300, 1000]
# }
# # Create a based model
# rf = RandomForestClassifier()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 3, n_jobs = -1, verbose = 2)

# # Fit the grid search to the data
# grid_search.fit(xtrain, ytrain)
# grid_search.best_params_

# optimised_random_forest_grid_search = rf_random.best_estimator_

# y_pred = optimised_random_forest_grid_search.predict(xtest)

# print("Precision -",precision_score(ytest, y_pred))
# print("Recall -",recall_score(ytest, y_pred))
# print("F1 Score -",f1_score(ytest, y_pred))
# print("Accuracy -",f1_score(ytest, y_pred))

# In[67]:


print(xtest.shape)


# In[68]:


#Importing required models/metrics etc.
from sklearn import linear_model
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
import lightgbm as lgb
import catboost  as cat
import xgboost as xgb


# In[69]:


#Linear models
model = linear_model.LogisticRegression()

#train the model using trian data
np.random.seed(321)
model.fit(xtrain,ytrain)


# In[70]:


#check the performance of the model using test data
ypred = model.predict(xtest)
print("Accuracy: ",round(metrics.accuracy_score(ytest,ypred),2))
print("Recall: ",round(metrics.recall_score(ytest,ypred),2))
print("Precision: ",round(metrics.precision_score(ytest,ypred),2))
print("F1 Score: ",round(metrics.f1_score(ytest,ypred),2))


# In[71]:


#Nearest neighbor models
model = neighbors.KNeighborsClassifier()

#train the model using trian data
np.random.seed(321)
model.fit(xtrain,ytrain)

#check the performance of the model using test data
ypred = model.predict(xtest)
print("Accuracy: ",round(metrics.accuracy_score(ytest,ypred),2))
print("Recall: ",round(metrics.recall_score(ytest,ypred),2))
print("Precision: ",round(metrics.precision_score(ytest,ypred),2))
print("F1 Score: ",round(metrics.f1_score(ytest,ypred),2))


# In[72]:


#Dtree models
model = tree.DecisionTreeClassifier()

#train the model using trian data
np.random.seed(321)
model.fit(xtrain,ytrain)

#check the performance of the model using test data
ypred = model.predict(xtest)
print("Accuracy: ",round(metrics.accuracy_score(ytest,ypred),2))
print("Recall: ",round(metrics.recall_score(ytest,ypred),2))
print("Precision: ",round(metrics.precision_score(ytest,ypred),2))
print("F1 Score: ",round(metrics.f1_score(ytest,ypred),2))


# In[73]:


#RF models
model = ensemble.RandomForestClassifier()

#train the model using trian data
np.random.seed(321)
model.fit(xtrain,ytrain)

#check the performance of the model using test data
ypred = model.predict(xtest)
print("Accuracy: ",round(metrics.accuracy_score(ytest,ypred),2))
print("Recall: ",round(metrics.recall_score(ytest,ypred),2))
print("Precision: ",round(metrics.precision_score(ytest,ypred),2))
print("F1 Score: ",round(metrics.f1_score(ytest,ypred),2))


# In[74]:


#GBM Models
model = ensemble.GradientBoostingClassifier()

#train the model using trian data
np.random.seed(321)
model.fit(xtrain,ytrain)

#check the performance of the model using test data
ypred = model.predict(xtest)
print("Accuracy: ",round(metrics.accuracy_score(ytest,ypred),2))
print("Recall: ",round(metrics.recall_score(ytest,ypred),2))
print("Precision: ",round(metrics.precision_score(ytest,ypred),2))
print("F1 Score: ",round(metrics.f1_score(ytest,ypred),2))


# In[75]:


#Light GBM models
model = lgb.LGBMClassifier()

#train the model using trian data
np.random.seed(321)
model.fit(xtrain,ytrain)

#check the performance of the model using test data
ypred = model.predict(xtest)
print("Accuracy: ",round(metrics.accuracy_score(ytest,ypred),2))
print("Recall: ",round(metrics.recall_score(ytest,ypred),2))
print("Precision: ",round(metrics.precision_score(ytest,ypred),2))
print("F1 Score: ",round(metrics.f1_score(ytest,ypred),2))


# In[76]:


#Cat boost models
model = cat.CatBoostClassifier()

#train the model using trian data
np.random.seed(321)
model.fit(xtrain,ytrain)

#check the performance of the model using test data
ypred = model.predict(xtest)
print("Accuracy: ",round(metrics.accuracy_score(ytest,ypred),2))
print("Recall: ",round(metrics.recall_score(ytest,ypred),2))
print("Precision: ",round(metrics.precision_score(ytest,ypred),2))
print("F1 Score: ",round(metrics.f1_score(ytest,ypred),2))


# In[77]:


#Xgboost models
model = xgb.XGBClassifier(use_label_encoder = False)

#train the model using trian data
np.random.seed(321)
model.fit(xtrain,ytrain)

#check the performance of the model using test data
ypred = model.predict(xtest)
print("Accuracy: ",round(metrics.accuracy_score(ytest,ypred),2))
print("Recall: ",round(metrics.recall_score(ytest,ypred),2))
print("Precision: ",round(metrics.precision_score(ytest,ypred),2))
print("F1 Score: ",round(metrics.f1_score(ytest,ypred),2))


# In[78]:


# Feature Importance
optimised_random_forest_grid_search.feature_importances_


# In[79]:


feature_names = list(xtrain.columns)


# In[85]:


plt.barh(feature_names[:33], optimised_random_forest_grid_search.feature_importances_[:33])


# physician_hospital_affiliation, total_seminar_as_speaker, semair_as_attendee, Vouchers_dropped,
# last - saving_cards_droped, revisit_sampleDrop_interaction

# In[81]:


importances = optimised_random_forest_grid_search.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
# indices of the coulumns in decending order of their feature importance
sorted_indices


# In[82]:


#reordering the column names same as decending order of feature importance
feature_names = [feature_names[i] for i in sorted_indices]


# In[83]:


#reordering the importance in decending order
importances = [importances[i] for i in sorted_indices]


# In[84]:


plt.barh(feature_names[:30], importances[:30])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





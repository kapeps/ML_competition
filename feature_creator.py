# imports
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from numpy import save
from numpy import load
from dateutil.parser import parse
import re

oneHotCutOut = 10

# read datasets
train_df = pd.read_csv('train_ml.csv', index_col=0)
test_df = pd.read_csv('test_ml.csv', index_col=0)

train_y = train_df[['updates', 'personal', 'promotions',
                        'forums', 'purchases', 'travel',
                        'spam', 'social']]

oneHotCutOut = 10

train_df = pd.read_csv('train_ml.csv', index_col=0)
test_df = pd.read_csv('test_ml.csv', index_col=0)

train_y = train_df[['updates', 'personal', 'promotions',
                        'forums', 'purchases', 'travel',
                        'spam', 'social']]


## ccs 
data_field_name = 'ccs'

train_x = train_df[[data_field_name]]
train_x = train_x.fillna(value=0)
test_x = test_df[[data_field_name]]
test_x = test_x.fillna(value=0)

features_names= np.array(['ccs'])

S =  StandardScaler()
train_x = S.fit_transform(train_x)
test_x = S.transform(test_x)
features_train = np.array( train_x)
features_test = np.array(test_x)



## images  
data_field_name = 'images'

train_x = train_df[[data_field_name]]
train_x = train_x.fillna(value=0)
test_x = test_df[[data_field_name]]
test_x = test_x.fillna(value=0)

for i in range(train_x.shape[1]):
    features_names = np.append(features_names,['images'])

features_train = np.concatenate((features_train, train_x), axis=1)
features_test = np.concatenate((features_test, test_x), axis=1)


## urls  
data_field_name = 'urls'

train_x = train_df[[data_field_name]]
train_x = train_x.fillna(value=0)
test_x = test_df[[data_field_name]]
test_x = test_x.fillna(value=0)

for i in range(train_x.shape[1]):
    features_names = np.append(features_names,['urls'])

features_train = np.concatenate((features_train, train_x), axis=1)
features_test = np.concatenate((features_test, test_x), axis=1)


## salutations   
data_field_name = 'salutations'

train_x = train_df[[data_field_name]]
train_x = train_x.fillna(value=0)
test_x = test_df[[data_field_name]]
test_x = test_x.fillna(value=0)

for i in range(train_x.shape[1]):
    features_names = np.append(features_names,['salutations'])

features_train = np.concatenate((features_train, train_x), axis=1)
features_test = np.concatenate((features_test, test_x), axis=1)


## bcced    
data_field_name = 'bcced'

train_x = train_df[[data_field_name]]
train_x = train_x.fillna(value=0)
test_x = test_df[[data_field_name]]
test_x = test_x.fillna(value=0)

for i in range(train_x.shape[1]):
    features_names = np.append(features_names,['bcced'])

features_train = np.concatenate((features_train, train_x), axis=1)
features_test = np.concatenate((features_test, test_x), axis=1)


## designation    
data_field_name = 'designation'

train_x = train_df[[data_field_name]]
train_x = train_x.fillna(value=0)
test_x = test_df[[data_field_name]]
test_x = test_x.fillna(value=0)

for i in range(train_x.shape[1]):
    features_names = np.append(features_names,['designation'])

features_train = np.concatenate((features_train, train_x), axis=1)
features_test = np.concatenate((features_test, test_x), axis=1)


## chars_in_subject   
data_field_name = 'chars_in_subject'

train_x = train_df[[data_field_name]]
train_x = train_x.fillna(value=0)
test_x = test_df[[data_field_name]]
test_x = test_x.fillna(value=0)

for i in range(train_x.shape[1]):
    features_names = np.append(features_names,['chars_in_subject'])

features_train = np.concatenate((features_train, train_x), axis=1)
features_test = np.concatenate((features_test, test_x), axis=1)


## chars_in_body    
data_field_name = 'chars_in_body'

train_x = train_df[[data_field_name]]
train_x = train_x.fillna(value=0)
test_x = test_df[[data_field_name]]
test_x = test_x.fillna(value=0)

for i in range(train_x.shape[1]):
    features_names = np.append(features_names,['chars_in_body'])

features_train = np.concatenate((features_train, train_x), axis=1)
features_test = np.concatenate((features_test, test_x), axis=1)


# date
test_x = test_df[['date']]
test_x = test_x.fillna(value='None')
test_x = np.array(test_x)

d = np.array(parse(test_x[0][0]).timetuple())
d = np.concatenate((d,np.array([0])), axis =None)
first = True
x = 0

for row in test_x:
    if first==True:
        first = False
    else:
        if (len(row[0].split()[-1]) == 5):
            if(row[0].split()[-1][-1] == ')'):                
                timezone = int(row[0].split()[-2][3:])+int(row[0].split()[-2][:3])*60
            else:
                timezone = int(row[0].split()[-1][3:])+int(row[0].split()[-1][:3])*60
        else:
            if (len(row[0].split()[-2]) == 5):
                timezone = int(row[0].split()[-2][3:])+int(row[0].split()[-2][:3])*60
            else:
                timezone = 0
        row = np.array(parse(re.sub(' +', ' ', row[0])[:31]).timetuple())
        row = np.concatenate([row, np.array([timezone])])

        d = np.vstack([d, row])

features_test = np.concatenate((features_test, np.array(d)), axis=1)

train_x = train_df[['date']]
train_x = train_x.fillna(value='None')
train_x = np.array(train_x)
d = np.array(parse(train_x[0][0]).timetuple())
d = np.concatenate((d,np.array([0])), axis =None)

first = True
for row in train_x:
    if first==True:
            first = False
    else:
        if (len(row[0].split()[-1]) == 5):
            if(row[0].split()[-1][-1] == ')'):                
                timezone = int(row[0].split()[-2][3:])+int(row[0].split()[-2][:3])*60
            else:
                timezone = int(row[0].split()[-1][3:])+int(row[0].split()[-1][:3])*60
        else:
            if (len(row[0].split()[-2]) == 5):
                timezone = int(row[0].split()[-2][3:])+int(row[0].split()[-2][:3])*60
            else:
                timezone = 0
        row = np.array(parse(re.sub(' +', ' ', row[0])[:31]).timetuple())
        row = np.concatenate([row, np.array([timezone])])
        d = np.vstack([d, row])

for i in range(d.shape[1]):
    features_names = np.append(features_names,['date'])

features_train = np.concatenate((features_train, np.array(d)), axis=1)
train_x = np.divide(features_train[:,1],features_train[:,7]).T.reshape((39671,1))
test_x =  np.divide(features_test[:,1],features_test[:,7]).T.reshape((17002,1))

S =  StandardScaler()
train_x = S.fit_transform(train_x)
test_x = S.transform(test_x)

features_train = np.hstack([features_train, train_x])
features_test = np.hstack([features_test,test_x])
features_names = np.append(features_names,['ratioImageBody'])

train_x = np.divide(features_train[:,2],features_train[:,7]).T.reshape((39671,1))
test_x =  np.divide(features_test[:,2],features_test[:,7]).T.reshape((17002,1))

S =  StandardScaler()
train_x = S.fit_transform(train_x)
test_x = S.transform(test_x)

features_train = np.hstack([features_train, train_x])
features_test = np.hstack([features_test,test_x])
features_names = np.append(features_names,['ratioUrlBody'])

train_x = np.divide(features_train[:,6],features_train[:,7]).T.reshape((39671,1))
test_x =  np.divide(features_test[:,6],features_test[:,7]).T.reshape((17002,1))

S =  StandardScaler()
train_x = S.fit_transform(train_x)
test_x = S.transform(test_x)

features_train = np.hstack([features_train, train_x])
features_test = np.hstack([features_test,test_x])
features_names = np.append(features_names,['ratioSubjectBody'])


train_x = np.multiply(features_train[:,3],features_train[:,4]).T.reshape((39671,1))
test_x =  np.multiply(features_test[:,3],features_test[:,4]).T.reshape((17002,1))

S =  StandardScaler()
train_x = S.fit_transform(train_x)
test_x = S.transform(test_x)

features_train = np.hstack([features_train, train_x])
features_test = np.hstack([features_test,test_x])
features_names = np.append(features_names,['ratioSubjectBody'])


##Correcting the data
train_x = np.array(features_train[:,[1,2,6,7]])
test_x = np.array(features_test[:,[1,2,6,7]])
S =  StandardScaler()
train_x = S.fit_transform(train_x)
test_x = S.transform(test_x)
features_train[:,[1,2,6,7]] = train_x
features_test[:,[1,2,6,7]] = test_x

##Correcting the data
train_x = np.array(features_train[:,[8,9,10,11,12,13,14,15,17]])
test_x = np.array(features_test[:,[8,9,10,11,12,13,14,15,17]])
S =  StandardScaler()
train_x = S.fit_transform(train_x)
test_x = S.transform(test_x)
features_train[:,[8,9,10,11,12,13,14,15,17]] = train_x
features_test[:,[8,9,10,11,12,13,14,15,17]] = test_x

features_train = np.delete(features_train, 13, 1)
features_test = np.delete(features_test, 13, 1)


##mail_type
train_x = train_df[['mail_type']]
train_x = train_x.fillna(value='None')

test_x = test_df[['mail_type']]
test_x = test_x.fillna(value='None')

feat_enc = OneHotEncoder()
feat_enc.fit(np.vstack([train_x, test_x]))
train_x_featurized = feat_enc.transform(train_x).toarray()
test_x_featurized = feat_enc.transform(test_x).toarray()

one_test = np.array(train_x_featurized.sum(axis=0))
test_x_featurized = np.array(test_x_featurized).T[np.where(one_test>5)].T   
train_x_featurized = np.array(train_x_featurized).T[np.where(one_test>5)].T 

for i in range(test_x_featurized.shape[1]):
        features_names = np.append(features_names,['mail_type'])


features_train = np.concatenate((features_train, np.array(train_x_featurized)), axis=1)
features_test = np.concatenate((features_test, np.array(test_x_featurized)), axis=1)



## ORG
data_field_name = 'org'

train_x = train_df[[data_field_name]]
train_x = train_x.fillna(value='None')
test_x = test_df[[data_field_name]]
test_x = test_x.fillna(value='None')

feat_enc = OneHotEncoder()
feat_enc.fit(np.vstack([train_x, test_x]))
train_x_featurized = feat_enc.transform(train_x).toarray()
test_x_featurized = feat_enc.transform(test_x).toarray()

one_test = np.array(train_x_featurized.sum(axis=0))
test_x_featurized = np.array(test_x_featurized).T[np.where(one_test>10)].T   
train_x_featurized = np.array(train_x_featurized).T[np.where(one_test>10)].T 
for i in range(test_x_featurized.shape[1]):
        features_names = np.append(features_names,['Org'])

features_train = np.concatenate((features_train, np.array(train_x_featurized)), axis=1)
features_test = np.concatenate((features_test, np.array(test_x_featurized)), axis=1)


## TLD
data_field_name = 'tld'

train_x = train_df[[data_field_name]]
train_x = train_x.fillna(value='None')
test_x = test_df[[data_field_name]]
test_x = test_x.fillna(value='None')

feat_enc = OneHotEncoder()
feat_enc.fit(np.vstack([train_x, test_x]))
train_x_featurized = feat_enc.transform(train_x).toarray()
test_x_featurized = feat_enc.transform(test_x).toarray()

one_test = np.array(train_x_featurized.sum(axis=0))
train_x_featurized = np.array(train_x_featurized).T[np.where(one_test>5)].T 
test_x_featurized = np.array(test_x_featurized).T[np.where(one_test>5)].T

features_train = np.concatenate((features_train, np.array(train_x_featurized)), axis=1)
features_test = np.concatenate((features_test, np.array(test_x_featurized)), axis=1)

print("Finished Data preprocessing and feature engineering!")
print("Features shape"+str(features_train.shape))

save('features/features_train.npy', features_train)
save('features/features_test.npy', features_test)
save('features/train_y.npy',train_y)
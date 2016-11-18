

# import stuff and define RMSLE
import warnings
warnings.simplefilter("ignore")
​
import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().magic('pylab inline')
​
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, ARDRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
​
#!dir data
​
def rmsle(est, x, y):
    p = est.predict(x)
    y, p = np.log1p(y), np.log1p(p)
    rmsle = np.square(y-p)
    rmsle = rmsle.sum() / len(rmsle)
    return rmsle
​
​
# In[2]:
​
run_on_full = 1
# 0 = 50 k samples
# 1 = 2 mil samples
# > 1 = full train data
​
​
# In[3]:
​
# Data cleaning / Merging etc
if run_on_full == 0:
    print('Running on a smaller database.****')
    try:
        train = pd.read_csv('data/small_db.csv').sample(50000)
        print('Loading Small DB from disk !!!!', '')
    except:
        print('Loading entire DB and recreating Small DB!!!')
        train = pd.read_csv('data/train.csv')
        train.sample(n=2e6).to_csv('data/small_db.csv', index=False)# significant for 75 mil data points
elif run_on_full == 1:
    print('Running on significant sample.')
    try:
        train = pd.read_hdf('data/large_sample.hdf', key='large_sample')
    except:
        train = pd.read_csv('data/train.csv').sample(n=1e6)
        train.to_hdf('data/large_sample.hdf', key='large_sample')
else:
    print('Running on full database. Caution!!!')
    train = pd.read_csv('data/train.csv', chunksize=1000000)
    
cl = pd.read_csv('data/cliente_tabla.csv')
products = pd.read_csv('data/producto_tabla.csv')
town = pd.read_csv('data/town_state.csv')
print('Read the raw data')
​
print('Creating some more features on supporting data')
products['grams'] = products.NombreProducto.str.extract('.* (\d+)g.*', expand=False).astype(float)
products['ml'] = products.NombreProducto.str.extract('.* (\d+)ml.*', expand=False).astype(float)
products['inches'] = products.NombreProducto.str.extract('.* (\d+)in.*', expand=False).astype(float)
products['pct'] = products.NombreProducto.str.extract('.* (\d+)pct.*', expand=False).astype(float)
products['pieces'] = products.NombreProducto.str.extract('.* (\d+)p.*', expand=False).astype(float)
​
tn = pd.concat([town.Agencia_ID,
                pd.get_dummies(town.State),
                pd.get_dummies(town.Town)
               ],
              axis=1)
​
print('Merging the suporting data')
merged = pd.merge(train, cl, on='Cliente_ID', how='left')
merged = pd.merge(merged, products, on='Producto_ID', how='left')
merged = pd.merge(merged, tn, on='Agencia_ID', how='left')
merged.drop(['Agencia_ID', 'Producto_ID'], axis=1, inplace=True)
print('Merged everything into one thing')
​
​
print('Creating new features on merged data')
#TODO: Change to match outlier client IDs. Clients/Agencies which order too much
m, s = merged.Demanda_uni_equil.mean(), merged.Demanda_uni_equil.std()
merged['Outlier'] = 0
merged.loc[merged.Demanda_uni_equil > (m + (s)), 'Outlier'] = 1
merged.loc[merged.Demanda_uni_equil > (m + (2*s)), 'Outlier'] = 2
merged.loc[merged.Demanda_uni_equil > (m + (3*s)), 'Outlier'] = 3
​
# count zeros in the row
#merged['ZeroCount'] = (merged == 0).astype(int).sum(axis=1)
print('Created new features')
​
​
to_drop = ['Venta_uni_hoy',
           'Venta_hoy',
           'Dev_uni_proxima',
           'Dev_proxima',
          ]
merged.drop(to_drop + [ 'NombreCliente','NombreProducto'], axis=1, inplace=True)
train.drop(to_drop, axis=1, inplace=True)
​
merged.fillna(-1, inplace=True)
train.fillna(-1, inplace=True)
​
print('Complete')
​
​
# In[ ]:
​
# Model definition, Cross validation
def ts_CV(train, time_field, target_field, classifier, scorer, *, verbose=True):
    """
    Train is a DataFrame with features + timefield + target field
    time_field is the nameof the time field
    target field is the name of the target field
    
    classifier must provide fit and predict functions
    scorer must be pluggable into cross_val_score
    """
    time, scores = list(train[time_field].unique()), []
    time.sort()
    
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    vprint(time, 'Are the unique time indicators found')
​
    for index in range(len(time)):
        before, after = time[:index], time[index:]
        if len(before) < 1:
            continue
        vprint('Splitting at ', after[0], end='')
        if len(after) > 0:
            after, train_parts = after[0], set()  # just the first one
            
            for t in before:
                train_parts = train_parts.union(
                    set(train.loc[train[time_field] == t].index))
                
            tr = train.iloc[list(train_parts)]
            ts = train.loc[train[time_field] == after]
            
            tr_X, tr_Y = tr.drop(target_field, axis=1), tr[target_field]
            ts_X, ts_Y = ts.drop(target_field, axis=1), ts[target_field]
            vprint('*** Train/Test shape', tr_X.shape, ts_X.shape, end='')
​
            classifier.fit(tr_X, tr_Y)
            vprint('***Fit Complete***', end='')
            score = scorer(classifier, ts_X, ts_Y)
            vprint('*** Score =', score)
            scores.append(score)
    return np.array(scores)
​
​
clf = RandomForestRegressor(n_estimators=50,
                            n_jobs=-1,
                            #max_features='sqrt',
                            random_state=2016
                           ); print(clf, '\n'*2)  # 400 after Grid search
​
df = merged
​
scores = ts_CV(df, 'Semana', 'Demanda_uni_equil', clf, rmsle)
scores.mean(), scores.std()
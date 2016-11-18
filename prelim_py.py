
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%pylab inline

from sklearn.ensemble import RandomForestRegressor

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA

#!dir data

def rmsle_scorer(est, x, y):
    p = est.predict(x)
    y, p = np.log1p(y), np.log1p(p)
    rmsle = np.square(y-p)
    rmsle = rmsle.sum() / len(rmsle)
    return rmsle

try:
    train = pd.read_csv('data/small_db.csv').sample(10000)
    print('Loading Small DB from disk !!!!', '')
except:
    print('Loading entire DB and recreating Small DB!!!')
    train = pd.read_csv('data/train.csv')
    train.sample(n=2e6).to_csv('data/small_db.csv', index=False)# significant for 75 mil data points


print('Read the raw data')

cl = pd.read_csv('data/cliente_tabla.csv')
products = pd.read_csv('data/producto_tabla.csv')
town = pd.read_csv('data/town_state.csv')
#cl = pd.read_csv('data/clients_encoded.csv')
#pro = pd.read_csv('data/products_encoded.csv')
#town = pd.read_csv('data/town_encoded.csv')
print('Creating some more features')
products['grams'] = products.NombreProducto.str.extract('.* (\d+)g.*', expand=False).astype(float)
products['ml'] = products.NombreProducto.str.extract('.* (\d+)ml.*', expand=False).astype(float)
products['inches'] = products.NombreProducto.str.extract('.* (\d+)in.*', expand=False).astype(float)
products['pct'] = products.NombreProducto.str.extract('.* (\d+)pct.*', expand=False).astype(float)
products['pieces'] = products.NombreProducto.str.extract('.* (\d+)p.*', expand=False).astype(float)

tn = pd.concat([town.Agencia_ID,
                pd.get_dummies(town.State),
                pd.get_dummies(town.Town)
               ],
              axis=1)

print('Read the suporting data')
merged = pd.merge(train, cl, on='Cliente_ID', how='left')
merged = pd.merge(merged, products, on='Producto_ID', how='left')
merged = pd.merge(merged, tn, on='Agencia_ID', how='left')

print('Merged everything into one thing')
merged.drop(['Agencia_ID', 'Producto_ID'], axis=1, inplace=True)

to_drop = ['Venta_uni_hoy',
           'Venta_hoy',
           'Dev_uni_proxima',
           'Dev_proxima',
          ]
merged.drop(to_drop + [ 'NombreCliente','NombreProducto'], axis=1, inplace=True)
train.drop(to_drop, axis=1, inplace=True)

train.fillna(-1, inplace=True)

products.NombreProducto.head(n=10)

train.Demanda_uni_equil.hist()
train.loc[train.Demanda_uni_equil > 500]

merged.fillna(-1, inplace=True)
X, Y = merged.drop('Demanda_uni_equil', axis=1), merged.Demanda_uni_equil

clf = RandomForestRegressor(n_jobs=-1, n_estimators=400)  # 400 after Grid search

scores = cross_val_score(clf, 
                         X, Y,
                         cv=10,
                         scoring=rmsle_scorer
                        )

m, s = scores.mean(), scores.std()
m+s, m-s

train.drop(['Venta_uni_hoy',
            'Venta_hoy',
            'Dev_uni_proxima',
            'Dev_proxima'], axis=1, inplace=True)

X = train.drop(['Demanda_uni_equil'], axis=1)
Y = train[['Demanda_uni_equil']]

cross_val_score(pipe, X, Y, scoring='mean_squared_error', cv=10)

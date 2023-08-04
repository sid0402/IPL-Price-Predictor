import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneOut, StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from numpy import mean
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

df = pd.read_csv('ml/df_bar.csv')

df = df[df['Role_All-Rounder']==0]
#print(df)

#Make different dfs based on roles
def make_dfs(df):
    df_bar = df[df['Role_Batsman']==0]
    df_bar = df_bar[df_bar['Role_Wicket Keeper']==0]
    df_bar.to_csv("ml/df_bar.csv")

    df = df[df['Role_Bowler']==0]
    df = df[df['Role_All-Rounder']==0]

    return df

def drop_bowldf(df):
    bowl_col = ['overs', 'last_wkts', 'last_bowlavg', 'last_bowlsr',
       'last_bowler', 'last_bowlpercent', 'birthyear','Overs', 'Balls Bowled', 'Wickets',
       '3 Wickets in Innings', '4 Wickets in Innings', 'Economy Rate',
       'Strike Rate','prog_bowlsr', 'prog_er', 'prog_wkts', 'prog_bowlavg']
    df = df.drop(bowl_col, axis=1)
    return df

cat_columns = ['Role_All-Rounder', 'Role_Batsman',
       'Role_Bowler', 'Role_Wicket Keeper', 'Nation_Indian',
       'Nation_Overseas', 'Mega']

def skew_high(df):
    skew_col = []
    for i in df.columns:
        if (i != 'Amount' and not(i in cat_columns) and df[i].skew() > 1.5):
            skew_col.append(i)
    return skew_col

skew_col = skew_high(df)
print(skew_col)

def transform_log(df):
    for i in skew_col:
        df[i] = np.log(df[i]+0.5)
    return df

#df = transform_log(df)

def find_feat(df, thresh=0.85):
    col_corr = set()
    df_corr = df.corr()
    for i in range(len(df_corr.columns)):
        for j in range(i):
            if abs(df_corr.iloc[i,j] > thresh):
                col_corr.add(df_corr.columns[i])
    col_corr = list(col_corr)
    col_corr.append('Amount')
    df = df[col_corr]
    return df, col_corr
df, col_corr = find_feat(df, 0.80)
print(col_corr)
print(len(col_corr))

X = df.drop(['Amount'],axis=True)
Y = df['Amount']

X_feat = X

cat_columns = ['Role_All-Rounder', 'Role_Batsman',
       'Role_Bowler', 'Role_Wicket Keeper', 'Nation_Indian',
       'Nation_Overseas', 'Mega']

#scaling df
def scale_df(X, scaler):
    sc = scaler
    X.loc[:,~X.columns.isin(cat_columns)] = sc.fit_transform(X.loc[:,~X.columns.isin(cat_columns)])
    return X

X_feat = scale_df(X_feat,MinMaxScaler())

#FEATURE SELECTION
def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature
cor_support, cor_feature = cor_selector(X_feat,Y,15)
X_feat = X_feat[cor_feature]
print(cor_feature)

#Plotting multiple subplots
def plot_grid(col, row, X_feat, Y):
    fig, axs = plt.subplots(col, row, figsize=(15, 15))
    for i in range(col):
        for j in range(row):
            axs[i][j].scatter(X_feat[cor_feature[i+j]],Y)
            axs[i][j].set_title(cor_feature[i+j])
    plt.show()

#APPLYING PCA
def PCA_transform(X_feat, ncomp):
    pca = PCA(n_components = ncomp)
    X_feat = pca.fit_transform(X_feat)
    return X_feat
#X_feat = PCA_transform(X_feat, 7)

#PERMUTATION TEST FOR PCA
'''
def de_correlate_df(df):
    X_aux = df.copy()
    for col in df.columns:
        X_aux[col] = df[col].sample(len(df)).values
        
    return X_aux

pca = PCA()
pca.fit(X_feat)
original_variance = pca.explained_variance_ratio_

N_permutations = 1000
variance = np.zeros((N_permutations, len(X_feat.columns)))

for i in range(N_permutations):
    X_aux = de_correlate_df(X_feat)
    
    pca.fit(X_aux)
    variance[i, :] = pca.explained_variance_ratio_

p_val = np.sum(variance > original_variance, axis=0) / N_permutations
fig = go.Figure()
fig.add_trace(go.Scatter(x=[f'PC{i}' for i in range(len(df.columns))], y=p_val, name='p-value on significance'))
fig.update_layout(title="PCA Permutation Test p-values")
fig.show()
'''

#CROSS VALIDATION METHODS
def kfold(X_train,y_train, nfolds, model):
    predictor = model
    folds = KFold(n_splits = nfolds, shuffle = True, random_state = 100)
    scores = cross_val_score(predictor, X_train, y_train, scoring='neg_mean_absolute_error', cv=folds)
    preds = cross_val_predict(predictor, X_train, y_train, cv=folds)
    print(scores)
    return scores, preds
print((kfold(X_feat,Y,4,LinearRegression()))[0].mean())


def loocv(X_train, y_train, model):
    cv = LeaveOneOut()
    predictor = model
    scores = cross_val_score(predictor, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    y_pred = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=-1)
    return y_pred,scores
y_pred, scores = loocv(X_feat,Y,LinearRegression())
print(scores.mean())
#print(len(y_pred))
#print(len(X_feat))

df_pred = pd.DataFrame({'Y':Y,'Pred':y_pred})
df_pred['Difference'] = df_pred['Y'] - df_pred['Pred']
df_pred['Diff%'] = abs(df_pred['Difference']/df_pred['Y'])
df_pred = df_pred.sort_values(by='Diff%',ascending=False)
#print(df_pred)

print(r2_score(df_pred['Y'],df_pred['Pred']))

for i in range(len(df)):
    if (df.iloc[i].name==227):
        #print(df.iloc[i])
        #print(i)
        pass

#print(df_pred[df_pred['Y']>7.5])
plt.plot(df_pred.index, df_pred['Y'], label="Actual")
plt.plot(df_pred.index, df_pred['Pred'], label="Predicted")
plt.legend()
#plt.show()

#FINDING BEST MODEL
def best_model(X_train,y_train):
    models = [LinearRegression(), DecisionTreeRegressor(),
            RandomForestRegressor(n_estimators = 10, random_state = 0), SVR(C=10, gamma=1)]
    for model in models:
        print(str(model).split("(")[0])
        scores, preds = kfold(X_train, y_train, 4, model)
        print("Kfold: "+str(mean(scores)))
        y_pred, scores = loocv(X_train, y_train, model)
        print("LOOCV: "+str(mean(scores)))
        print()
    print("Polynomial Regression")
    for i in range(3):
        poly_reg = PolynomialFeatures(degree = i)
        X_poly = poly_reg.fit_transform(X_train)
        lin_reg = LinearRegression()
        print("degree: "+str(i))
        scores, preds = kfold(X_poly, y_train, 4, lin_reg)
        print("Kfold: "+str(mean(scores)))
        y_pred,scores = loocv(X_poly, y_train, lin_reg)
        print("LOOCV: "+str(mean(scores)))
#best_model(X_feat, Y)

#GridSearch
def gridsearch(X_train, y_train,nfolds=4):
    param_grid = {'C': [0.1, 1, 10, 100, 100], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'epsilon': [0.1, 0.25, 0.5, 0.75, 1],
              'kernel': ['rbf']}

    grid = GridSearchCV(SVR(), param_grid, cv=LeaveOneOut(), refit = True, verbose = 20, n_jobs=-1, scoring="neg_mean_absolute_error")
    grid.fit(X_train, y_train)

    print(grid.best_params_)
    print(grid.best_estimator_)

    print(pd.DataFrame(grid.cv_results_)[['mean_test_score']])
#gridsearch(X_feat, Y)

#MAKING THE MODEL
def make_model(X_train, y_train, model):
    predictor = model
    predictor.fit(X_train, y_train)
    X_train, X_test, y_train, y_test = train_test_split(X_feat, Y, test_size = 0.2, random_state = 0)
    y_pred = predictor.predict(X_test)
    print(mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred))
make_model(X_feat, Y, SVR(C=10,gamma=1))
import math
import numpy as np
import pandas as pd
import sys

from sklearn.cross_validation import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import log_loss

def train_test(clf, X, y):
    N = 10
    kf = StratifiedKFold(y, n_folds=N, shuffle=True, random_state=None)
    scores = cross_val_score(clf, X, y, cv=kf, scoring='log_loss', n_jobs=-1)
    print("log loss: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std()/math.sqrt(N)))
    sys.stdout.flush()

if __name__ == "__main__":
    file_in = 'datasets/numerai_training_data.csv'
    tp = pd.read_csv(file_in, iterator=True, chunksize=1000)
    X = pd.concat(tp, ignore_index=True)
    y = X['target']
    X.drop( 'target', axis = 1, inplace=True)
    rState = None

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=rState)

    clf = ExtraTreesClassifier(n_estimators=500, max_depth=8, n_jobs=-1) 
    etc = Pipeline([('poly_features', PolynomialFeatures(2)),('new_clf', clf)])

    train_test(etc, Xtrain, ytrain)

    etc.fit(Xtrain, ytrain)
    p = etc.predict_proba(Xtest)
    print log_loss(ytest, p)

    etc.fit(X, y)

    # predict
    file_in = 'datasets/numerai_tournament_data.csv'
    file_out = 'predictions.csv'
    tp = pd.read_csv(file_in, iterator=True, chunksize=1000)
    X = pd.concat(tp, ignore_index=True)
    p = etc.predict_proba(X.drop( 't_id', axis = 1 ))
    X['probability'] = p[:,1]
    X.to_csv( file_out, columns = ( 't_id', 'probability' ), index = None )


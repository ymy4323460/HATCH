# -*- coding: utf-8 -*-

import pandas as pd, numpy as np, warnings
from scipy import optimize

from util import _LinUCBnTSSingle
from joblib import Parallel, delayed


class HATCH:
    def __init__(self, nchoices, gmm, J, pai, B, t, contextdic, context_dim, alpha=1.0, njobs=1):
        self.pai = np.array(pai)
        self.budget = B
        self.t = t
        self.B = B
        self.time = t
        self.contextdic = contextdic
        self.gmm = gmm
        self._add_common_lin(alpha, nchoices, njobs, J, context_dim)
        self.ustar = [0 for i in range(J)]

    def _add_common_lin(self, alpha, nchoices, njobs, J, context_dim):
        if isinstance(alpha, int):
            alpha = float(alpha)
        assert isinstance(alpha, float)

        self.njobs = njobs
        self.alpha = alpha
        self.nchoices = nchoices
        self.J = J
        self._oraclesa = [[_LinUCBnTSSingle(1.0, context_dim) for n in range(nchoices)]for i in range(J)]
        self._oraclesj = [_LinUCBnTSSingle(1.0, context_dim) for n in range(J)]
        self.uj = np.array([float(1) for i in range(self.J)])
        
    def set_time_budget(self, t,b):
        self.budget = self.B - b
        self.t = self.time - t
        
        
    def _get_ALP_predict(self):
        c = np.multiply(self.uj, self.pai.T)
        A = np.array([self.pai])
        b = np.array([float(self.budget)/float(self.t)])
        bound = [(0,1) for i in range(self.J)]
        assert b.shape[0] == 1
        res = optimize.linprog(-c,A,b,bounds = bound)

        return res
    
    def _get_expect_ALP_predict(self):
        c = np.multiply(self.uj, self.pai.T)
        A = np.array([self.pai])
        b = np.array([float(self.B)/float(self.time)])
        bound = [(0,1) for i in range(self.J)]
        assert b.shape[0] == 1
        res = optimize.linprog(-c,A,b,bounds = bound)

        return res
    
    def fit(self, X, a, r):
        self.ndim = X.shape[1]
        Xj = np.array(list(map(lambda x:self.contextdic[x], list(self.gmm.predict(X)))))

        Parallel(n_jobs=self.njobs, verbose = 0, require="sharedmem")(delayed(self._fit_single)(j, X, Xj, a, r) for j in range(self.J))
        
        for j in range(self.J):
            self.uj[j] = self._oraclesj[j].predict(np.array([self.contextdic[j]]), exploit = True)
            
            
        return self

    def _fit_single(self, j, X, Xj, a, r):
        xj = self.gmm.predict(X)
        this_context = xj == j
        self._oraclesj[j].fit(Xj[this_context, :], r[this_context].astype('float64'))
        for choice in range(self.nchoices):
            this_action = a == choice
            self._oraclesa[j][choice].fit(X[this_action, :], r[this_action].astype('float64'))
                
    def partial_fit(self, X, a, r):
        context = self.gmm.predict(X)
        this_context = a == choice
        self._oraclesj[choice].fit(X[this_context, :], r[this_context].astype('float64'))
        
        X, a, r = _check_fit_input(X, a, r, self.choice_names)
        for n in range(self.nchoices):
            this_action = a == n
            self._oracles[n].partial_fit(X[this_action, :], r[this_action].astype('float64'))
            
        return self
    
        
    def predict(self, X, a, exploit=True, output_score=False):
        pred = np.zeros(self.nchoices)
        j = self.gmm.predict(X)    
        for choice in range(self.nchoices):
            pred[choice] = self._oraclesa[j[0]][choice].predict(X,exploit = True)
        score_max = np.max(np.array([pred]), axis=1)
        pred = np.argmax(np.array([pred]), axis=1)
        
        if pred == a:
            pred = pred
        else:
            pred = -2
            return pred

        ust = pred
            
        if self.budget > 0 :
            acc_percentage = self._get_ALP_predict()
            if np.random.uniform(0, 1) <= acc_percentage['x'][j]:
                pred = pred
            else:
                pred = -1

            if pred > 0:
                self.budget -= 1
            self.t -= 1
            if np.random.uniform(0, 1) <= self._get_expect_ALP_predict()['x'][j]:
                ust = -1
        else:
            pred = -1
        if not output_score:
            return pred
        else:
            return pred, score_max

    def get_remain_budget(self):
        return (self.budget,(self.budget/self.t))
    
    def _predict(self, j, choice, pred, exploit, X):
        if exploit:
            pred[:, j, choice] = self._oraclesa[j][choice].exploit(X)
        else:
            pred[:, choice] = self._oraclesa[j][choice].predict(X)
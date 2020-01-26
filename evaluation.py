# -*- coding: utf-8 -*-

import pandas as pd, numpy as np

def evaluateRejectionSampling(gmm, policy, X, a, r,num, online=True, start_point_online=0, batch_size=10, pretrain = False,rho = 0.375):
    
    x_class = gmm.predict(X)
    interval = 500
    if start_point_online=='random':
        start_point_online=np.random.randint(X.shape[0])
    else:
        if isinstance(start_point_online, int):
            pass
        elif isinstance(start_point_online, float):
            pass
        else:
            raise ValueError("'start_point_online' must be one of 'random', float [0,1] or int [0, sample_size]")
    
    if not online:
        pred=policy.predict(X)
        match=pred==a
        return (np.mean(r[match]), match.sum())
    else:
        cum_r=0
        cum_n=0
        b = 0
        ix_chosen=list()
        all_chosen = list()
        rr = 0.0
        for i in range(start_point_online, X.shape[0]):
            obs=X[i,:].reshape(1,-1)
            would_choose=policy.predict(obs,a[i])
            policy.set_time_budget(cum_n, b)
            
            if would_choose != -2:
                rr += r[i]
                
            if would_choose == -1:
                cum_r = cum_r
                cum_n+=1
                all_chosen.append(i)
                    
            if would_choose==a[i]:
                cum_r+=r[i]
                b += 1
                cum_n+=1
                ix_chosen.append(i)
                all_chosen.append(i)
                if (cum_n%batch_size)==0:
                    ix_fit=np.array(ix_chosen)
                    policy.fit(X[ix_fit,:], a[ix_fit], r[ix_fit])
                    
            if would_choose!= -2 and cum_n%1000 == 0 and cum_n!= 0:
                print ((cum_r/cum_n), cum_n)
                print('true', (rr/cum_n)*rho)
                print(policy.get_remain_budget())
                
            if cum_n >= num:
                break
        for i in range(0, start_point_online):
            obs=X[i,:].reshape(1,-1)
            would_choose=policy.predict(obs)
            print(would_choose)
            b = 0
            policy.set_time_budget(cum_n, b)
            if would_choose == -1:
                cum_r = cum_r
                cum_n+=1
                if cum_n % interval == 0:
                    print (cum_r/cum_n, cum_n)
                    
            if would_choose==a[i]:
                cum_r+=r[i]
                print(a[i])
                b += 1
                cum_n+=1
                ix_chosen.append(i)
                if (cum_n%batch_size)==0:
                    ix_fit=np.array(ix_chosen)
                    policy.fit(X[ix_fit,:], a[ix_fit], r[ix_fit])
                if cum_n % interval == 0:
                    print (cum_r/cum_n, cum_n)
                    
        if cum_n==0:
            raise ValueError("Rejection sampling couldn't obtain any matching samples.")
            
        ix_allfit=np.array(all_chosen)
        return (cum_r/cum_n, cum_n,x_class[ix_fit],x_class[ix_allfit], r[ix_fit])
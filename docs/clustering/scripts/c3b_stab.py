import numpy as np, json
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score as ARI
z=np.load('../labels.npz'); X=z['X']; m=z['m']; idx=np.where(m)[0]
ev=np.random.default_rng(99).choice(idx,40000,replace=False); out={}
for K in range(2,9):
    P=[]
    for s in range(4):
        sb=np.random.default_rng(100+s).choice(idx,25000,replace=False); P.append(GaussianMixture(K,covariance_type='full',n_init=2,random_state=s).fit(X[sb]).predict(X[ev]))
    a=[ARI(P[i],P[j]) for i in range(4) for j in range(i+1,4)]; out[K]=(float(np.mean(a)),float(np.min(a))); print(K,out[K])
json.dump(out,open('../facts_stab.json','w'))

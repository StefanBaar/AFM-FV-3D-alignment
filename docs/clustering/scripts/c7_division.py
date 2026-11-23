"""Mechanics around cell division: parent track before, daughters after (confirmed divisions), aligned on the division frame."""
import numpy as np, csv, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
d=dict(np.load('../pix.npz')); z=np.load('../labels.npz'); X=z['X']; m=z['m']; fr=d['frame']; nf=int(fr.max())+1; feats=list(z['feats'])
names={'logE':'log E','alpha':'α','dstar':'δ*','adh1':'adh. strength','adh2':'adh. short vs long'}
tt=np.array([d['t_h'][fr==k][0] for k in range(nf)])*60
# per (frame,track) medians
cm={}
for k in range(nf):
    s=m&(fr==k)&(d['cell']>0)
    for t in np.unique(d['cell'][s]).astype(int):
        q=s&(d['cell']==t)
        if q.sum()>=4: cm[(k,t)]=np.median(X[q],0)
T=list(csv.DictReader(open('/home/claude/seg2/run3/tracks.csv')))
kids={}
for r in T:
    if r['division']=='confirmed' and r['parent']!='0': kids.setdefault(int(r['parent']),[]).append((int(r['track']),int(r['first_frame'])))
rel=[];val=[]
for p,ch in kids.items():
    kd=min(f for _,f in ch)                                   # first frame of the daughters
    for k in range(max(0,kd-8),kd):
        if (k,p) in cm: rel.append(tt[k]-tt[kd]); val.append(cm[(k,p)])
    for c,f0 in ch:
        for k in range(f0,min(nf,f0+8)):
            if (k,c) in cm: rel.append(tt[k]-tt[kd]); val.append(cm[(k,c)])
rel=np.array(rel); val=np.array(val)
bins=np.array([-60,-40,-25,-12,0,12,25,40,60]); cen=(bins[1:]+bins[:-1])/2; out={'n_divisions':len(kids),'n_obs':len(rel),'bins_min':bins.tolist(),'median':{},'sem':{},'n':[]}
for i,f in enumerate(feats):
    md=[];se=[]
    for b in range(len(bins)-1):
        s=(rel>=bins[b])&(rel<bins[b+1]); md.append(float(np.median(val[s,i])) if s.sum()>3 else np.nan); se.append(float(1.2533*val[s,i].std()/np.sqrt(max(s.sum(),1))))
    out['median'][f]=md; out['sem'][f]=se
out['n']=[int(((rel>=bins[b])&(rel<bins[b+1])).sum()) for b in range(len(bins)-1)]
# simple test: last obs before vs first after (paired by division)
from scipy.stats import wilcoxon
pre=[];post=[]
for p,ch in kids.items():
    kd=min(f for _,f in ch); a=[cm[(k,p)] for k in range(max(0,kd-3),kd) if (k,p) in cm]; b=[cm[(k,c)] for c,f0 in ch for k in range(f0,min(nf,f0+3)) if (k,c) in cm]
    if a and b: pre.append(np.mean(a,0)); post.append(np.mean(b,0))
pre=np.array(pre);post=np.array(post); out['paired_n']=len(pre)
out['paired']={f:dict(pre=float(np.median(pre[:,i])),post=float(np.median(post[:,i])),p=float(wilcoxon(pre[:,i],post[:,i]).pvalue)) for i,f in enumerate(feats)}
json.dump(out,open('../facts_division.json','w'),indent=1); print(json.dumps({k:out[k] for k in ['n_divisions','n_obs','n','paired_n','paired']},indent=1))
fig,ax=plt.subplots(figsize=(5,3))
for i,f in enumerate(feats): ax.errorbar(cen+i*0.8-1.6,out['median'][f],out['sem'][f],marker='o',ms=3,label=names[f])
ax.axvline(0,c='k',lw=.6,ls=':');ax.axhline(0,c='gray',lw=.5);ax.set_xlabel('time relative to division (min)\n(left: mother cell, right: daughters)');ax.set_ylabel('cell median (robust z)');ax.legend(frameon=False,fontsize=7,ncol=2)
plt.savefig('../fig/division.pdf');plt.close()

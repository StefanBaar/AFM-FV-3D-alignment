"""Replot the profile and cell-identity bar charts from the facts files (label layout only)."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
feats=['logE','alpha','dstar','adh1','adh2']; names={'logE':'log E','alpha':'α','dstar':'δ*','adh1':'adhesion\nstrength','adh2':'adhesion\nshort vs long'}
c=json.load(open('../facts_cluster.json')); prof=c['profiles']; K=len(prof); cols=plt.get_cmap('tab10').colors
fig,ax=plt.subplots(figsize=(5.6,3)); w=0.8/K
for j in range(K): ax.bar(np.arange(5)+(j-(K-1)/2)*w,[prof[j]['z'][f] for f in feats],w,color=cols[j],label=f'C{j} ({100*prof[j]["frac"]:.0f}%)')
ax.axhline(0,c='k',lw=.6);ax.set_xticks(range(5),[names[f] for f in feats]);ax.set_ylabel('median (robust z, per frame)');ax.legend(frameon=False,fontsize=7,ncol=5,loc='upper center',bbox_to_anchor=(.5,1.15));plt.savefig('../fig/profiles.pdf');plt.close()
d=json.load(open('../facts_cells.json')); xs=np.arange(5)
fig,ax=plt.subplots(figsize=(4.6,2.9))
ax.bar(xs-.2,[d['ICC'][f] for f in feats],.4,label='ICC (variance share between cells)');ax.bar(xs+.2,[d['persist_lag'][f] for f in feats],.4,label='r, same cell, next observation')
ax.set_xticks(xs,[names[f] for f in feats],fontsize=8);ax.set_ylim(0,1.15);ax.legend(frameon=False,fontsize=7,loc='upper center',ncol=2,bbox_to_anchor=(.5,1.12));plt.savefig('../fig/cell_identity.pdf');plt.close()

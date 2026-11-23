import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.size':9,'savefig.bbox':'tight'})
d=dict(np.load('../pix.npz')); z=np.load('../labels.npz'); L=z['L']; fr=d['frame']; K=int(L.max())+1
cols=plt.get_cmap('tab10').colors[:K]; cm=ListedColormap(cols)
NAMES=json.load(open('../cluster_names.json'))
def grid(k,v):
    s=fr==k; ny=int(d['iy'][s].max())+1; nx=int(d['ix'][s].max())+1; return v[s].reshape(ny,nx)
sel=[0,8,16,24,32,40]
fig,ax=plt.subplots(2,6,figsize=(12,5.6))
for c,k in enumerate(sel):
    G=grid(k,L).astype(float); G[G<0]=np.nan; C=grid(k,d['cell'])
    ny,nx=G.shape; ext=[0,nx*1.0135,0,ny*0.5]
    E=grid(k,d['logE']).astype(float); E[grid(k,d['excl'])>0]=np.nan
    ax[0,c].imshow(E,origin='lower',extent=ext,cmap='viridis',vmin=np.nanpercentile(E,2),vmax=np.nanpercentile(E,98),interpolation='nearest')
    ax[1,c].imshow(G,origin='lower',extent=ext,cmap=cm,vmin=-.5,vmax=K-.5,interpolation='nearest')
    for a in ax[:,c]: a.set_xticks([]);a.set_yticks([])
    ax[0,c].set_title(f'{d["t_h"][fr==k][0]:.1f} h',fontsize=9)
ax[0,0].set_ylabel('log E (1 ms)');ax[1,0].set_ylabel('cluster')
h=[plt.Rectangle((0,0),1,1,color=cols[j]) for j in range(K)]
fig.legend(h,[f'C{j} {NAMES[j]}' for j in range(K)],loc='lower center',ncol=K,frameon=False,fontsize=8,bbox_to_anchor=(.5,-.02))
plt.tight_layout(rect=(0,.05,1,1));plt.savefig('../fig/cluster_maps.pdf',dpi=200);plt.close()
# 3D registered: three frames
fig=plt.figure(figsize=(12,4))
for c,k in enumerate([0,24,40]):
    a=fig.add_subplot(1,3,c+1,projection='3d'); s=(fr==k)&(L>=0)
    a.scatter(d['X'][s],d['Y'][s],d['Z'][s],c=[cols[j] for j in L[s]],s=3,lw=0,depthshade=False,rasterized=True)
    a.set_title(f'{d["t_h"][fr==k][0]:.1f} h (registered to frame 1)',fontsize=9); a.view_init(28,-60); a.set_box_aspect((1,1,.6))
    a.set_xlabel('x (µm)',fontsize=7);a.set_ylabel('y (µm)',fontsize=7);a.set_zlabel('z (µm)',fontsize=7);a.tick_params(labelsize=6)
fig.legend(h,[f'C{j} {NAMES[j]}' for j in range(K)],loc='lower center',ncol=K,frameon=False,fontsize=8)
plt.tight_layout(rect=(0,.06,1,1));plt.savefig('../fig/cluster_3d.pdf',dpi=200);plt.close()
# all frames stacked in registered coordinates (occupancy of the tether + stiff clusters)
fig,ax=plt.subplots(1,K,figsize=(13,2.8))
m=L>=0; X,Y=d['X'][m],d['Y'][m]; xb=np.linspace(np.percentile(X,1),np.percentile(X,99),45); yb=np.linspace(np.percentile(Y,1),np.percentile(Y,99),45)
Ntot,_,_=np.histogram2d(X,Y,[xb,yb])
for j in range(K):
    Nj,_,_=np.histogram2d(X[L[m]==j],Y[L[m]==j],[xb,yb]); P=np.where(Ntot>20,Nj/np.maximum(Ntot,1),np.nan)
    im=ax[j].imshow(P.T,origin='lower',extent=[xb[0],xb[-1],yb[0],yb[-1]],cmap='magma',vmin=0,vmax=np.nanpercentile(P,98),aspect='auto');ax[j].set_title(f'C{j} {NAMES[j]}',fontsize=8);plt.colorbar(im,ax=ax[j],fraction=.046)
    ax[j].set_xlabel('X (µm)',fontsize=7)
ax[0].set_ylabel('Y (µm)'); plt.tight_layout(); plt.savefig('../fig/occupancy_registered.pdf'); plt.close()

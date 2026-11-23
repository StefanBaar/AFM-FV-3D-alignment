import numpy as np, json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi
import trace as T, segment_cells as S
plt.rcParams.update({'font.size':9,'axes.titlesize':9.5,'figure.dpi':100,'savefig.bbox':'tight','pdf.fonttype':42})
fr=T.frames(); k=[i for i,f in enumerate(fr) if f['time']=='19:00:35'][0]; f=fr[k]; o=T.trace(f)
px,py,H=f['px'],f['py'],T.H; ny,nx=f['v']['topo'].shape
extN=[-px/2,(nx-.5)*px,-py/2,(ny-.5)*py]
def extI(A): return [-H/2,(A.shape[1]-.5)*H*((nx-1)*px/((A.shape[1]-1)*H)),-H/2,(A.shape[0]-.5)*H*((ny-1)*py/((A.shape[0]-1)*H))]
def show(a,A,ext,cmap='gray',title='',p=(2,98),cb=None):
    im=a.imshow(A,origin='lower',extent=ext,cmap=cmap,vmin=np.nanpercentile(A,p[0]),vmax=np.nanpercentile(A,p[1]),interpolation='nearest',aspect='equal')
    a.set_title(title); a.set_xlabel('x (µm)'); a.set_ylabel('y (µm)')
    if cb: plt.colorbar(im,ax=a,fraction=.046,pad=.03,label=cb)
    return im
facts=dict(frame=f['time'],k=k,nx=nx,ny=ny,px=px,py=py,n_cells=int(o['lab'].max()),n_seeds=int(o['seeds'].max()),
           dropped=[(d[1],round(float(d[2]),2),round(float(d[3]),3)) for d in o['dropped']],
           spikes={n:int(r['n_spikes']) for n,r in o['ch'].items()},sig_y=o['ch']['adh_force_nN']['sig_y_px'],iso_shape=list(o['B'].shape))
# ---- F1 channels
v=f['v']
fig,ax=plt.subplots(1,5,figsize=(15,3.1))
show(ax[0],v['adh_force_nN'],extN,'magma','adhesion force',cb='nN')
show(ax[1],v['adh_work_fJ'],extN,'magma','work of adhesion',cb='fJ')
show(ax[2],np.log10(v['E']),extN,'viridis','log$_{10}$ E',cb='log$_{10}$ Pa')
show(ax[3],v['delta'],extN,'plasma','indentation δ',cb='µm')
show(ax[4],v['topo'],extN,'cividis','height',cb='µm')
for a in ax[1:]: a.set_ylabel('')
plt.tight_layout(); plt.savefig('fig/channels.pdf'); plt.close()
# ---- F2 preprocessing (adhesion force)
r=o['ch']['adh_force_nN']
fig,ax=plt.subplots(2,3,figsize=(13,7.4))
show(ax[0,0],r['raw'],extN,'magma','1  raw map (45 × 75 px, 1.01 × 0.50 µm)',cb='nN')
show(ax[0,1],r['despiked'],extN,'magma',f'2  outlier filter: {r["n_spikes"]} px replaced (cyan)',cb='nN')
yy,xx=np.nonzero(r['spike_mask']); ax[0,1].plot(xx*px,yy*py,'s',mfc='none',mec='cyan',ms=3.5,mew=.8)
show(ax[0,2],r['destriped'],extN,'magma','3  scan-line offsets removed',cb='nN')
ins=ax[0,2].inset_axes([1.32,0.0,0.32,1.0]); ins.plot(r['line_offset'],np.arange(ny)*py,c='k',lw=.8); ins.set_title('offset per line',fontsize=8); ins.set_xlabel('nN',fontsize=8); ins.tick_params(labelsize=7); ins.set_ylim(extN[2],extN[3])
show(ax[1,0],r['iso'],extI(r['iso']),'magma','4  cubic resampling to 0.25 × 0.25 µm',cb='nN')
show(ax[1,1],r['yeq'],extI(r['iso']),'magma',f'5  Gaussian in y, σ$_y$ = {r["sig_y_px"]*H:.2f} µm',cb='nN')
show(ax[1,2],r['z'],extI(r['iso']),'magma','6  minus Gaussian(σ = 4 µm), robust z',cb='robust SD')
plt.tight_layout(); plt.savefig('fig/preprocess.pdf'); plt.close()
# ---- F3 1D ridge profile across junctions
Z=r['z']; row=int(round(20.0/H)); xs=np.arange(Z.shape[1])*H*((nx-1)*px/((Z.shape[1]-1)*H))
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
prof=Z[row]
fig,ax=plt.subplots(2,1,figsize=(7,4.6),sharex=True)
ax[0].plot(xs,prof,c='k',lw=1.2); ax[0].set_ylabel('z-scored\nadhesion force'); ax[0].axhline(0,c='grey',lw=.5)
ax[0].set_title(f'profile along y = {row*H:.1f} µm (step 6 map): junctions = narrow bright maxima')
for s_um,c in [(0.32,'#d62728'),(0.72,'#1f77b4')]:
    sg=s_um/H; Hm=hessian_matrix(-np.nan_to_num(Z),sg,mode='nearest',use_gaussian_derivatives=True); ev=hessian_matrix_eigvals(Hm)
    ax[1].plot(xs,sg**2*np.maximum(ev[0],0)[row],c=c,lw=1.1,label=f'σ = {s_um} µm: σ²·max(−λ$_{{min}}$, 0)')
ax[1].plot(xs,r['sato'][row],c='k',lw=1.5,ls='--',label='Sato response = max over σ')
ax[1].legend(frameon=False,fontsize=8); ax[1].set_xlabel('x (µm)'); ax[1].set_ylabel('ridge strength')
plt.tight_layout(); plt.savefig('fig/ridge1d.pdf'); plt.close()
facts['ridge_row_um']=row*H
# ---- F4 ridge maps
fig,ax=plt.subplots(1,5,figsize=(16,3.3))
show(ax[0],r['z'],extI(Z),'magma','input (step 6)')
show(ax[1],r['sato_s1'],extI(Z),'gray','Sato σ = 0.32 µm',p=(1,99.5))
show(ax[2],r['sato_s2'],extI(Z),'gray','Sato σ = 0.72 µm',p=(1,99.5))
show(ax[3],r['ridge_z'],extI(Z),'gray','max over σ, robust z (force)',p=(1,99.5))
show(ax[4],o['B'],extI(Z),'gray','combined B = ½ force + ½ work',p=(1,99.5))
for a in ax[1:]: a.set_ylabel('')
plt.tight_layout(); plt.savefig('fig/ridge2d.pdf'); plt.close()
# ---- F5 watershed
Bs=o['Bs']; e=extI(Bs)
def edges(L):
    return (L>=0)&((np.roll(L,1,0)!=L)|(np.roll(L,1,1)!=L)|(np.roll(L,-1,0)!=L)|(np.roll(L,-1,1)!=L))
fig,ax=plt.subplots(1,4,figsize=(16,3.6))
show(ax[0],Bs,e,'gray','boundary map B$_s$ + seeds (h-minima, h = 0.1)',p=(1,99.5))
sy_,sx_=np.nonzero(o['seeds']); sc=(nx-1)*px/((Bs.shape[1]-1)*H)
ax[0].plot(sx_*H*sc,sy_*H*((ny-1)*py/((Bs.shape[0]-1)*H)),'.',c='#00d0ff',ms=1.5)
show(ax[1],Bs,e,'gray',f'watershed: {o["ws"].max()} basins',p=(1,99.5))
rg=np.zeros(Bs.shape+(4,)); rg[(o['ws']==0)&o['vi']]=[1,.15,.1,1]; ax[1].imshow(rg,origin='lower',extent=e,interpolation='nearest')
show(ax[2],Bs,e,'gray','dropped: area < 2.4 µm² (red), contrast < 0.2 (orange)',p=(1,99.5))
rg=np.zeros(Bs.shape+(4,))
for lb,why,a_,c_ in o['dropped']: rg[o['ws']==lb]=[1,0,0,.75] if why=='area' else [1,.6,0,.75]
ax[2].imshow(rg,origin='lower',extent=e,interpolation='nearest')
bg=S.to_iso(S.fill_nan(v['adh_force_nN']),px,py,H,order=3); bg=bg-ndi.gaussian_filter(bg,16)
show(ax[3],np.where(o['vi'],bg,np.nan),e,'gray',f'result: {o["lab"].max()} cells on the adhesion force map')
rg=np.zeros(Bs.shape+(4,)); rg[edges(np.where(o['lab']>0,o['lab'],-1))|((o['lab']==0)&o['vi'])]=[1,.15,.1,1]; ax[3].imshow(rg,origin='lower',extent=e,interpolation='nearest')
for a in ax[1:]: a.set_ylabel('')
plt.tight_layout(); plt.savefig('fig/watershed.pdf',dpi=200); plt.close()
# ---- F6 h-minima in 1D on a real profile of Bs
from skimage.morphology import reconstruction
rowb=int(round(15.0/H)); p_=Bs[rowb]; xb=np.arange(len(p_))*H*sc
hh=0.1*4  # shown with a larger h for visibility? keep the real h
for hval,name in [(0.1,'hmin1d.pdf')]:
    rec=reconstruction(p_+hval,p_,method='erosion'); mins=(rec-p_)>=hval-1e-9
    fig,a=plt.subplots(figsize=(7,2.8))
    a.plot(xb,p_,c='k',lw=1.3,label='B$_s$ along y = %.1f µm'%(rowb*H*((ny-1)*py/((Bs.shape[0]-1)*H))))
    a.plot(xb,rec,c='#d62728',lw=1.2,ls='--',label='R = reconstruction by erosion of B$_s$ + h')
    lab1,n1=ndi.label(mins)
    for j in range(1,n1+1):
        ii=np.nonzero(lab1==j)[0]; a.plot(xb[ii],p_[ii],'o',c='#00a0d0',ms=7,alpha=.85,label='seed = minimum with depth ≥ h' if j==1 else None)
    a.set_xlim(0,21.3); vis=xb<=21.3; a.set_ylim(p_[vis].min()-0.12,p_[vis].max()+0.12); a.set_xlabel('x (µm)'); a.set_ylabel('B$_s$ (robust units)'); a.legend(frameon=False,fontsize=8,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.25))
    plt.tight_layout(); plt.savefig('fig/'+name); plt.close()
json.dump(facts,open('facts1.json','w'),indent=1,default=float); print(facts)

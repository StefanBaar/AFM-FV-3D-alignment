import sys; sys.path.insert(0,'/home/claude/out')
import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.signal import savgol_filter
import extract_topo_E as X
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
BLUE,ORANGE,RED,DRED,GREEN,TEAL,GREY='#2a6fbb','#e08a1e','#d62839','#7a0010','#2a9d55','#17a2b8','#888888'
z=np.load('/home/claude/test/full/home_claude_data.npz'); P=json.loads(str(z['params_json'])); k=P['k']*1e3
F=z['curves_F_nN'].astype(float); Z=z['curves_Z_um'].astype(float); t=z['curves_t_s']
ci=z['curve_index_map'].ravel().astype(int); noise=np.empty(len(F)); noise[ci]=z['noise_nN'].ravel()
facts={}
def curve(i):
    r,d=X.adhesion_curve(F[i],Z[i],k,noise[i],details=True); return r,d
def save(name): plt.savefig(f'fig/{name}.pdf',bbox_inches='tight'); plt.close()

# ---------------------------------------------------------------- 1. full curve (where the retract part is)
i=103; r,d=curve(i); ip=int(np.argmax(Z[i]))
fig,ax=plt.subplots(1,2,figsize=(10.5,3.6))
a=ax[0]; a.plot(t[:ip+1]*1e3,F[i,:ip+1],c=BLUE,lw=1.2,label='approach'); a.plot(t[ip:]*1e3,F[i,ip:],c=ORANGE,lw=1.2,label='retract')
a.axhline(0,c='k',lw=.6); a.axvline(t[ip]*1e3,c=GREY,ls='--',lw=.8); a.text(t[ip]*1e3+0.4,a.get_ylim()[1]*0.85 if False else 4.2,'Z turnaround',color=GREY,fontsize=9)
a.set_xlabel('time (ms)'); a.set_ylabel('force F (nN)'); a.legend(frameon=False,loc='upper left'); a.set_title('one force curve vs time',fontsize=10)
a=ax[1]; zz=Z[i]-Z[i,ip]; a.plot(zz[:ip+1],F[i,:ip+1],c=BLUE,lw=1.2); a.plot(zz[ip:],F[i,ip:],c=ORANGE,lw=1.2); a.axhline(0,c='k',lw=.6)
a.fill_between([-9,0],-1.2,0,color=ORANGE,alpha=.08); a.text(-8.6,-1.05,'adhesion analysis:\nretract part, F < 0',color=ORANGE,fontsize=9)
a.set_xlabel('stage position Z − Z$_{max}$ (µm)'); a.set_ylabel('force F (nN)'); a.set_title('same curve vs Z',fontsize=10); a.set_xlim(-9.2,0.6)
save('full_curve')
facts['ex_full']=i

# ---------------------------------------------------------------- 2. overview (annotated retract curve)
i=103; r,d=curve(i); s,Fr,j0,jm,jl,thr=d['s'],d['F'],d['j0'],d['jm'],d['jl'],d['thr']
x=s; s0=s[j0]
fig,a=plt.subplots(figsize=(10.5,5.2))
seg=slice(j0,jl+2); a.fill_between(x[seg],np.minimum(Fr[seg],0),0,color=ORANGE,alpha=.22,lw=0)
a.plot(x,Fr,c='k',lw=1.1,zorder=3); a.axhline(0,c='k',lw=.6)
a.axhline(-thr,c=GREEN,ls=':',lw=1.2)
a.plot([s0],[0],'o',c=BLUE,ms=7,zorder=5); a.annotate('zero-force point s$_0$ (first retract point with F < 0,\nstart of the tensile part; tip still in adhesive contact)',(s0,0),(0.9,0.17),color=BLUE,fontsize=9,arrowprops=dict(arrowstyle='->',color=BLUE))
a.plot([x[jm]],[Fr[jm]],'o',c=RED,ms=7,zorder=5)
a.annotate('',(x[jm],Fr[jm]),(x[jm],0),arrowprops=dict(arrowstyle='<->',color=RED,lw=1.6))
a.annotate(f'adhesion force F$_{{adh}}$ = {r["adh_force_nN"]:.2f} nN\n= −(minimum force)',(x[jm],Fr[jm]*0.55),(3.0,-0.62),color=RED,fontsize=10,arrowprops=dict(arrowstyle='->',color=RED,lw=.8))
yb=Fr[jm]-0.09; a.annotate('',(s0,yb),(x[jm],yb),arrowprops=dict(arrowstyle='<->',color=TEAL,lw=1.4)); a.text((s0+x[jm])/2,yb-0.03,f'pull-off distance {r["adh_pulloff_um"]:.2f} µm',color=TEAL,ha='center',va='top',fontsize=9)
yl=Fr[jm]-0.22; a.annotate('',(s0,yl),(x[jl],yl),arrowprops=dict(arrowstyle='<->',color=GREEN,lw=1.4)); a.text((s0+x[jl])/2,yl-0.03,f'detachment length {r["adh_length_um"]:.2f} µm (s$_0$ → last point below −threshold)',color=GREEN,ha='center',va='top',fontsize=9)
a.plot([x[jl]],[Fr[jl]],'s',c=GREEN,ms=6,zorder=5)
a.annotate(f'work of adhesion W = {r["adh_work_fJ"]:.2f} fJ\n= shaded area ∫ (−F) ds',(1.05,-0.3),(3.0,-0.40),color='#b5651d',fontsize=10,arrowprops=dict(arrowstyle='->',color='#b5651d',lw=.8))
pa,pb=d['plateaus'][-1]; a.plot(x[pa:pb+1],np.full(pb-pa+1,-r['adh_tether_nN']),c='#8e44ad',lw=3,alpha=.8)
a.annotate(f'tether plateau {r["adh_tether_nN"]:.3f} nN over {r["adh_tether_len_um"]:.2f} µm',((x[pa]+x[pb])/2,-r['adh_tether_nN']),(3.0,-0.24),color='#8e44ad',fontsize=9.5,arrowprops=dict(arrowstyle='->',color='#8e44ad',lw=.8))
e,h=d['events'][-1]; a.annotate('',(x[e]+0.03,Fr[e+1]+0.005),(x[e]+0.03,Fr[e]),arrowprops=dict(arrowstyle='->',color=DRED,lw=1.6))
a.annotate(f'rupture step {h:.3f} nN at {x[e]-s0:.2f} µm from s$_0$',(x[e]+0.03,Fr[e]-0.01),(5.4,-0.16),color=DRED,fontsize=9.5,arrowprops=dict(arrowstyle='->',color=DRED,lw=.8))
a.text(6.4,-0.22,f'dotted: −threshold = −{thr:.3f} nN',color=GREEN,fontsize=9,va='top')
a.set_xlabel('tip retract distance s (µm), s = 0 at the Z turnaround'); a.set_ylabel('force F (nN)')
a.set_ylim(Fr[jm]-0.36,0.3); a.set_xlim(-0.1,x[-1])
a.set_title(f'retract part of curve #{i} (data_213626)',fontsize=10)
save('overview')
facts['ov']=dict(i=i,**{k_:float(v) for k_,v in r.items() if isinstance(v,(float,int)) and np.isfinite(v)},thr=float(thr),s0=float(s0),
                 last_rupt_pos=float(x[e]-s0))

# ---------------------------------------------------------------- 3. tip position vs Z (magnitude)
i=103; ip=int(np.argmax(Z[i])); zz=Z[i,ip]-Z[i,ip:]; ss=d['s']
fig,ax=plt.subplots(1,2,figsize=(10.5,3.4))
a=ax[0]; a.plot(zz,F[i,ip:],c=GREY,lw=1.4,label='vs stage travel Z$_{max}$ − Z'); a.plot(ss,F[i,ip:],c='k',lw=1,ls='--',label='vs tip travel s')
a.axhline(0,c='k',lw=.5); a.legend(frameon=False,fontsize=9); a.set_xlabel('travel (µm)'); a.set_ylabel('F (nN)'); a.set_xlim(-0.2,3)
a.set_title('retract part, first 3 µm',fontsize=10)
a=ax[1]; a.plot(zz,(ss-zz)*1e3,c=BLUE); a.axhline(0,c='k',lw=.5); a.set_xlabel('stage travel Z$_{max}$ − Z (µm)'); a.set_ylabel('s − (Z$_{max}$ − Z)  (nm)')
a.set_title('difference = change of the cantilever deflection',fontsize=10)
save('tip_vs_z')
facts['defl_nm_at_Fadh']=float(r['adh_force_nN']/k*1e3); facts['defl_nm_peak']=float(F[i,ip]/k*1e3)

# ---------------------------------------------------------------- 4. noise threshold
i=103; fig,ax=plt.subplots(1,2,figsize=(10.5,3.3))
a=ax[0]; nb=600//8; a.plot(t[:110]*1e3,F[i,:110],c=GREY,lw=.8); a.plot(t[:nb]*1e3,F[i,:nb],c=BLUE,lw=1.1); a.axhline(0,c='k',lw=.5)
a.axvspan(0,t[nb-1]*1e3,color=BLUE,alpha=.07); a.text(0.2,0.085,'baseline window: first 600 samples (5.4 ms)',color=BLUE,fontsize=8.5)
sg=noise[i]; a.fill_between([0,t[nb-1]*1e3],-sg,sg,color=BLUE,alpha=.18,label=f'±σ = ±{sg*1e3:.1f} pN (this curve)')
a.axhline(-max(5*sg,0.03),c=GREEN,ls=':',label=f'−threshold = −max(5σ, 0.03) = −{max(5*sg,0.03):.3f} nN'); a.legend(frameon=False,fontsize=8,loc='lower right')
a.set_xlabel('time (ms)'); a.set_ylabel('F (nN)'); a.set_title('start of the approach (tip far from the sample)',fontsize=10); a.set_ylim(-0.1,0.1)
a=ax[1]; nz=z['noise_nN'].ravel()*1e3; a.hist(nz,bins=40,color=BLUE,alpha=.75); a.axvline(np.median(nz),c='k',ls='--',lw=1)
a.set_xlabel('baseline noise σ per curve (pN)'); a.set_ylabel('curves'); a.set_title(f'all 2775 curves: median σ = {np.median(nz):.1f} pN',fontsize=10)
save('noise')
facts['sigma_med_pN']=float(np.median(nz)); facts['thr_med']=float(np.median(np.maximum(5*z['noise_nN'],0.03)))

# ---------------------------------------------------------------- 5. detachment: detached vs still attached
fig,ax=plt.subplots(1,2,figsize=(10.5,3.6))
for a,i,tt in ((ax[0],65,'detached before the end (flag = 1)'),(ax[1],1,'still attached at the end (flag = 0)')):
    r_,d_=curve(i); s_,F_,j0_,jl_,thr_=d_['s'],d_['F'],d_['j0'],d_['jl'],d_['thr']
    a.plot(s_,F_,c=GREY,lw=.8,label='force'); a.plot(s_,savgol_filter(F_,7,2),c='k',lw=1.1,label='smoothed (7 points)')
    a.axhline(0,c='k',lw=.5); a.axhline(-thr_,c=GREEN,ls=':',lw=1.2,label='−threshold')
    a.axvline(s_[j0_],c=BLUE,ls=':',lw=1); a.axvline(s_[jl_],c=GREEN,ls='--',lw=1.2)
    a.annotate('',(s_[j0_],0.12),(s_[jl_],0.12),arrowprops=dict(arrowstyle='<->',color=GREEN)); a.text((s_[j0_]+s_[jl_])/2,0.15,f'{r_["adh_length_um"]:.2f} µm',color=GREEN,ha='center',fontsize=9)
    a.set_ylim(-0.4,0.25); a.set_xlim(-0.1,s_[-1]+0.05); a.set_xlabel('tip retract distance s (µm)'); a.set_ylabel('F (nN)'); a.set_title(f'#{i}: {tt}',fontsize=10)
    if i==65: a.legend(frameon=False,fontsize=8,loc='lower right')
    facts[f'det_{i}']=dict(L=r_['adh_length_um'],det=r_['adh_detached'])
save('detachment')

# ---------------------------------------------------------------- 6. work of adhesion
i=103; r,d=curve(i); s,Fr,j0,jm,jl=d['s'],d['F'],d['j0'],d['jm'],d['jl']
fig,a=plt.subplots(figsize=(7.8,3.6))
seg=slice(j0,jl+2); a.fill_between(s[seg],np.maximum(-Fr[seg],0),0,color=ORANGE,alpha=.35,lw=0,step=None)
a.plot(s,-Fr,c='k',lw=1); a.axhline(0,c='k',lw=.5)
# trapezoid illustration
for q in range(jm+30,jm+33): a.plot([s[q],s[q],s[q+1],s[q+1]],[0,-Fr[q],-Fr[q+1],0],c=RED,lw=.8)
a.annotate('one trapezoid:\n½ (−F$_n$ − F$_{n+1}$) (s$_{n+1}$ − s$_n$)',(s[jm+31],-Fr[jm+31]*0.5),(s[jm]+1.4,0.75),fontsize=9,color=RED,arrowprops=dict(arrowstyle='->',color=RED))
a.text(s[jl],0.05,'s$_{end}$',color=GREEN,ha='center'); a.text(s[j0],0.05,'s$_0$',color=BLUE,ha='center')
a.set_xlabel('tip retract distance s (µm)'); a.set_ylabel('−F (nN)  (tensile force)'); a.set_xlim(-0.1,s[-1]); a.set_ylim(-0.05,0.95)
a.set_title(f'curve #{i}: W = {r["adh_work_fJ"]:.2f} fJ',fontsize=10)
save('work')

# ---------------------------------------------------------------- 7. rupture detection windows
def fitwin(Fr,e,hw=10):
    x=np.arange(-hw,hw+1.0); Xm=np.c_[np.ones_like(x),x,x*x,(x>0).astype(float)]; y=Fr[e-hw:e+hw+1]
    XtXi=np.linalg.inv(Xm.T@Xm); c=XtXi@(Xm.T@y); res=y-Xm@c; sig=np.sqrt(res@res/(len(y)-4)); se=np.sqrt(XtXi[3,3])
    cq=np.linalg.lstsq(Xm[:,:3],y,rcond=None)[0]; return x,y,Xm@c,Xm[:,:3]@cq,c[3],sig,se
fig,ax=plt.subplots(1,3,figsize=(11.5,3.4))
cases=[(0,None,'clear step (tether rupture)'),(125,None,'step on the steep unbinding slope'),(125,'smooth','smooth unbinding (no step)')]
for a,(i,mode,tt) in zip(ax,cases):
    r_,d_=curve(i); Fr_=d_['F']; e=d_['events'][-1][0]
    if mode=='smooth': e=e-14
    x,y,fs,fq,h,sig,se=fitwin(Fr_,e)
    sige=max(sig,noise[i]); ok=h>=d_['thr'] and h>=6*sige*se
    a.plot(x,y,'o',c='k',ms=3.5,label='data (72 µs/point)'); a.plot(x,fq,'--',c=GREEN,lw=1.4,label='quadratic only'); a.plot(x,fs,'-',c=RED,lw=1.8,label='quadratic + step')
    a.set_title(f'#{i}: {tt}',fontsize=10); a.set_xlabel('sample relative to candidate'); a.set_ylabel('F (nN)')
    a.text(0.03,0.97,f'h = {h:.3f} nN\nthreshold = {d_["thr"]:.3f} nN\n6·SE(h) = {6*sige*se:.3f} nN\n→ {"accepted" if ok else "rejected"}',transform=a.transAxes,va='top',fontsize=8.5,
           color=DRED if ok else GREY,bbox=dict(fc='white',ec='none',alpha=.8))
    facts[f'rupt_{i}_{mode}']=dict(h=float(h),thr=float(d_['thr']),six_se=float(6*sige*se),ok=bool(ok))
ax[0].legend(frameon=False,fontsize=8,loc='lower right')
save('rupture_fit')

# ---------------------------------------------------------------- 8. tether plateaus
fig,ax=plt.subplots(1,2,figsize=(10.5,3.6))
for a,i,tt in ((ax[0],103,'plateau ends with a rupture → detached'),(ax[1],1,'plateau lasts until the end → still attached')):
    r_,d_=curve(i); s_,F_=d_['s'],d_['F']; Fw=savgol_filter(F_,31,2)
    a.plot(s_,F_,c=GREY,lw=.7,label='force'); a.plot(s_,Fw,c='k',lw=1.2,label='smoothed (31 points ≈ 1 µm)')
    a.axhline(0,c='k',lw=.5)
    thp=max(3*noise[i],0.03); a.axhline(-thp,c=GREEN,ls=':',lw=1.1,label=f'−max(3σ, 0.03 nN)')
    for pa,pb in d_['plateaus']: a.axvspan(s_[pa],s_[pb],color='#8e44ad',alpha=.18)
    pa,pb=d_['plateaus'][-1]; a.hlines(-r_['adh_tether_nN'],s_[pa],s_[pb],color='#8e44ad',lw=2.5)
    a.text((s_[pa]+s_[pb])/2,0.06,f'tether: {r_["adh_tether_nN"]:.3f} nN, {r_["adh_tether_len_um"]:.2f} µm',ha='center',color='#8e44ad',fontsize=9,bbox=dict(fc='white',ec='none',alpha=.85))
    a.set_ylim(-0.55,0.22); a.set_xlim(-0.1,s_[-1]+0.05); a.set_xlabel('tip retract distance s (µm)'); a.set_ylabel('F (nN)'); a.set_title(f'#{i}: {tt}',fontsize=10)
    if i==103: a.legend(frameon=False,fontsize=8,loc='lower right')
    facts[f'teth_{i}']=dict(T=r_['adh_tether_nN'],L=r_['adh_tether_len_um'],det=r_['adh_detached'])
save('tether')

# ---------------------------------------------------------------- 9. maps
M=dict(np.load('M.npz')); ext=[0,37.5,0,37.5]
panels=[('adh_force_nN','adhesion force (nN)','magma'),('adh_work_fJ','work of adhesion (fJ)','magma'),('adh_gamma_JKR_mJ_m2','JKR w (mJ/m²)','magma'),
        ('adh_pulloff_um','pull-off distance (µm)','viridis'),('adh_length_um','detachment length (µm)','viridis'),('adh_detached','detached (1) / attached (0)','Greys'),
        ('adh_n_ruptures','rupture events','Blues'),('adh_tether_nN','tether force (nN)','PuRd'),('adh_tether_len_um','tether length (µm)','PuRd')]
fig,ax=plt.subplots(3,3,figsize=(10.5,9.6))
for a,(key,tt,cm) in zip(ax.ravel(),panels):
    m=M[key]; lo,hi=(0,max(1,np.nanmax(m))) if key in('adh_n_ruptures','adh_detached') else np.nanpercentile(m,[1,99])
    h=a.imshow(m,origin='lower',extent=ext,cmap=cm,vmin=lo,vmax=hi,interpolation='nearest'); a.set_title(tt,fontsize=10); plt.colorbar(h,ax=a,shrink=.78)
    a.set_xticks([0,10,20,30]); a.set_yticks([0,10,20,30]); a.tick_params(labelsize=8)
plt.tight_layout(); save('maps')
json.dump(facts,open('facts.json','w'),indent=1,default=float)
print(json.dumps(facts,indent=1,default=float))

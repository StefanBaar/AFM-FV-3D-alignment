import numpy as np, json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
import common as C
AP, P = C.AP, C.P
plt.rcParams.update({'font.size':9,'axes.titlesize':9.5,'savefig.bbox':'tight','pdf.fonttype':42})
BL,OR,GR,RD,PU='#2a6fbb','#e07b39','#2a9d55','#d62839','#8e44ad'
facts={}
i=103; a=C.raw(i); na,nr,dt=P['num_app'],P['num_ret'],P['dt']
# ---------- T1 raw signals -> force
d=np.r_[a[:na],a[na:na+nr]].astype(float); z=np.r_[a[na+nr:2*na+nr],a[2*na+nr:]].astype(float)
traw=np.arange(len(d))*dt*1e3
t,s,F,zz,sig=AP.prep(a,P,C.B)
fig,ax=plt.subplots(1,3,figsize=(13.5,3.3))
ax[0].plot(traw,z,c='k',lw=1); ax[0].set_ylabel('Z signal (V)'); ax[0].set_xlabel('time (ms)'); ax[0].axvline(na*dt*1e3,c='grey',ls=':')
ax[0].set_title('raw Z (approach | retract)'); ax2=ax[0].twinx(); ax2.plot(traw,z*P['zsens'],alpha=0); ax2.set_ylabel('Z (µm) = V × 25 µm/V')
ax[1].plot(traw,d,c='#999',lw=.5,label='raw deflection (9 µs)'); ax[1].set_xlabel('time (ms)'); ax[1].set_ylabel('deflection (V)')
ax[1].axvspan(0,P['baseline']*dt*1e3,color=BL,alpha=.12,label='baseline window (600 samples)'); ax[1].legend(frameon=False,fontsize=7.5); ax[1].set_title('raw deflection')
ax[2].plot(t*1e3,F,c=BL,lw=1); ax[2].axhline(0,c='k',lw=.5); ax[2].set_xlabel('time (ms)'); ax[2].set_ylabel('F (nN)')
ax[2].set_title('force: drag-corrected, zeroed, block-averaged ×8 (72 µs)')
plt.tight_layout(); plt.savefig('fig/raw_to_force.pdf'); plt.close()
# drag
v=savgol_filter(z,P['savgol'],3,deriv=1)*P['zsens']/dt
facts['v_app']=float(np.median(v[na-200:na])); facts['v_ret']=float(np.median(v[-300:]))
facts['drag_pN_s_um']=float(C.Z['drag_pN_s_per_um']); facts['drag_force_app_pN']=facts['drag_pN_s_um']*abs(facts['v_app'])
fig,ax=plt.subplots(figsize=(6.8,3.0))
sm=lambda x: np.convolve(x,np.ones(64)/64,'same')
d0=(d-np.median(d[:600]))*P['invols']*P['k']*1e3
dc=d-C.B*v; dc=(dc-np.median(dc[:600]))*P['invols']*P['k']*1e3
ax.plot(traw,sm(d0),c='#999',lw=1,label='before: zeroed on the approach baseline')
ax.plot(traw,sm(dc),c=BL,lw=1,label='after: d − b·v$_Z$, zeroed')
ax.axvspan(0,600*dt*1e3,color=BL,alpha=.08); ax.axvspan(traw[-300],traw[-1],color=OR,alpha=.12)
ax.text(1,0.27,'approach baseline\n(v$_Z$ ≈ +494 µm/s)',fontsize=7.5,color=BL); ax.text(traw[-300]-15,0.27,'retract baseline\n(v$_Z$ ≈ −471 µm/s)',fontsize=7.5,color=OR)
ax.set_ylim(-0.45,0.4); ax.axhline(0,c='k',lw=.5); ax.set_xlabel('time (ms)'); ax.set_ylabel('F (nN), averaged over 64 samples'); ax.legend(frameon=False,fontsize=7.5,loc='lower left')
ax.set_title('hydrodynamic drag shifts the baseline with the direction of motion',fontsize=9)
plt.tight_layout(); plt.savefig('fig/drag.pdf'); plt.close()
# ---------- fit internals
ip=int(np.argmax(F)); Fs=savgol_filter(F,9,2); thr=max(4*sig,0.03*Fs[ip]); lo=np.where(Fs[:ip]<thr)[0]; ic=lo[-1]
neg=np.where(Fs[ip:]<0)[0]; i_end=ip+neg[0]
best=None
for dz_ in (0.0,-0.4):
    x0=[0.3,s[ic]+dz_]
    r=least_squares(AP.ting_model,x0,args=(t,s,F,i_end,P),bounds=([0.0,s[ic]-4],[0.95,s[ic]+1.5]),diff_step=[1e-2,2e-3/max(abs(x0[1]),1)],x_scale=[0.1,0.1])
    if best is None or r.cost<best.cost: best=r
E1,Fm,dl=AP.ting_model(best.x,t,s,F,i_end,P,True)
al,zc=best.x
facts.update(E1=float(E1*1e3),alpha=float(al),zc=float(zc),thr=float(thr),ic=int(ic),ip=int(ip),i_end=int(i_end),sigma=float(sig),
             dmax=float(dl.max()),Fmax=float(F[ip]),t_contact_ms=float((t[i_end]-t[np.where(dl>0)[0][0]])*1e3),
             t_load_ms=float((t[np.argmax(dl)]-t[np.where(dl>0)[0][0]])*1e3),E2ms=float(E1*1e3*(0.002)**(-al)),E10ms=float(E1*1e3*(0.01)**(-al)))
cc=slice(max(ic-10,0),i_end); res=(Fm-F)[cc]; facts['R2']=float(1-np.sum(res**2)/np.sum((F[cc]-F[cc].mean())**2))
# T8: fit in time and vs indentation
fig,ax=plt.subplots(1,3,figsize=(14,3.6))
tt=t*1e3; i0=np.where(dl>0)[0][0]
ax[0].plot(tt,F,c='#888',lw=.9,label='data'); ax[0].plot(tt[:i_end],Fm[:i_end],c=RD,lw=1.6,label='Ting model (fitted window)'); ax[0].plot(tt[i_end:],Fm[i_end:],c=RD,lw=1,ls=':',label='model, outside window')
ax[0].axvspan(tt[0],tt[i_end],color=GR,alpha=.06); ax[0].axvline(tt[i_end],c=GR,ls='--',lw=.8); ax[0].text(tt[i_end]+.3,F.max()*.9,'end of window:\nfirst F < 0\nafter the peak',fontsize=7.5,color=GR)
ax[0].set_xlabel('time (ms)'); ax[0].set_ylabel('F (nN)'); ax[0].legend(frameon=False,fontsize=7.5,loc='upper left'); ax[0].set_title('fit in time (approach + retract)')
ax[0].set_xlim(tt[i0]-8,tt[i_end]+6)
ax[1].plot(dl[:ip+1],F[:ip+1],c=BL,lw=.9,label='approach'); ax[1].plot(dl[ip:i_end],F[ip:i_end],c=OR,lw=.9,label='retract')
ax[1].plot(dl[:i_end],Fm[:i_end],c=RD,lw=1.4,ls='--',label='model'); ax[1].set_xlabel('indentation δ = s − z$_c$ (µm)'); ax[1].set_ylabel('F (nN)'); ax[1].legend(frameon=False,fontsize=8)
ax[1].set_title('same fit vs indentation: hysteresis = viscoelasticity')
ax[2].plot(tt,dl,c='k'); ax[2].set_xlabel('time (ms)'); ax[2].set_ylabel('δ (µm)'); ax[2].set_xlim(tt[i0]-8,tt[i_end]+6); ax[2].set_title('indentation history δ(t) (input of the model)')
ax[2].axvline(tt[np.argmax(dl)],c='grey',ls=':',lw=.8); ax[2].text(tt[np.argmax(dl)]+.3,dl.max()*.5,'t$_m$: max. δ',fontsize=8)
plt.tight_layout(); plt.savefig('fig/fit103.pdf'); plt.close()
# contact point detail
fig,ax=plt.subplots(figsize=(6.4,3.0))
j=slice(ic-120,ip+5)
ax.plot(s[j]-s[ic],F[j],c='#888',lw=.9,label='F'); ax.plot(s[j]-s[ic],Fs[j],c=BL,lw=1,label='smoothed (9 points)')
ax.axhline(thr,c=GR,ls=':',label=f'threshold max(4σ, 3 % F$_{{max}}$) = {thr:.3f} nN'); ax.axvline(0,c=GR,lw=.8)
ax.axvline(zc-s[ic],c=RD,lw=1.2,label=f'fitted contact z$_c$ ({(zc-s[ic])*1e3:+.0f} nm from start)')
ax.plot(s[j]-s[ic],Fm[j],c=RD,lw=1,ls='--')
ax.set_xlim(-1.5,1.5); ax.set_ylim(-.1,1.0); ax.set_xlabel('tip position s − s(start) (µm)'); ax.set_ylabel('F (nN)'); ax.legend(frameon=False,fontsize=7.5)
ax.set_title('contact point: start value from a threshold, final value from the fit',fontsize=9)
plt.tight_layout(); plt.savefig('fig/contact.pdf'); plt.close()
# ---------- T4 contact laws
R=3.0; dd=np.linspace(0,3,300)
fig,ax=plt.subplots(figsize=(5.6,3.2))
for g,c,l in [('sphere',BL,'sphere, exact (Sneddon)'),('paraboloid',OR,'paraboloid (Hertz)'),('cone',GR,'cone, 17.5° half angle')]:
    Pg=dict(P,geometry=g); ax.plot(dd,AP.phi(dd,Pg),c=c,label=l)
ax.set_xlabel('δ (µm)'); ax.set_ylabel('φ(δ) = F(1−ν²)/E  (µm²)'); ax.legend(frameon=False,fontsize=8); ax.set_title('elastic indentation law, R = 3 µm')
ax.axvline(facts['dmax'],c='grey',ls=':'); ax.text(facts['dmax']+.03,0.5,'δ$_{max}$ #103',fontsize=7.5,color='grey')
plt.tight_layout(); plt.savefig('fig/contactlaws.pdf'); plt.close()
Ps=dict(P,geometry='sphere'); Ph=dict(P,geometry='paraboloid')
facts['sphere_vs_hertz_at_dmax']=float(AP.phi(np.array([facts['dmax']]),Ps)[0]/AP.phi(np.array([facts['dmax']]),Ph)[0])
# ---------- T5 power law
tl=np.logspace(-4,1,200)
fig,ax=plt.subplots(figsize=(5.6,3.2))
for a_,c in zip([0.1,0.3,0.5,0.7],[BL,GR,OR,RD]):
    ax.loglog(tl,20*(tl/1)**(-a_),c=c,label=f'α = {a_}')
ax.axvspan(1e-3,3e-2,color='grey',alpha=.12); ax.text(1.15e-3,2.2,'time scale of\none indentation',fontsize=7.5)
ax.plot([1],[20],'ko'); ax.annotate('E$_1$ = E(t$_0$ = 1 s)',(1,20),(0.05,3),fontsize=8,arrowprops=dict(arrowstyle='->'))
ax.set_xlabel('time t (s)'); ax.set_ylabel('relaxation modulus E(t) (Pa)'); ax.legend(frameon=False,fontsize=8); ax.set_title('E(t) = E$_1$ (t/t$_0$)$^{-α}$, all with E$_1$ = 20 Pa')
plt.tight_layout(); plt.savefig('fig/powerlaw.pdf'); plt.close()
# ---------- T6 simulated triangle: effect of alpha
dts=72e-6; N=600; tri=np.r_[np.linspace(0,2,N//2),np.linspace(2,0,N//2)]
fig,ax=plt.subplots(1,2,figsize=(11,3.4))
for a_,c in zip([0.0,0.2,0.5,0.8],[BL,GR,OR,RD]):
    g=AP.ting_shape(tri,a_,dts,P)*(200*0.01**a_)/1e3        # E1 chosen so that E(10 ms) = 200 Pa
    ax[0].plot(np.arange(N)*dts*1e3,g,c=c,label=f'α = {a_}'); ax[1].plot(tri,g,c=c,label=f'α = {a_}')
ax[0].set_xlabel('time (ms)'); ax[0].set_ylabel('F (nN)'); ax[0].set_title('triangular δ(t): 2 µm in 21.6 ms and back; all with E(10 ms) = 200 Pa'); ax[0].legend(frameon=False,fontsize=8)
ax[1].set_xlabel('δ (µm)'); ax[1].set_ylabel('F (nN)'); ax[1].set_title('same vs δ: hysteresis grows with α; force reaches 0 before δ = 0')
plt.tight_layout(); plt.savefig('fig/alpha_effect.pdf'); plt.close()
# t1(t) for alpha=0.5 numerically: solve int_{t1}^{t} E(t-tau) ddelta = 0
a_=0.5; tt_=np.arange(N)*dts; dd_=np.diff(tri); im=np.argmax(tri)
w=AP.kernel_weights(a_,N,dts,1.0)
t1=[]
for k in range(im+1,N):
    acc=0.0; j1=None
    for j in range(k-1,-1,-1):
        acc+=w[k-1-j]*dd_[j]
        if acc>=0: j1=j; break
    t1.append(tt_[j1] if j1 is not None else np.nan)
fig,ax=plt.subplots(figsize=(5.4,3.0))
ax.plot(tt_[im+1:]*1e3,np.array(t1)*1e3,c=OR,label='t$_1$(t), α = 0.5'); ax.plot(tt_[im+1:]*1e3,2*tt_[im]*1e3-tt_[im+1:]*1e3,c='grey',ls='--',label='elastic (α = 0): mirror time')
ax.set_xlabel('time t during unloading (ms)'); ax.set_ylabel('t$_1$ (ms)'); ax.legend(frameon=False,fontsize=8); ax.set_title('Ting: the contact radius at t equals that at t$_1$')
plt.tight_layout(); plt.savefig('fig/t1.pdf'); plt.close()
# ---------- T9 cost landscape alpha x zc
A=np.linspace(0.3,0.7,31); ZC=np.linspace(zc-0.4,zc+0.4,31); Cst=np.zeros((len(ZC),len(A))); E1g=np.zeros_like(Cst)
for ii,zz_ in enumerate(ZC):
    for jj,aa in enumerate(A):
        r_=AP.ting_model([aa,zz_],t,s,F,i_end,P); Cst[ii,jj]=0.5*np.sum(r_**2)
        E1g[ii,jj]=AP.ting_model([aa,zz_],t,s,F,i_end,P,True)[0]*1e3
fig,ax=plt.subplots(1,3,figsize=(16,3.8))
im_=ax[0].contourf(A,(ZC-zc)*1e3,np.log10(Cst),20,cmap='viridis'); plt.colorbar(im_,ax=ax[0],label='log$_{10}$ cost ½Σr² (nN²)')
ax[0].plot(al,0,'r*',ms=12); ax[0].set_xlabel('α'); ax[0].set_ylabel('z$_c$ − fitted z$_c$ (nm)'); ax[0].set_title('least-squares cost (E$_1$ solved exactly at every point)')
im2=ax[1].contourf(A,(ZC-zc)*1e3,E1g,20,cmap='magma'); plt.colorbar(im2,ax=ax[1],label='best E$_1$ (Pa)'); ax[1].plot(al,0,'c*',ms=12)
ax[1].set_xlabel('α'); ax[1].set_title('E$_1$ (at t$_0$ = 1 s) that goes with each (α, z$_c$)')
E10=E1g*0.01**(-A[None,:])
im3=ax[2].contourf(A,(ZC-zc)*1e3,E10,20,cmap='magma'); plt.colorbar(im3,ax=ax[2],label='E(10 ms) (Pa)'); ax[2].plot(al,0,'c*',ms=12)
ax[2].set_xlabel('α'); ax[2].set_title('E(10 ms) for the same points: depends much less on α')
plt.tight_layout(); plt.savefig('fig/landscape.pdf'); plt.close()
# ---------- T10 parameter effects on the real curve
fig,ax=plt.subplots(1,3,figsize=(14,3.4))
for f_,c in [(0.5,BL),(1,'k'),(2,RD)]:
    ax[0].plot(tt[i0-20:i_end+40],(Fm*f_)[i0-20:i_end+40],c=c,label=f'E$_1$ × {f_}')
for aa,c in [(al-0.15,BL),(al,'k'),(al+0.15,RD)]:
    E_,Fm_,_=AP.ting_model([aa,zc],t,s,F,i_end,P,True); Fm_=Fm_/E_*E1*0.01**(aa-al)     # same E(10 ms)
    ax[1].plot(tt[i0-20:i_end+40],Fm_[i0-20:i_end+40],c=c,label=f'α = {aa:.2f} (same E(10 ms))')
for dz_,c in [(-0.3,BL),(0,'k'),(0.3,RD)]:
    E_,Fm_,_=AP.ting_model([al,zc+dz_],t,s,F,i_end,P,True); Fm_=Fm_/max(E_,1e-12)*E1
    ax[2].plot(tt[i0-20:i_end+40],Fm_[i0-20:i_end+40],c=c,label=f'z$_c$ {dz_*1e3:+.0f} nm')
for a__,ttl in zip(ax,['E$_1$: scales the whole curve','α: shape (relaxation, hysteresis)','contact point: onset and depth']):
    a__.plot(tt[i0-20:i_end+40],F[i0-20:i_end+40],c='#bbb',lw=.8,zorder=0); a__.set_title(ttl); a__.legend(frameon=False,fontsize=7.5); a__.set_xlabel('time (ms)'); a__.set_ylabel('F (nN)')
plt.tight_layout(); plt.savefig('fig/param_effects.pdf'); plt.close()
json.dump(facts,open('facts.json','w'),indent=1); print(json.dumps(facts,indent=1))
j0=np.argmin(np.abs(ZC-zc))
facts['landscape']=dict(E1_range=[float(E1g[j0,0]),float(E1g[j0,-1])],E10_range=[float(E10[j0,0]),float(E10[j0,-1])],alpha_range=[float(A[0]),float(A[-1])])
json.dump(facts,open('facts.json','w'),indent=1); print(facts['landscape'])

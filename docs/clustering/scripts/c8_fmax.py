"""Peak-force dependence of the adhesion parameters on the full map data_213626."""
import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
plt.rcParams.update({'font.size':9,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
F=np.load('/home/claude/fullv/home_claude_data.npz',allow_pickle=True)
ok=(F['R2']>0.9)&np.isfinite(F['E_at_1ms_Pa'])&np.isfinite(F['adh_force_nN'])&np.isfinite(F['Fmax_nN'])
g=lambda k:F[k][ok]; Fm=g('Fmax_nN'); out={'Fmax_pct':np.percentile(Fm,[5,50,95]).tolist()}
for k in ['adh_force_nN','adh_work_fJ','adh_length_um','adh_pulloff_um','adh_n_ruptures','adh_tether_nN','delta_max_um','alpha']:
    out['rho_'+k+'_Fmax']=float(spearmanr(g(k),Fm).statistic)
ratio=g('adh_force_nN')/Fm; out['ratio_pct']=np.percentile(ratio,[5,50,95]).tolist()
out['rho_ratio_logE']=float(spearmanr(ratio,np.log10(g('E_at_1ms_Pa'))).statistic); out['rho_ratio_alpha']=float(spearmanr(ratio,g('alpha')).statistic)
out['rho_Fmax_logE']=float(spearmanr(Fm,g('E_at_1ms_Pa')).statistic); out['rho_Fmax_slope']=float(spearmanr(Fm,g('slope_deg')).statistic)
A=np.column_stack([np.ones(ok.sum()),Fm]); c=np.linalg.lstsq(A,g('adh_force_nN'),rcond=None)[0]; out['R2_Fadh_Fmax']=float(1-np.var(g('adh_force_nN')-A@c)/np.var(g('adh_force_nN'))); out['Fadh_vs_Fmax_coef']=c.tolist()
json.dump(out,open('../facts_fmax.json','w'),indent=1)
fig,ax=plt.subplots(1,3,figsize=(9.5,2.9))
sc=ax[0].scatter(np.log10(g('E_at_1ms_Pa')),Fm,s=3,c=g('slope_deg'),cmap='viridis',lw=0,rasterized=True);ax[0].set_xlabel(r'$\log_{10}E_{1\,ms}$ (Pa)');ax[0].set_ylabel(r'peak force $F_\max$ (nN)');plt.colorbar(sc,ax=ax[0],label='slope (°)')
ax[1].scatter(Fm,g('adh_force_nN'),s=3,c='k',lw=0,alpha=.5,rasterized=True);xx=np.linspace(0,Fm.max(),5);ax[1].plot(xx,c[0]+c[1]*xx,'C3');ax[1].set_xlabel(r'$F_\max$ (nN)');ax[1].set_ylabel(r'$F_\mathrm{adh}$ (nN)');ax[1].set_title(f'ρ = {out["rho_adh_force_nN_Fmax"]:.2f}, $R^2$ = {out["R2_Fadh_Fmax"]:.2f}',fontsize=9)
ax[2].scatter(np.log10(g('E_at_1ms_Pa')),ratio,s=3,c='k',lw=0,alpha=.5,rasterized=True);ax[2].set_xlabel(r'$\log_{10}E_{1\,ms}$');ax[2].set_ylabel(r'$F_\mathrm{adh}/F_\max$');ax[2].set_title(f'ρ = {out["rho_ratio_logE"]:.2f}',fontsize=9)
plt.tight_layout();plt.savefig('../fig/fmax_confound.pdf',dpi=220)

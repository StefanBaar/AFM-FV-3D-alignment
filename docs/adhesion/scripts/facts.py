import sys; sys.path.insert(0,'/home/claude/out')
import numpy as np, json
import extract_topo_E as X
z=np.load('/home/claude/test/full/home_claude_data.npz'); P=json.loads(str(z['params_json']))
print('params:',{k:P[k] for k in ('dt','block','k','invols','radius','geometry','baseline','savgol','zsens','num_app','num_ret')})
F=z['curves_F_nN'].astype(float); Z=z['curves_Z_um'].astype(float); t=z['curves_t_s']
print('decimated step %.1f us, points/curve %d'%((t[1]-t[0])*1e6,len(t)))
ip=np.argmax(Z,1); print('Z turnaround index median',np.median(ip),' retract points median',np.median(F.shape[1]-ip))
dz=np.array([Z[i,ip[i]]-Z[i,-1] for i in range(len(Z))]); print('retract Z travel um: median %.2f'%np.median(dz))
print('v_ret (stored) %.0f um/s, v_app %.0f'%(float(z['v_retract_um_s']),float(z['v_approach_um_s'])))
noise=z['noise_nN']; print('noise sigma nN median %.4f p5 %.4f p95 %.4f'%(np.nanmedian(noise),*np.nanpercentile(noise,[5,95])))
thr=np.maximum(5*noise,0.03); print('threshold nN median %.3f, fraction at floor 0.03: %.2f'%(np.nanmedian(thr),np.nanmean(thr<=0.0300001)))
k=P['k']*1e3
M=X.adhesion_maps(z['curves_F_nN'],z['curves_Z_um'],P['k'],z['noise_nN'],z['curve_index_map'],P['radius'])
S={}
for key,m in M.items():
    v=m[np.isfinite(m)]; S[key]=dict(median=float(np.median(v)),p5=float(np.percentile(v,5)),p95=float(np.percentile(v,95)),n=int(v.size))
    print(f'{key:22s} median {S[key]["median"]:.4g}  p5 {S[key]["p5"]:.4g}  p95 {S[key]["p95"]:.4g}  n {v.size}')
print('fraction curves: with rupture %.3f, with tether %.3f, not detached %.3f, no adhesion (F never <0) %.4f'%(np.mean(M['adh_n_ruptures']>0),np.mean(M['adh_tether_nN']>0),np.mean(M['adh_detached']==0),np.mean(M['adh_force_nN']==0)))
json.dump(S,open('stats.json','w'),indent=1)
np.savez('M.npz',**M)
# deflection at max adhesion in um (to show tip vs Z difference)
print('max |F|/k at adhesion (um): median %.3f'%(np.nanmedian(M['adh_force_nN'])/k), ' at peak load: %.3f'%(np.nanmedian(z['Fmax_nN'])/k))

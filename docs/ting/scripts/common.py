import sys, json, numpy as np
sys.path.insert(0, '/home/claude/out')
import afm_ting_pipeline as AP
Z = np.load('/home/claude/test/full/home_claude_data.npz')
PJ = json.loads(str(Z['params_json']))
P = {k: PJ[k] for k in ('num_app','num_ret','dt','zsens','invols','k','geometry','radius','half_angle','nu','t0','block','savgol','baseline')}
B = float(Z['drag_V_per_um_s'])
def raw(i):
    return np.loadtxt(f'/home/claude/data/ForceCurve/ForceCurve_{i:05d}.lvm', dtype=np.float32)[:2*P['num_app']+2*P['num_ret']]

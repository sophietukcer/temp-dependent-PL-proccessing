import numpy as np, arviz as az
idata = az.from_netcdf("results_pymc/20261112_092549/pl_inference_data.nc")
def flat(g, v):
    a = np.asarray(idata[g][v].values); return a.reshape((-1,) + a.shape[2:])

print("param            post_med  prior_med  shift(prior sd)  sd_ratio(post/prior)")
for v in ["e_ref","de_dt","gamma_0","gamma_ac","gamma_lo","e_lo","i_0","e_a","ln_q_ref","eta","rel_err","baseline"]:
    if v not in idata.posterior: continue
    po, pr = flat("posterior", v), flat("prior", v)
    for idx in np.ndindex(po.shape[1:]):
        p, q = po[(slice(None),)+idx], pr[(slice(None),)+idx]
        print(f"{v}{list(idx)!s:12s} {np.median(p):9.3g} {np.median(q):9.3g} {(np.median(p)-np.median(q))/q.std():10.2f} {p.std()/q.std():12.2f}")
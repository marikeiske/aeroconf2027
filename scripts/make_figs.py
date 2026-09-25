import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # run from repo root
from analysis import *
from figs import *
S=pd.read_csv("data/summary.csv", keep_default_na=False, na_values=[""])
R=pd.concat([load("p1_results.csv",1),load("p2_results.csv",1)])
fig_method()
M=fig_perf(S)
fig_ecdf(R)
# overhead frame
rows=[]
for f in ["data/p1_overhead.csv.meta.csv","data/p2_overhead.csv.meta.csv"]:
    m=pd.read_csv(f, keep_default_na=False); m=m[m.kernel=="gemm"]
    m["label"]=[LABEL[(a,b,c)][0] for a,b,c in zip(m.model,m.impl,m.device)]
    for (p,l,sc),g in m.groupby(["platform","label","scenario"]):
        if l=="C++ serial": continue
        steady=S[(S.platform==p)&(S.label==l)&(S.kernel=="gemm")&(S["size"]==256)&(S.scenario=="iso")]["median"].values[0]/1e3
        tag="" if sc in ("warm","ovh") else ""
        name=f"{p} {l}" + (f" ({sc} cache)" if sc in ("cold",) or (sc=="warm" and l in ("OpenCL PoCL","SYCL OMP JIT USM","SYCL→OCL Intel USM")) else "")
        rows.append(dict(label=name,t_init_ms=g.t_init_ns.median()/1e6,t_build_ms=max(g.t_build_ns.median()/1e6,1e-3),
                         t_first_ms=g.t_first_ns.median()/1e6,steady_ms=steady))
O=pd.DataFrame(rows); O.to_csv("data/overhead_summary.csv",index=False); print(O.round(2))
O["t_build_ms"]=O["t_build_ms"].where(O["t_build_ms"]>0.01,np.nan)
fig_overhead(O,S)
G=fig_tail(S)
E=fig_evt(R,[("P2","opencl","amd-ocl","gemm",256,"P2 OpenCL AMD iGPU — GEMM 256","OpenCL"),
             ("P1","sycl","acpp-ocl-jit-usm","gemm",256,"P1 SYCL→OCL Intel USM — GEMM 256","SYCL")])
print(E[["platform","config","scenario","lb_p","ks_p","gof_p","xi","pwcet_1e-06","mowcet"]].round(3))

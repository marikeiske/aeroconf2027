import numpy as np, pandas as pd, warnings, os, csv; warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from analysis import load, LABEL, pwcet
from scipy import stats
import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(ROOT)  # data/ paths below are relative to the repository root
D = os.path.join(HERE, "data")
S=pd.read_csv("data/summary.csv", keep_default_na=False, na_values=[""])
R=pd.concat([load("p1_results.csv",1),load("p2_results.csv",1)])
# ---- fig3 heatmap (P1 iso, normalized to C++ serial)
ORDER=["C++ serial","OpenCL PoCL","OpenCL Intel","SYCL OMP AOT USM","SYCL OMP JIT USM","SYCL OMP JIT buf","SYCL→OCL Intel USM","SYCL→OCL Intel buf"]
KS=[("gemm",256),("gemm",512),("conv",512),("conv",1024),("kalman",4096),("kalman",32768)]
s=S[(S.platform=="P1")&(S.scenario=="iso")]
rows=[]
for i,lab in enumerate(ORDER):
    for j,(k,n) in enumerate(KS):
        base=s[(s.label=="C++ serial")&(s.kernel==k)&(s["size"]==n)]["median"].values[0]
        v=s[(s.label==lab)&(s.kernel==k)&(s["size"]==n)]["median"].values[0]/base
        rows.append((j,i,v,np.log2(v)))
with open(f"{D}/fig3_heatmap.dat","w") as f:
    f.write("% x = kernel/size column, y = configuration row (0 = C++ serial), ratio = median/C++ serial\nx y ratio log2r\n")
    last=None
    for x,y,v,l in sorted(rows,key=lambda r:(r[1],r[0])):
        if last is not None and y!=last: f.write("\n")
        f.write(f"{x} {y} {v:.3f} {l:.4f}\n"); last=y
# ---- fig6 ECDF (P2)
for k,n in [("gemm",512),("kalman",32768)]:
    for m,impl,tag in [("cpp","serial","cpu"),("opencl","amd-ocl","gpu")]:
        for sc in ["iso","intf"]:
            v=np.sort(R[(R.platform=="P2")&(R.model==m)&(R.impl==impl)&(R.kernel==k)&(R["size"]==n)&(R.scenario==sc)].lat_ns.values/1e3)
            y=np.arange(1,len(v)+1)/len(v)
            np.savetxt(f"{D}/fig6_{k}{n}_{tag}_{sc}.dat",np.c_[v,y],fmt="%.2f %.5f",header="latency_us ecdf",comments="")
# ---- fig7 EVT
evt=[]
for plat,m,impl,tag in [("P2","opencl","amd-ocl","p2gpu"),("P1","sycl","acpp-ocl-jit-usm","p1sycl")]:
    for sc in ["iso","intf"]:
        v=R[(R.platform==plat)&(R.model==m)&(R.impl==impl)&(R.kernel=="gemm")&(R["size"]==256)&(R.scenario==sc)].lat_ns.values/1e3
        e=pwcet(v); vs=np.sort(v); cc=1-np.arange(1,len(vs)+1)/(len(vs)+1)
        np.savetxt(f"{D}/fig7_{tag}_{sc}_emp.dat",np.c_[vs,cc],fmt="%.2f %.6e",header="latency_us exceedance",comments="")
        u,xi,sg,z=e["u"],e["xi"],e["sigma"],e["k"]/e["n"]
        xs=np.geomspace(u,max(e["pwcet_1e-06"],vs[-1])*1.05,120)
        np.savetxt(f"{D}/fig7_{tag}_{sc}_gpd.dat",np.c_[xs,z*stats.genpareto.sf(xs-u,xi,0,sg)],fmt="%.2f %.6e",header="latency_us exceedance",comments="")
        evt.append(dict(tag=tag,sc=sc,pw6_ms=e["pwcet_1e-06"]/1e3,iid=min(e["lb_p"],e["ks_p"],e["gof_p"])>0.05))
pd.DataFrame(evt).to_csv(f"{D}/fig7_summary.csv",index=False)
print(pd.DataFrame(evt))
# ---- fig4 overhead
O=pd.read_csv("data/overhead_summary.csv"); O.to_csv(f"{D}/fig4_overhead.csv",index=False); print(O.round(2))
# ---- fig5 tail
t=S[S.kernel!="null"].groupby(["platform","label","family","scenario"])["tail999"].median().unstack("scenario").reset_index().dropna(subset=["iso","intf"])
t=t.sort_values(["platform","intf"]).reset_index(drop=True); t.to_csv(f"{D}/fig5_tail.csv",index=False); print(t.round(2))
# ---- fig1 msl
d=pd.read_csv(os.path.join(ROOT, "msl", "msl_included.csv")); c=d.groupby(["domain","metric"]).size().reset_index(name="n"); c.to_csv(f"{D}/fig1_msl.csv",index=False); print(c)

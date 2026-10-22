"""Development/calibration only. Never accesses confirmatory test seeds."""
import json,time
import numpy as np,pandas as pd
from attention_core import *

def main():
    model=FrozenHI();q,k,v=dataset(size=128,seed=210001)
    z,a,y=reference(q,k,v);rows=[]
    for gain in [1.,1.025,1.05,1.075,1.10,1.125,1.15,1.20,1.25]:
        ah=model.analytic(z,gain);m,_=metrics(y,ah@v)
        rows.append({'gain':gain,**m})
    cal=pd.DataFrame(rows);cal.to_csv(ROOT/'source_data'/'calibration_gain.csv',index=False)
    gain=float(cal.loc[cal.rmse.idxmin(),'gain']);print('gain',gain,cal.to_string(index=False),flush=True)
    (ROOT/'calibration.json').write_text(json.dumps({'gain':gain,'matrices':128,'seed':210001,'objective':'analytic attention-output RMSE on calibration data','model_frozen':True,'grid':[r['gain'] for r in rows]},indent=2))
    configs=[('bipolar_iid','iid','bipolar',False,1.),('magnitude_iid','iid','magnitude',False,1.),('magnitude_sobol','sobol','magnitude',False,1.),('centered','sobol','magnitude',True,1.),('calibrated','sobol','magnitude',True,gain)]
    q,k,v=dataset(size=16,seed=210002);out=[]
    for name,source,encoding,center,g in configs:
        t=time.time();m,_,_=run_core(q,k,v,model,256,71,source,encoding,center,g)
        row={'method':name,**m,'seconds':time.time()-t};out.append(row);print(row,flush=True)
    pd.DataFrame(out).to_csv(ROOT/'source_data'/'development_pilot.csv',index=False)

if __name__=='__main__':main()

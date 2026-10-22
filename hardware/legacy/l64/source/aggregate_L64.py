from pathlib import Path
import json,re,csv,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1]
old=json.loads((R/'baseline/stage3_summary.json').read_text())
verify=json.loads((R/'results/rtl_verification_L64.json').read_text())
corner=json.loads((R/'results/corner_verification.json').read_text())
rows=[];all_runs=[];routes=[]
for n in ['retime','l64','l64c']:
    vv=[v for v in verify if v['design']==n]
    assert len(vv)==3 and sum(v['checked_words'] for v in vv)==153600
    assert all(v['reference_mismatches']==v['cycle_mismatches']==0 for v in vv)
    rr=[]
    for seed in [1,2,3]:
        d=json.loads((R/f'reports/route_{n}_f8_s{seed}.json').read_text())
        fmax=min(x['achieved'] for x in d['fmax'].values());assert fmax>=8
        v=d['utilization'];x={'design':n,'seed':seed,'comb':v['TRELLIS_COMB']['used'],'ff':v['TRELLIS_FF']['used'],'dsp':v['MULT18X18D']['used'],'bram18':v['DP16KD']['used'],'timing_estimate_MHz':fmax}
        rr.append(x);routes.append(x)
    for k in ['comb','ff','dsp','bram18']:assert len({x[k] for x in rr})==1
    st=(R/f'reports/asic/{n}_stat.json').read_text();area=json.loads(st[st.index('{'):])['design']['area']
    powers=[]
    for seed in [4004,5005,6006]:
        stem=f'{n}_L64_seed{seed}';pp=json.loads((R/f'reports/asic/{stem}_power.json').read_text())['Total']
        assert re.search(r'unannotated\s+0\b',(R/f'reports/asic/{stem}_annotation.txt').read_text())
        assert 'slack (MET)' in (R/f'reports/asic/{stem}_timing.txt').read_text()
        g=np.genfromtxt(R/f'results/gate_power_{stem}.csv',delimiter=',',names=True)
        t=np.genfromtxt(R/f'results/rtl_power_{stem}.csv',delimiter=',',names=True)
        assert len(g)==16 and g.dtype.names==t.dtype.names
        assert all(np.array_equal(g[k],t[k]) for k in g.dtype.names)
        tile=int(g['cycles'].sum());assert tile==2592
        x={'design':n,'seed':seed,'area_um2':area,'power_mW':pp['total']*1000,'dynamic_mW':(pp['internal']+pp['switching'])*1000,'leakage_mW':pp['leakage']*1000,'energy_uJ':pp['total']*tile/15,'tile_cycles':tile,'gate_checked_words':16*16}
        powers.append(x);all_runs.append(x)
    row={'design':n,**{k:rr[0][k] for k in ['comb','ff','dsp','bram18']},'fmax_min_MHz':min(x['timing_estimate_MHz'] for x in rr),'fmax_max_MHz':max(x['timing_estimate_MHz'] for x in rr),'area_um2':area,'tile_cycles':2592,'fpga_latency_us':324,'asic_latency_us':172.8}
    for k in ['power_mW','dynamic_mW','leakage_mW','energy_uJ']:
        row[k]=float(np.mean([x[k] for x in powers]));row[k+'_sd']=float(np.std([x[k] for x in powers],ddof=1))
    rows.append(row)
baseline=[]
for n in ['fp32_1','fp32_4','fx16','frost']:
    f=next(x for x in old['fpga'] if x['design']==n)
    a=next(x for x in old['asic'] if x['design']==n and (n!='frost' or x['L']==64))
    p=next(x for x in old['performance'] if x['design']==n and (n!='frost' or x['L']==64))
    baseline.append({'design':n,**{k:f[k] for k in ['comb','ff','dsp','bram18']},'area_um2':a['area_um2'],'tile_cycles':p['tile_cycles'],'fpga_latency_us':p['FPGA_tile_us_at8MHz'],'asic_latency_us':p['ASIC_tile_us_at15MHz'],'power_mW':a['total_mW'],'power_mW_sd':a['total_mW_sd'],'dynamic_mW':a['dynamic_mW'],'leakage_mW':a['leakage_mW'],'energy_uJ':a['tile_energy_uJ'],'energy_uJ_sd':a['tile_energy_uJ_sd']})
final=next(x for x in rows if x['design']=='l64c')
improvements=[]
for b in baseline:
    improvements.append({'baseline':b['design'],**{k+'_reduction_percent':100*(1-final[k]/b[k]) for k in ['comb','ff','area_um2','power_mW','asic_latency_us','energy_uJ']}})
data={'scope':'L64 only, same output sequence as original L64 kernel; FPGA routes at 8 MHz; separate Nangate45 pre-layout activity at 15 MHz; no quality matching between arithmetic controls','baseline':baseline,'variants':rows,'improvements':improvements,'verification':{'rtl_words_all_variants':sum(x['checked_words'] for x in verify),'rtl_words_final':sum(x['checked_words'] for x in verify if x['design']=='l64c'),'corner_words_final':sum(x['checked_words'] for x in corner),'gate_words_all_variants':sum(x['gate_checked_words'] for x in all_runs),'mismatches':0},'area_units':'um2','energy_units':'uJ/tile','tile_queries':16,'power_workload_tiles':1,'power_stream_seeds':[4004,5005,6006]}
(R/'results/L64_summary.json').write_text(json.dumps(data,indent=2)+'\n')
for name,rs in [('L64_power_runs.csv',all_runs),('L64_route_runs.csv',routes),('L64_optimized_summary.csv',rows),('L64_improvements.csv',improvements)]:
    with (R/'results'/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rs[0].keys());w.writeheader();w.writerows(rs)
print(json.dumps(data,indent=2))

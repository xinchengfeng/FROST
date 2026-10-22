from pathlib import Path
import json,csv,hashlib,re
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'results';Q=R/'reports';A=Q/'asic'
def save_csv(name,rows):
 with (O/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read_words(p):
 a=np.genfromtxt(p,names=True,delimiter=',');return a,np.stack([a[f'word{i}'] for i in range(16)],-1).astype(np.uint32)
rtl=json.loads((O/'rtl_verification.json').read_text());assert len(rtl)==12 and sum(x['checked_words'] for x in rtl)==614400
fpga=[];routes=[]
for n in ['fp32_1','fp32_4','fx16','frost']:
 rr=[]
 for seed in [1,2,3]:
  d=json.loads((Q/f'route_{n}_f8_s{seed}.json').read_text());fmax=min(v['achieved'] for v in d['fmax'].values());assert fmax>=8
  x={'design':n,'route_seed':seed,'comb':d['utilization']['TRELLIS_COMB']['used'],'ff':d['utilization']['TRELLIS_FF']['used'],'dsp':d['utilization']['MULT18X18D']['used'],'bram18':d['utilization']['DP16KD']['used'],'fmax_estimate_MHz':fmax};routes.append(x);rr.append(x)
 for k in ['comb','ff','dsp','bram18']:assert len(set(x[k] for x in rr))==1
 fpga.append({'design':n,**{k:rr[0][k] for k in ['comb','ff','dsp','bram18']},'common_clock_MHz':8,'fmax_estimate_min_MHz':min(x['fmax_estimate_MHz'] for x in rr),'fmax_estimate_max_MHz':max(x['fmax_estimate_MHz'] for x in rr)})
save_csv('fpga_route_runs.csv',routes);save_csv('fpga_resources.csv',fpga)
performance=[]
for n in ['fp32_1','fp32_4','fx16','frost']:
 for L in ([64,256,1024] if n=='frost' else [256]):
  records=[x for x in rtl if x['design']==n and x['L']==L];z=records[0];tile=z['tile16_cycles_mean']
  assert all(x['tile16_cycles_mean']==tile for x in records)
  performance.append({'design':n,'L':L if n=='frost' else 0,'tile_cycles':int(tile),'cold_row_cycles':z['cold_row_cycles'],'resident_row_cycles':z['resident_row_cycles_min'],'FPGA_tile_us_at8MHz':tile/8,'FPGA_tiles_per_s_at8MHz':8e6/tile,'ASIC_tile_us_at15MHz':tile/15})
save_csv('kernel_performance.csv',performance)
quality=[];a=np.load(R/'inputs/evaluate_rows.npz');z=a['scores'].astype(np.float64);v=a['values'][a['matrix_index']].astype(np.float64);e=np.exp(z-z.max(-1,keepdims=True));p=e/e.sum(-1,keepdims=True);y=np.einsum('rk,rkd->rd',p,v)
for x in rtl:
 _,w=read_words(O/f"{x['design']}_L{x['L']}_seed{x['seed']}.csv");err=np.linalg.norm(w.view(np.int32)/1024-y)
 quality.append({'design':x['design'],'L':x['L'] if x['design']=='frost' else 0,'seed':x['seed'],'output_nrmse':err/np.linalg.norm(y),'centered_output_nrmse':err/np.linalg.norm(y-v.mean(1))})
save_csv('rtl_output_quality.csv',quality)
power_runs=[];powers=[]
for n in ['fp32_1','fp32_4','fx16','frost']:
 stat=(A/f'{n}_stat.json').read_text();area=json.loads(stat[stat.index('{'):])['design']['area']
 for L in ([64,256,1024] if n=='frost' else [256]):
  rr=[]
  for seed in ([4004,5005,6006] if n=='frost' else [4004]):
   stem=f'{n}_L{L}_seed{seed}';g,gw=read_words(O/f'gate_power_{stem}.csv');t,tw=read_words(O/f'rtl_power_{stem}.csv');assert len(g)==16 and np.array_equal(gw,tw) and np.array_equal(g['cycles'],t['cycles'])
   assert re.search(r'unannotated\s+0\b',(A/f'{stem}_annotation.txt').read_text());assert 'slack (MET)' in (A/f'{stem}_timing.txt').read_text()
   pp=json.loads((A/f'{stem}_power.json').read_text())['Total'];tile=int(g['cycles'].sum());row={'design':n,'L':L if n=='frost' else 0,'seed':seed,'area_um2':area,'dynamic_mW':1e3*(pp['internal']+pp['switching']),'leakage_mW':1e3*pp['leakage'],'total_mW':1e3*pp['total'],'tile_cycles':tile,'tile_energy_uJ':pp['total']*tile/15,'checked_gate_words':int(gw.size),'gate_RTL_mismatches':0};rr.append(row);power_runs.append(row)
  agg={'design':n,'L':L if n=='frost' else 0,'area_um2':area,'runs':len(rr),'tile_cycles':rr[0]['tile_cycles']}
  for key in ['dynamic_mW','leakage_mW','total_mW','tile_energy_uJ']:
   agg[key]=float(np.mean([r[key] for r in rr]));agg[key+'_sd']=float(np.std([r[key] for r in rr],ddof=1)) if len(rr)>1 else 0.
  powers.append(agg)
save_csv('asic_power_runs.csv',power_runs);save_csv('asic_power_summary.csv',powers)
summary={'scope':'hardware-only kernel comparison; precision matching waived; not a complete Transformer, FPGA power or measured chip results','rtl_checked_words':sum(x['checked_words'] for x in rtl),'gate_checked_words':sum(x['checked_gate_words'] for x in power_runs),'fpga':fpga,'performance':performance,'asic':powers,'quality':quality}
(O/'stage3_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['rtl_checked_words','gate_checked_words','fpga','performance','asic']},indent=2))

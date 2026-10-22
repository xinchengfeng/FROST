"""Human-readable source-data tables for every plotted numerical relationship."""
from pathlib import Path
import csv,json
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
P=ROOT/'source_data';protocol=json.loads((P/'protocol.json').read_text())
inp=np.load(P/'inputs.npz');lengths=protocol['lengths'];seeds=protocol['stream_seeds']
rank=np.empty(128,dtype=int);rank[inp['entropy_order']]=np.arange(1,129)
rep=int(inp['representative_row'])

def write(name,rows):
    with (P/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

atlas=[];simplex=[]
for method in ['SC_base','FROST']:
    tr=[np.load(P/f'trace_{method}_seed{s}.npz') for s in seeds]
    tv=np.stack([a['row_tv'] for a in tr]);analytic=tr[0]['analytic']
    stationary_tv=.5*np.abs(analytic-inp['reference']).sum(-1)
    for li,L in enumerate(lengths):
        for ri in range(128):
            atlas.append(dict(method=method,L=L,row_id_zero_based=ri,
                entropy_rank=int(rank[ri]),reference_entropy_nats=float(inp['entropy'][ri]),
                mean_stream_row_tv=float(tv[:,li,ri].mean()),
                sample_sd_stream_row_tv=float(tv[:,li,ri].std(ddof=1)),
                stationary_row_tv=float(stationary_tv[ri])))
    if method=='FROST':
        for seed,a in zip(seeds,tr):
            order=a['order'][rep]
            for li,L in enumerate(lengths):
                p=a['probabilities'][li,rep]
                g=np.array([p[order[0]],p[order[1]],p[order[2:]].sum()])
                simplex.append(dict(stream_seed=seed,L=L,top_key_index_one_based=int(order[0])+1,
                    second_key_index_one_based=int(order[1])+1,top_key_mass=float(g[0]),
                    second_key_mass=float(g[1]),other_14_mass=float(g[2]),
                    simplex_x=float(g[1]+g[2]/2),simplex_y=float(g[2]*np.sqrt(3)/2)))
write('atlas_row_tv.csv',atlas);write('simplex_checkpoint_coordinates.csv',simplex)
t=np.load(P/'trace_FROST_seed101.npz');micro=[]
for i in range(128):
    row=dict(recorded_cycle=i+1,emitted_sorted_key_rank=int(t['events_sorted'][i,rep])+1)
    row.update({f'state_of_score_rank_{j+1}':int(t['representative_states'][i,j]) for j in range(16)})
    micro.append(row)
write('representative_states_and_events.csv',micro)
print('Exported',len(atlas),'atlas cells,',len(simplex),'simplex checkpoints and',len(micro),'microstate cycles')

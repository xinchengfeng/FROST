from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'summary/dataset_summary.json').read_text())
def rows(split):return {(r['method'],r['L']):r for r in s[split]['statistics']}
def pair(split,a,b):return next(r for r in s[split]['paired_comparisons'] if tuple(r['method_a'])==a and tuple(r['method_b'])==b)
allrows=rows('all_200');holdrows=rows('nondevelopment_184')
a=allrows['FROST-LT',4096];b=allrows['FROST-LT',1024];h=holdrows['FROST-LT',4096]
g=pair('all_200',('FROST-LT',4096),('FROST_original',4096))
hg=pair('nondevelopment_184',('FROST-LT',4096),('FROST_original',4096))
short=pair('all_200',('FROST-LT',1024),('FROST_original',4096))
mapping={}
for prefix,row in [('LT4096',a),('LT1024',b),('HOLD',h)]:
    for key,field,places in [('PSNR','psnr_mean',2),('PSNR_SD','psnr_sd',2),('SSIM','ssim_mean',4),('SSIM_SD','ssim_sd',4)]:
        mapping[prefix+'_'+key]=f'{row[field]:.{places}f}'
mapping.update({'GAIN_ORIGINAL':f"{g['psnr_gain']:.2f}",'GAIN_ORIGINAL_LO':f"{g['image_bootstrap_95CI'][0]:.2f}",'GAIN_ORIGINAL_HI':f"{g['image_bootstrap_95CI'][1]:.2f}",
                'IMPROVED_ORIGINAL':str(g['n_images_improved']),'LT4096_GE30':str(a['images_ge_30']),'LT4096_GE40':str(a['images_ge_40']),
                'LT4096_RUNS_GE30':str(a['runs_ge_30']),'LT4096_MIN_RUN':f"{a['min_run_psnr']:.2f}",
                'SHORT_GAIN':f"{short['psnr_gain']:.2f}",'HOLD_GAIN':f"{hg['psnr_gain']:.2f}",
                'HOLD_GAIN_LO':f"{hg['image_bootstrap_95CI'][0]:.2f}",'HOLD_GAIN_HI':f"{hg['image_bootstrap_95CI'][1]:.2f}"})
assert short['psnr_gain']>0, 'Revise wording if short-stream gain is not positive'
assert a['psnr_mean']<allrows['W8A8',None]['psnr_mean'], 'Revise reference ordering if changed'
table=[]
for (method,L),r in allrows.items():
    name={'SC_base':'Base TT (uncalibrated)','FROST_original':'FROST (original)'}.get(method,method)
    table.append(f"{name} & {L if L else '--'} & ${r['psnr_mean']:.2f}\\pm{r['psnr_sd']:.2f}$ & ${r['ssim_mean']:.4f}\\pm{r['ssim_sd']:.4f}$ & {r['psnr_loss_vs_Float']:.2f} & {r['images_ge_30']}/200 \\\\")
mapping['TABLE_ROWS']='\n'.join(table)
template=(ROOT/'replacement_template.tex').read_text()
new=re.sub(r'@@([A-Z0-9_]+)@@',lambda m:mapping[m[1]],template)
assert '@@' not in new
(ROOT/'FROST_BSDS500_LT_replacement.tex').write_text(new)
print(new)

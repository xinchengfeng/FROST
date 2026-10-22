"""Generate paper text and table from measured data only."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from data import ROOT,DATASETS
LABELS=['MNIST','Fashion-MNIST','CIFAR-10']

def main():
 df=pd.read_csv(ROOT/'source_data'/'all_metrics.csv')
 full=df[df.split=='full_test'];diag=df[(df.split=='diagnostic')&(df.stream_seed==101)]
 assert len(full)==36
 def values(frame,config,metric,L=256,mult=1):
  return [frame[(frame.dataset==d)&(frame.config==config)&(frame.L==L)][metric].mean()*mult for d in DATASETS]
 def listing(vals,nd=2,suffix=''):
  a=[f'{v:.{nd}f}'+suffix for v in vals];return ', '.join(a[:-1])+' and '+a[-1]
 fp=values(full,'reference','accuracy',mult=100);w8=values(full,'w8a8','accuracy',mult=100)
 sc=values(full,'calibrated_full','accuracy',mult=100);delta=values(full,'calibrated_full','accuracy_delta_pp');analytic_acc=values(full,'calibrated_analytic','accuracy',mult=100)
 error=values(full,'calibrated_full','centered_logit_relative_error');agree=values(full,'calibrated_full','agreement',mult=100)
 e64=values(diag,'calibrated_full','centered_logit_relative_error',64);e1024=values(diag,'calibrated_full','centered_logit_relative_error',1024)
 a64=values(diag,'calibrated_full','accuracy',64,100);a1024=values(diag,'calibrated_full','accuracy',1024,100)
 generic=values(diag,'generic_full','centered_logit_relative_error');cal=values(diag,'calibrated_full','centered_logit_relative_error')
 ageneric=values(diag,'generic_full','agreement',mult=100);acal=values(diag,'calibrated_full','agreement',mult=100)
 analytic=values(diag,'calibrated_analytic','centered_logit_relative_error')
 text=rf'''\subsection{{Validation across image-classification datasets}}
\label{{sec:dataset_validation}}

We next evaluated FROST on MNIST, Fashion-MNIST and CIFAR-10, covering
handwritten digits, grayscale clothing images and RGB natural images
\cite{{lecun1998gradient,xiao2017fashion,krizhevsky2009learning}}.
Each dataset used a four-block patch Transformer with 16 tokens, two
attention heads per block and a feature width of 32. Images retained their
native resolution and were divided into a $4\times4$ patch grid. Three
reference models were trained independently for each dataset. Their
weights were then frozen, and a shared, rank-8 HI-SMURF operator was
calibrated for each model using separate images from the training
partition. Calibration used reference softmax outputs without task labels.

On the complete official test sets of 10,000 images per dataset,
floating-point accuracies were {listing(fp,suffix=r'\%')},
respectively. Finite-stream FROST at $L=256$ achieved
{listing(sc,suffix=r'\%')}, corresponding to paired accuracy
changes of {listing(delta)} percentage points
(Fig.~\ref{{fig:dataset_accuracy}}). Agreement with the corresponding
floating-point class predictions was {listing(agree,suffix=r'\%')},
and relative centred-logit errors were {listing(error,3)}.
Analytic FROST achieved {listing(analytic_acc,suffix=r'\%')}
accuracy, compared with the finite-stream values above.
The simulated W8A8 control achieved {listing(w8,suffix=r'\%')}
accuracy. This control quantized the linear and attention products while
retaining floating-point nonlinear operations. It provides a numerical
precision comparison; no inference about hardware cost follows from
these accuracy measurements.

We assessed stream-length dependence on a prespecified, class-balanced
subset of 1,000 test images per dataset
(Fig.~\ref{{fig:dataset_streams}}). Increasing $L$ from 64 to 1,024
changed relative centred-logit errors from {listing(e64,3)}
to {listing(e1024,3)}, respectively. Corresponding accuracies changed
from {listing(a64,suffix=r'\%')} to
{listing(a1024,suffix=r'\%')}. Analytic evaluation of the same
calibrated finite-state operators yielded errors of
{listing(analytic,3)} on these subsets. Reporting both classification
accuracy and continuous-output error distinguishes changes in predicted
classes from deviations that leave the classification decision unchanged.

Activation calibration was evaluated against direct transfer of the
generic HI-SMURF operator on the same diagnostic images at $L=256$
(Fig.~\ref{{fig:dataset_calibration}}). Relative centred-logit errors
changed from {listing(generic,3)} under direct transfer to
{listing(cal,3)} after calibration. Prediction agreement changed
from {listing(ageneric,suffix=r'\%')} to
{listing(acal,suffix=r'\%')}. These comparisons preserve the
Transformer weights, FSM state count and TT rank. They assess adaptation
of a fixed-capacity operator to each model's activation distribution;
they do not establish zero-shot transfer of one calibrated operator
between datasets.

Reported central values are means across three independently trained
models; figure error bars denote their sample standard deviations.
Each complete-test evaluation uses one prespecified stream realization.
Additional realizations at $L=256$ were evaluated on the diagnostic
subsets and are reported separately. Together, these experiments quantify
task-level performance and numerical fidelity of the complete stochastic
encoder across three image datasets, while retaining the limitations of
a compact backbone trained from scratch. The simulation includes
numerical centering, pooling, residual and bias addition, range handling
and stage conversions. Hardware area, energy and latency were not
evaluated.
'''
 (ROOT/'section3_4_results.tex').write_text(text)
 table=[r'\begin{table*}[t]',r'\centering',r'\caption{Classification on complete official test sets. Accuracy is reported as mean $\pm$ sample standard deviation (\%) across three independently trained models, with 10,000 test images per model. Finite-stream FROST uses $L=256$ and one prespecified stream realization per model. $\Delta$ denotes the mean paired accuracy change from floating point in percentage points.}',r'\label{tab:dataset_results}',r'\begin{tabular}{lccccc}',r'\toprule',r'Dataset & Floating point & W8A8 & FROST, analytic & FROST, $L=256$ & $\Delta$ (pp) \\',r'\midrule']
 for d,label in zip(DATASETS,LABELS):
  vals=[]
  for conf in ['reference','w8a8','calibrated_analytic','calibrated_full']:
   a=full[(full.dataset==d)&(full.config==conf)].accuracy*100;vals.append(rf'${a.mean():.2f}\pm{a.std(ddof=1):.2f}$')
  ds=full[(full.dataset==d)&(full.config=='calibrated_full')].accuracy_delta_pp.mean()
  table.append(label+' & '+' & '.join(vals)+f' & ${ds:+.2f}$ '+r'\\')
 table.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}'])
 (ROOT/'results_table.tex').write_text('\n'.join(table)+'\n')
 summary=dict(datasets=DATASETS,float_accuracy=fp,w8a8_accuracy=w8,frost_accuracy_L256=sc,delta_pp=delta,agreement_percent=agree,relative_error=error,diagnostic_relative_error_L64=e64,diagnostic_relative_error_L1024=e1024,diagnostic_accuracy_L64=a64,diagnostic_accuracy_L1024=a1024,diagnostic_generic_error=generic,diagnostic_calibrated_error=cal)
 (ROOT/'source_data'/'headline_numbers.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

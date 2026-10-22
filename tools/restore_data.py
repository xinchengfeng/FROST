"""Restore public FROST research archives into module-local layouts.

Default is a dry run. Add --apply to write verified files. Restoring inputs
does not execute model checkpoints, train a model or run a hardware flow.
"""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,zipfile,sys

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT_MAP={
 'operators/softmax_selection/':'experiments/frost_direct_hi_smurf/',
 'operators/softmax_selected/':'experiments/section3_1_softmax/',
 'operators/softmax_extended/':'experiments/frost_softmax_study/',
 'operators/attention/':'experiments/attention/',
 'operators/retrieval/':'experiments/retrieval/',
 'operators/convergence/':'experiments/convergence/',
 'classification/':'experiments/classification/',
 'reconstruction_original/':'experiments/reconstruction_original/',
 'reconstruction_LT/':'experiments/reconstruction_lt/',
 'reconstruction_scalar/':'experiments/reconstruction_scalar/'}

def destination(member,root):
    n=PurePosixPath(member)
    if n.is_absolute() or '..' in n.parts or '\\' in member:raise ValueError('Unsafe archive path')
    if len(n.parts)<2:return None
    top=n.parts[0];rel='/'.join(n.parts[1:])
    if top=='FROST_Experimental_Records':
        for old,new in EXPERIMENT_MAP.items():
            if rel.startswith(old):return root/(new+rel[len(old):])
        return root/'local_data/archive_metadata/experiments'/rel
    if top=='FROST_Hardware_Records':
        if rel.startswith(('stage3/','stage5/','stage6/')):return root/'hardware'/rel
        return root/'local_data/archive_metadata/hardware'/rel
    if top=='FROST_Source_Data':return root/'local_data/source_data'/rel
    raise ValueError('Unsupported archive root: '+top)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-records',type=Path)
    p.add_argument('--hardware-records',type=Path)
    p.add_argument('--source-data',type=Path)
    p.add_argument('--waveforms',nargs='*',type=Path,default=[])
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--overwrite',action='store_true',help='Explicitly replace differing local records')
    a=p.parse_args();root=a.root.resolve()
    archives=[x for x in [a.experiment_records,a.hardware_records,a.source_data]+a.waveforms if x]
    if not archives:p.error('Provide at least one archive path')
    plans=[];conflicts=[];verified=0
    for arc in archives:
        with zipfile.ZipFile(arc) as z:
            names=[x for x in z.namelist() if not x.endswith('/')]
            if len(names)!=len(set(names)):raise ValueError('Duplicate archive member in '+str(arc))
            manifests=[n for n in names if n.count('/')==1 and (n.endswith('/MANIFEST_SHA256.json') or '/WAVEFORMS_PART' in n)]
            if len(manifests)!=1:raise ValueError('Expected one release manifest in '+str(arc))
            prefix=manifests[0].split('/')[0]+'/'
            expected={prefix+x['path']:x['sha256'] for x in json.loads(z.read(manifests[0]))}
            for n in names:
                dst=destination(n,root)
                if dst is None:continue
                if not dst.resolve().is_relative_to(root):raise ValueError('Destination escapes repository')
                data=z.read(n);digest=hashlib.sha256(data).hexdigest()
                if n in expected:
                    if digest!=expected[n]:raise ValueError('Hash mismatch: '+n)
                    verified+=1
                if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest()!=digest and not a.overwrite:conflicts.append(str(dst.relative_to(root)))
                plans.append((arc,n,dst,digest))
    if conflicts:
        print(json.dumps({'status':'blocked_by_local_conflicts','conflicts':conflicts,'hint':'Use a fresh checkout, or --overwrite only for records you intend to replace.'},indent=2));return 1
    # Reopen archives only when writing, keeping peak memory independent of total archive size.
    written=0
    if a.apply:
        new_paths=[dst.relative_to(root).as_posix() for source,n,dst,digest in plans if not dst.exists()]
        for arc in archives:
            with zipfile.ZipFile(arc) as z:
                for source,n,dst,digest in plans:
                    if source!=arc:continue
                    if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest()==digest:continue
                    dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(z.read(n));written+=1
        # Local Git exclusion avoids committing restored records as new source files.
        # This does not change the distributed .gitignore or any tracked file.
        git_dir=root/'.git'
        if git_dir.is_dir() and new_paths:
            exclude=git_dir/'info/exclude';exclude.parent.mkdir(parents=True,exist_ok=True)
            old=exclude.read_text() if exclude.exists() else ''
            patterns=['/'+x.replace('[','[[]').replace('*','[*]').replace('?','[?]') for x in new_paths]
            extra=[x for x in patterns if x not in old.splitlines()]
            if extra:exclude.write_text(old+'\n# Locally restored FROST research records\n'+'\n'.join(extra)+'\n')
    print(json.dumps({'status':'restored' if a.apply else 'dry_run','archives':len(archives),
       'manifest_files_verified':verified,'planned_files':len(plans),'written_files':written,
       'reference_pixels_restored':False,'note':'Original benchmark pixels are obtained separately. Read reconstruction prepare_data instructions before scoring.'},indent=2));return 0

if __name__=='__main__':sys.exit(main())

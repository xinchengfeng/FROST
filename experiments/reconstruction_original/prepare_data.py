"""Restore reconstruction layouts from public records and locally obtained BSDS500.

The public data archive omits upstream reference image pixels. This helper
reacquires no files: give it an extracted records directory and BSDS500 JPGs.
It checks all delivered reference hashes before restoring local working copies.
Original public records and input JPGs are never modified.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import re
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
EXPERIMENTS=HERE.parent

def digest(data): return hashlib.sha256(data).hexdigest()

def crop(path):
    image=Image.open(path).convert('RGB')
    w,h=image.size
    image=image.resize((round(w*96/min(w,h)),round(h*96/min(w,h))),Image.Resampling.BICUBIC)
    arr=np.asarray(image)
    h,w=arr.shape[:2]
    return arr[(h-64)//2:(h-64)//2+64,(w-64)//2:(w-64)//2+64].astype(np.float32)/255

def locate_records(path):
    path=path.resolve()
    if (path/'FROST_Experimental_Records').is_dir():path=path/'FROST_Experimental_Records'
    if not (path/'MANIFEST_SHA256.json').is_file():
        raise FileNotFoundError('Use the extracted FROST_Experimental_Records directory.')
    return path

class Installer:
    def __init__(self,records,images,overwrite=False):
        self.records=locate_records(records)
        self.manifest={e['path']:e for e in json.loads((self.records/'MANIFEST_SHA256.json').read_text())}
        self.references={e['path']:e for e in json.loads((self.records/'REFERENCE_ARRAY_HASHES.json').read_text())}
        self.photos={}
        for path in sorted(images.resolve().rglob('*')):
            if path.is_file() and path.suffix.lower() in {'.jpg','.jpeg'} and path.stem.isdigit():
                iid=int(path.stem)
                if iid in self.photos and path.read_bytes()!=self.photos[iid].read_bytes():
                    raise ValueError(f'Different source JPGs share image ID {iid}. Use the correct BSDS500 images directory.')
                self.photos[iid]=path
        if not self.photos:raise FileNotFoundError('No numeric-ID BSDS500 JPGs found.')
        self.crops={};self.overwrite=overwrite
        self.report={'copied_or_restored':0,'restored_reference_arrays':0,'unchanged':0,'derived_replay_inputs':[]}

    def reference(self,relative):
        info=self.references.get(relative,{})
        if 'truth' not in info: return None
        iid=int(re.findall(r'\d+',Path(relative).stem)[0])
        if iid not in self.crops:
            if iid not in self.photos:raise FileNotFoundError(f'Missing BSDS500 JPG for image {iid}.')
            self.crops[iid]=crop(self.photos[iid])
        truth=self.crops[iid];expected=info['truth']
        actual=digest(np.ascontiguousarray(truth).tobytes())
        if actual!=expected['array_sha256'] or list(truth.shape)!=expected['shape'] or str(truth.dtype)!=expected['dtype']:
            raise ValueError(f'Reference crop hash differs for {iid}. Check source JPG, Pillow version and protocol before proceeding.')
        return truth

    def read(self,relative,restore=True):
        data=(self.records/relative).read_bytes()
        expected=self.manifest.get(relative)
        if expected and digest(data)!=expected['sha256']:raise ValueError('Corrupt records file: '+relative)
        if not relative.endswith('.npz'):return data
        truth=self.reference(relative) if restore else None
        if truth is None:return data
        with np.load(io.BytesIO(data),allow_pickle=False) as archive:
            arrays={k:archive[k] for k in archive.files}
        arrays['truth']=truth
        buf=io.BytesIO();np.savez_compressed(buf,**arrays)
        self.report['restored_reference_arrays']+=1
        return buf.getvalue()

    def write(self,dest,data,replace_original_sha=None):
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if dest.read_bytes()==data:self.report['unchanged']+=1;return
            # A generic records installer may already have copied the public,
            # reference-free NPZ here. Replace only those verified bytes while
            # restoring its upstream truth array; protect new inference outputs.
            public_copy = replace_original_sha is not None and digest(dest.read_bytes()) == replace_original_sha
            if not self.overwrite and not public_copy:raise FileExistsError(f'{dest} exists and differs. Work in a fresh checkout or explicitly use --overwrite.')
        dest.write_bytes(data);self.report['copied_or_restored']+=1

    def copy(self,relative,dest,restore=True):
        delivered_sha=self.manifest.get(relative,{}).get('sha256')
        self.write(dest,self.read(relative,restore),replace_original_sha=delivered_sha)

    def tree(self,source,dest,filter_fn=None):
        for path in sorted((self.records/source).rglob('*')):
            if path.is_file():
                rel=path.relative_to(self.records)
                if filter_fn is None or filter_fn(path):self.copy(str(rel),dest/path.relative_to(self.records/source))

    def common(self,dest):
        for filename in ['softmax_N16_r8.pt','scalar_tables.json']:
            self.copy('classification/checkpoints/'+filename,dest/filename)

    def original(self):
        self.tree('reconstruction_original',HERE)
        self.common(HERE/'source/original/checkpoints')
        (HERE/'logs').mkdir(exist_ok=True)
        # The legacy summary checks original JPG bytes, as well as crop hashes.
        protocol=json.loads((HERE/'source_data/protocol.json').read_text())
        for item in protocol['images']:
            path=self.photos[item['image_id']]
            data=path.read_bytes()
            if digest(data)!=item['sha256']:raise ValueError(f'Original JPG hash differs: {path}')
            self.write(HERE/'data/test'/item['file'],data)

    def lt(self):
        dest=EXPERIMENTS/'reconstruction_lt'
        self.tree('reconstruction_LT',dest)
        self.tree('reconstruction_original/checkpoints',dest/'runtime/checkpoints')
        self.common(dest/'runtime/source/original/checkpoints')
        for path in sorted((self.records/'reconstruction_original/runs').glob('image_*.npz')):
            self.copy(str(path.relative_to(self.records)),dest/'inputs'/path.name)
        for iid in [2018,43033,235098]:
            for seed in [101,202,303]:
                name=f'{iid}_LocalTT33_stratified_S{seed}.npz'
                self.copy('reconstruction_scalar/results/'+name,dest/'provenance'/name)
        # The selected-model training module is also imported at inference.
        # Development/training rows remain under scalar/results in the data archive.
        for name in ['calibration_rows.npz','calibration_protocol.json','candidate_configs.json']:
            rel='reconstruction_scalar/results/'+name
            if (self.records/rel).exists():self.copy(rel,dest/'results'/name)

    def scalar(self):
        dest=EXPERIMENTS/'reconstruction_scalar'
        self.tree('reconstruction_scalar',dest)
        self.common(dest/'legacy_runtime/source/original/checkpoints')
        self.copy('reconstruction_LT/checkpoints/local33_r16_max8_aug.pt',dest/'checkpoints/local33_r16_max8_aug.pt')
        for iid in [2018,43033,235098]:
            for name in [f'render_{iid}.pt',f'operator_render_{iid}.pt']:
                self.copy('reconstruction_original/checkpoints/'+name,dest/'legacy_runtime/checkpoints'/name)
            self.copy(f'reconstruction_original/runs/image_{iid}.npz',dest/'legacy_runtime/runs'/f'image_{iid}.npz')
            with np.load(io.BytesIO(self.read(f'reconstruction_original/runs/image_{iid}.npz')),allow_pickle=False) as old:
                data={'truth':old['truth'],'Float':old['Float'],'FROST':old['FROST_L4096_S101']}
            for method,key in [('SExp','SExp_CountNorm'),('Bernstein','Bernstein_CountNorm')]:
                with np.load(io.BytesIO(self.read(f'reconstruction_scalar/results/{iid}_{method}_S101.npz')),allow_pickle=False) as pred:
                    data[key]=pred['prediction']
            buf=io.BytesIO();np.savez_compressed(buf,**data)
            self.write(dest/'inputs'/f'stage2_pilot_{iid}.npz',buf.getvalue())
            self.report['derived_replay_inputs'].append({'path':f'inputs/stage2_pilot_{iid}.npz','source':'Archived Float/original FROST/SExp/Bernstein seed101 arrays, with verified upstream reference pixels; no new inference'})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',required=True,type=Path)
    parser.add_argument('--bsds-images',required=True,type=Path)
    parser.add_argument('--targets',nargs='+',choices=['original','lt','scalar'],default=['original','lt','scalar'])
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args();installer=Installer(args.records,args.bsds_images,args.overwrite)
    for target in args.targets:getattr(installer,target)()
    print(json.dumps(installer.report,indent=2))

if __name__=='__main__':main()

"""Official BSDS500 images, preserving the published disjoint split."""
import argparse, concurrent.futures, hashlib, json, tarfile, urllib.request
from pathlib import Path
from PIL import Image
import numpy as np

URL='https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz'
SIZE=70763455

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path)
    a=ap.parse_args();root=a.root;root.mkdir(parents=True,exist_ok=True)
    archive=root/'BSR_bsds500.tgz'
    if not archive.exists():
        chunk=4*1024*1024
        def part(start):
            end=min(SIZE,start+chunk)-1
            for attempt in range(3):
                try:
                    request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(request,timeout=30) as r:
                        data=r.read();cr=r.headers.get('Content-Range')
                    assert cr==f'bytes {start}-{end}/{SIZE}' and len(data)==end-start+1
                    print('Downloaded',start,end,flush=True);return data
                except Exception:
                    if attempt==2:raise
            raise RuntimeError('download failed')
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            archive.write_bytes(b''.join(ex.map(part,range(0,SIZE,chunk))))
    assert archive.stat().st_size==SIZE
    manifest=dict(url=URL,bytes=SIZE,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),splits={})
    with tarfile.open(archive,'r:gz') as tar:
        members=[m for m in tar.getmembers() if '/images/' in m.name and m.name.endswith('.jpg')]
        for split,expected in [('train',200),('val',100),('test',200)]:
            selected=sorted([m for m in members if f'/images/{split}/' in m.name],key=lambda m:int(Path(m.name).stem))
            assert len(selected)==expected,(split,len(selected))
            (root/split).mkdir(exist_ok=True)
            manifest['splits'][split]=[]
            for m in selected:
                data=tar.extractfile(m).read();out=root/split/Path(m.name).name
                out.write_bytes(data)
                im=Image.open(out).convert('RGB')
                assert min(im.size)>=64
                manifest['splits'][split].append(dict(id=int(out.stem),file=out.name,sha256=hashlib.sha256(data).hexdigest(),size=list(im.size)))
    ids=[set(r['id'] for r in manifest['splits'][s]) for s in ['train','val','test']]
    assert not (ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    (root/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('VERIFIED splits 200 train / 100 val / 200 test, disjoint',flush=True)

if __name__=='__main__':main()

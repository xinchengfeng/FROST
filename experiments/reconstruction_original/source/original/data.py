"""Public benchmark downloads, disjoint training/calibration splits and patch tokens."""
from pathlib import Path
import gzip, hashlib, io, json, tarfile, urllib.request, concurrent.futures
import numpy as np

ROOT=Path(__file__).resolve().parent
DATASETS=['mnist','fashion_mnist','cifar10']
IDX_FILES=['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']
EXPECTED={
 'mnist':['f68b3c2dcbeaaa9fbdd348bbdeb94873','d53e105ee54ea40749a09fcbcd1e9432','9fb629c4189551a2d022fa330f9573f3','ec29112dd5afa0611ce80d1b7f02629c'],
 'fashion_mnist':['8d4fb7e6c68d591d4c3dfef9ec88bf0d','25c81989df183df01b3e8a0aad5dffbe','bef4ecab320f06d8554ea6380940ec79','bb300cfdad3c16e7a12a480ee83cd310'],
 'cifar10':['c32a1d4ab5d03f1284b67883e8d87530']}

def save_npz(path,**arrays):
 b=io.BytesIO();np.savez_compressed(b,**arrays);path.write_bytes(b.getvalue())

def fetch(job):
 dataset,name,url,md5=job
 p=ROOT/'data'/'raw'/dataset/name;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  if dataset=='cifar10':
   # Download independent byte ranges from the official host, and verify MD5.
   size=170052171;chunk=8*1024*1024
   def part(start):
    end=min(start+chunk,size)-1
    req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(req,timeout=60) as r:
     got=r.read();cr=r.headers.get('Content-Range')
    assert cr==f'bytes {start}-{end}/{size}' and len(got)==end-start+1,(cr,len(got))
    print('CIFAR-10 downloaded range',start,end,flush=True);return got
   with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:buf=b''.join(ex.map(part,range(0,size,chunk)))
  else:
   with urllib.request.urlopen(url,timeout=60) as r:buf=r.read()
  p.write_bytes(buf)
 got=hashlib.md5(p.read_bytes()).hexdigest()
 if got!=md5:raise ValueError((name,got,md5))
 return dict(dataset=dataset,file=name,url=url,bytes=p.stat().st_size,md5=got,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def stratified_indices(y,total,seed):
 rng=np.random.default_rng(seed);classes=np.unique(y);assert total%len(classes)==0
 indices=np.concatenate([rng.permutation(np.flatnonzero(y==c))[:total//len(classes)] for c in classes])
 return rng.permutation(indices)

def prepare():
 for folder in ['data/raw','checkpoints','source_data/runs','figures','provenance']:(ROOT/folder).mkdir(parents=True,exist_ok=True)
 jobs=[]
 for d in DATASETS[:2]:
  base='https://ossci-datasets.s3.amazonaws.com/mnist/' if d=='mnist' else 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'
  jobs.extend((d,n,base+n,m) for n,m in zip(IDX_FILES,EXPECTED[d]))
 jobs.append(('cifar10','cifar-10-binary.tar.gz','https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz',EXPECTED['cifar10'][0]))
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
  receipts=list(ex.map(fetch,jobs))
 (ROOT/'provenance'/'download_manifest.json').write_text(json.dumps(receipts,indent=2))
 for d in DATASETS:prepare_dataset(d)

def prepare_dataset(d):
 p=ROOT/'data'/'raw'/d
 if d!='cifar10':
  data=[gzip.decompress((p/n).read_bytes()) for n in IDX_FILES]
  train=np.frombuffer(data[0],np.uint8,offset=16).reshape(-1,1,28,28)
  labels=np.frombuffer(data[1],np.uint8,offset=8).astype(np.int64)
  test=np.frombuffer(data[2],np.uint8,offset=16).reshape(-1,1,28,28)
  targets=np.frombuffer(data[3],np.uint8,offset=8).astype(np.int64)
 else:
  with tarfile.open(p/'cifar-10-binary.tar.gz') as t:
   batches=[np.frombuffer(t.extractfile(f'cifar-10-batches-bin/data_batch_{k}.bin').read(),np.uint8).reshape(-1,3073) for k in range(1,6)]
   a=np.concatenate(batches);b=np.frombuffer(t.extractfile('cifar-10-batches-bin/test_batch.bin').read(),np.uint8).reshape(-1,3073)
  train=a[:,1:].reshape(-1,3,32,32);labels=a[:,0].astype(np.int64)
  test=b[:,1:].reshape(-1,3,32,32);targets=b[:,0].astype(np.int64)
 hold=stratified_indices(labels,5000,410001)
 # Balanced validation, calibration and calibration-validation partitions.
 val=hold[:4000] # Random, rather than class-sorted, subsets of a balanced holdout.
 cal=hold[4000:4512];calval=hold[4512:]
 tr=np.setdiff1d(np.arange(len(train)),hold)
 testsub=stratified_indices(targets,1000,420001)
 norm=train[tr].astype(np.float32)/255
 mean=norm.mean(axis=(0,2,3));std=norm.std(axis=(0,2,3));del norm
 np.save(ROOT/'data'/f'{d}_train_images.npy',train);np.save(ROOT/'data'/f'{d}_test_images.npy',test)
 save_npz(ROOT/'data'/f'{d}_metadata.npz',train_labels=labels,test_labels=targets,train_indices=tr,validation_indices=val,calibration_indices=cal,calibration_validation_indices=calval,test_subset_indices=testsub,mean=mean,std=std)
 print(d, 'train',len(tr),'validation',len(val),'calibration',len(cal),'calval',len(calval),'test',len(test),'mean',mean,'std',std,flush=True)

def load_data(dataset):
 meta=dict(np.load(ROOT/'data'/f'{dataset}_metadata.npz'))
 meta['images']=np.load(ROOT/'data'/f'{dataset}_train_images.npy',mmap_mode='r')
 meta['test_images']=np.load(ROOT/'data'/f'{dataset}_test_images.npy',mmap_mode='r')
 return meta

def patchify(images,mean,std):
 """Non-overlapping 4 x 4 patch grid, with no image downsampling."""
 x=(np.asarray(images,dtype=np.float32)/255-mean[None,:,None,None])/std[None,:,None,None]
 b,c,h,w=x.shape;p=h//4
 return x.reshape(b,c,4,p,4,p).transpose(0,2,4,1,3,5).reshape(b,16,c*p*p).copy()

if __name__=='__main__':prepare()

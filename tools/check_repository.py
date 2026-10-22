"""Check Python syntax and optional release integrity; does not run experiments."""
from pathlib import Path
import argparse,ast,hashlib,json,sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-manifest',action='store_true',help='Check exact packaged bytes; expected to change after repository edits')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    excluded={'.git','.venv','venv','__pycache__','local_data','outputs','obj_dir','build'}
    files=[x for x in root.rglob('*') if x.is_file() and not any(k in excluded for k in x.relative_to(root).parts)]
    syntax=[]
    for x in files:
        if x.suffix=='.py':
            try:ast.parse(x.read_text(encoding='utf-8-sig'),filename=str(x.relative_to(root)))
            except (SyntaxError,UnicodeError) as e:syntax.append(str(e))
    missing=[];changed=[]
    if a.verify_manifest:
        records=json.loads((root/'metadata/SOURCE_MANIFEST.json').read_text())
        for row in records:
            f=root/row['path']
            if not f.is_file():missing.append(row['path'])
            elif hashlib.sha256(f.read_bytes()).hexdigest()!=row['sha256']:changed.append(row['path'])
    result={'python_files_checked':sum(x.suffix=='.py' for x in files),'syntax_errors':syntax,
       'manifest_requested':a.verify_manifest,'missing':missing,'changed':changed,
       'largest_file_bytes':max((x.stat().st_size for x in files),default=0),
       'scientific_experiments_executed':False}
    print(json.dumps(result,indent=2));return 1 if syntax or missing or changed else 0

if __name__=='__main__':sys.exit(main())

"""Package validated artifacts without redistributing full public image archives."""
from pathlib import Path
import hashlib,io,json,time,zipfile
from data import ROOT

def included(p):
 rel=p.relative_to(ROOT)
 if '__pycache__' in rel.parts or p.name.startswith('.'):return False
 if rel.parts[0]=='data' and (len(rel.parts)>2 or not p.name.endswith('_metadata.npz')):return False
 return p.is_file() and p.name!='manifest_sha256.json'

def main():
 check=json.loads((ROOT/'provenance'/'result_checks.json').read_text());assert check['verified_shards']==810
 receipt={'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'recorded_prediction_shards':810,'recorded_image_evaluations':405000,'actual_stochastic_image_evaluations':135000,'full_test_configurations':36,'additional_actual_diagnostic_configurations':45,'diagnostic_records_including_full_test_subsets':81,'all_results_verified':True}
 (ROOT/'provenance'/'run_complete.json').write_text(json.dumps(receipt,indent=2))
 files=sorted(p for p in ROOT.rglob('*') if included(p))
 manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 (ROOT/'manifest_sha256.json').write_text(json.dumps(manifest,indent=2));files.append(ROOT/'manifest_sha256.json')
 out=ROOT.parent/'Section3_4_Dataset_Validation_Figures_and_Code.zip';buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:z.writestr(str(Path(ROOT.name)/p.relative_to(ROOT)),p.read_bytes())
 out.write_bytes(buf.getvalue())
 with zipfile.ZipFile(out) as z:assert z.testzip() is None
 print(json.dumps({'archive':str(out),'bytes':out.stat().st_size,'files':len(files)},indent=2))

if __name__=='__main__':main()

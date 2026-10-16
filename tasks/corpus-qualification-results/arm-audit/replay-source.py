import hashlib,json,sys
from pathlib import Path
import pandas as pd
from datacache import materialize
from hitlist import observations
from hitlist.export import generate_observations_table
root=Path('/Users/iskander/.hitlist/ci-corpus-downloads/corpora/7f7f4035d8ec080b26385710d447633dba36ac5cce49008e48e14cf8894f20d5/bundles/2026-10-16T03-22-13Z')
source=root/'observations.parquet'
fields=['peptide','pmid','mhc_restriction','mhc_class','mhc_species','source','is_binding_assay','assay_method','response_measured','qualitative_measurement','cell_name','source_tissue','antigen_processing_comments','assay_comments','restriction_evidence','attributed_sample_label','assay_iri','provenance_id','reference_iri','reference_title','culture_condition']
import pyarrow.parquet as pq
fields=[x for x in fields if x in pq.read_schema(source).names]
version=sys.argv[1]
yaml=Path('/private/tmp/hitlist-corpus-qualification/hitlist/data/pmid_overrides.yaml')
sourcehash='3faca5334ba9a03ba584bf72da6d856886634dc269cd4172a97e5258c5909ab8'
yamlhash=hashlib.sha256(yaml.read_bytes()).hexdigest()
def build(inputs,outputs):
 f=pd.read_parquet(inputs['observations.parquet'],columns=fields,filters=[('pmid','in',[32350084,33981976])])
 f.to_parquet(outputs['deposit.parquet'],index=False)
 observations.load_observations=lambda **kw:f.copy()
 out=generate_observations_table(exclude_non_peptide_ligand=False)
 keep=fields+[x for x in ['condition_id','sample_attribution','sample_label','sample_group','arm_resolution','mhc_allele_set','mhc_allele_provenance','condition_category','condition_control'] if x not in fields]
 out[[x for x in keep if x in out]].to_parquet(outputs['replay.parquet'],index=False)
 stats={}
 for pmid in [32350084,33981976]:
  d=out[out.pmid==pmid]; counts=lambda col:{str(k):int(v) for k,v in d[col].astype(str).value_counts().items() if v}
  stats[str(pmid)]={'n_observations':len(d),'n_resolved_observations':int(d.condition_id.ne('').sum()),'attribution':counts('sample_attribution'),'verdict':counts('arm_resolution'),'n_distinct_assay_comments':d.assay_comments.nunique()}
 contexts=out.groupby(['pmid','assay_comments','condition_id','sample_attribution','sample_group'],observed=True,dropna=False).size().reset_index(name='n_observations')
 contexts.to_csv(outputs['contexts.tsv'],sep='\t',index=False)
 Path(outputs['summary.json']).write_text(json.dumps(stats,indent=2)+'\n')
p=materialize('/Users/iskander/.hitlist/corpus-qualification/source-reaudit/694/'+version,{'observations.parquet':{'path':str(source),'sha256':sourcehash},'curation.yaml':{'path':str(yaml),'sha256':yamlhash}},transform={'version':'694-deposited-row-replay-v1','options':{'fields':fields,'pmids':[32350084,33981976]}},outputs={n:{} for n in ['deposit.parquet','replay.parquet','contexts.tsv','summary.json']},builder=build)
print(Path(p['summary.json']).read_text())
Path('/private/tmp/hitlist-694-'+version+'.json').write_text(json.dumps({n:{'path':str(v),'sha256':hashlib.sha256(Path(v).read_bytes()).hexdigest(),'size':Path(v).stat().st_size} for n,v in p.items()},indent=2)+'\n')

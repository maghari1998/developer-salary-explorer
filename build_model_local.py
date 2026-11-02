"""One-time preparation: run locally against the original assignment CSV.
Never commit the survey data. The output bundle omits all respondent-level data.
Run: python build_model_local.py '/path/to/processed_data_assignment2_2026(1).csv'
"""
import sys, os, hashlib
import pandas as pd
import joblib
from assignment2_bonus_dashboard import train_bundle

if len(sys.argv)!=2:
    raise SystemExit('Usage: python build_model_local.py /path/to/original.csv')
source=sys.argv[1]
print('Reading original CSV...', flush=True)
df=pd.read_csv(source,low_memory=False)
print('Training models on dataset shape',df.shape,flush=True)
b=train_bundle(df)
# Preserves the exact original survey choices, without including individual records.
choices={col:sorted(set(b['train'][col].dropna().astype(str)))
         for col in ['Country','Role','RoleFunctions','ICorPM','OrgSize','Education']}
# Never publish raw dataframe, including any original survey row. Selected model data only.
b.pop('train')
b['choices']=choices
b['metadata']={'source_filename':os.path.basename(source),'row_count':len(df),
               'feature_count':len(b['selected_names']),'model_type':'14 cumulative logistic regressions',
               'sha256_first_12':hashlib.sha256(open(source,'rb').read()).hexdigest()[:12]}
joblib.dump(b,os.path.join(os.path.dirname(__file__),'model_bundle.joblib'),compress=3)
print('Saved model_bundle.joblib',os.path.getsize(os.path.join(os.path.dirname(__file__),'model_bundle.joblib')),'bytes',flush=True)

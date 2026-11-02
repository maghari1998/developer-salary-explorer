"""MIE1624 A2 optional bonus: local Streamlit dashboard.
Run: streamlit run assignment2_bonus_dashboard.py
Requirements: pandas numpy scipy scikit-learn matplotlib streamlit.
Place the original processed_data_assignment2_2026.csv alongside the script,
or upload that unchanged CSV in the browser. Never upload the CSV publicly.
Model is fitted in memory from the original CSV: no pretrained artifacts or
encoded inputs are required from users.
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

ORDER = [
    "$0–<$10,000", "$10,000–<$20,000", "$20,000–<$30,000",
    "$30,000–<$40,000", "$40,000–<$50,000", "$50,000–<$75,000",
    "$75,000–<$100,000", "$100,000–<$125,000",
    "$125,000–<$150,000", "$150,000–<$200,000",
    "$200,000–<$250,000", "$250,000–<$300,000",
    "$300,000–<$400,000", "$400,000–≤$500,000", ">$500,000"
]
SELECTED = [
    "Country", "YearsCode", "WorkExp", "ToolCount", "Role", "RoleFunctions",
    "ICorPM", "OrgSize", "Education", "LanguagesHaveWorkedWith",
    "DatabasesHaveWorkedWith", "CloudPlatHaveWorkedWith"
]
TECH_GROUPS = ["LanguagesHaveWorkedWith", "DatabasesHaveWorkedWith", "CloudPlatHaveWorkedWith"]

class CumulativeOrdinal(ClassifierMixin, BaseEstimator):
    def __init__(self, class_order=tuple(ORDER), C=0.3, penalty="l1"):
        self.class_order, self.C, self.penalty = class_order, C, penalty
    def fit(self, X, y):
        self.classes_ = np.asarray(self.class_order)
        labels = pd.Categorical(y,categories=self.class_order,ordered=True).codes
        if np.any(labels < 0): raise ValueError("Unknown salary-bucket label")
        self.models_=[]
        for k in range(len(self.classes_)-1):
            model=LogisticRegression(C=self.C,penalty=self.penalty,
                                     solver="liblinear",max_iter=2000,random_state=42)
            model.fit(X,(labels <= k).astype(int))
            self.models_.append(model)
        return self
    def predict_proba(self,X):
        cdf=np.column_stack([m.predict_proba(X)[:,1] for m in self.models_])
        cdf=np.maximum.accumulate(cdf,axis=1)
        return np.diff(np.column_stack([np.zeros(X.shape[0]),cdf,np.ones(X.shape[0])]),axis=1)
    def predict(self,X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def train_bundle(data):
    """Reproduce the submitted final-model pipeline on training rows only."""
    df=data.drop(columns=["Compensation","ConvertedCompYearly","ResponseId"])
    X=df.drop(columns=["ConvertedCompYearlyBucket"])
    y=df["ConvertedCompYearlyBucket"]
    Xt, Xv, yt, yv=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    mask=Xt["Employment"].isin(["Employed","Independent contractor, freelancer, or self-employed"])
    Xt=Xt.loc[mask].copy();yt=yt.loc[mask].copy()
    drop_cols=Xt.columns[(Xt.isna().mean()*100)>95]
    Xt=Xt.drop(columns=drop_cols)
    numeric=Xt.select_dtypes(include="number").columns.tolist()
    strings=Xt.select_dtypes(exclude="number").columns.tolist()
    multi=[c for c in strings if Xt[c].dropna().astype(str).str.contains(";",regex=False).mean()>0.10]
    single=[c for c in strings if c not in multi]
    caps={c:Xt[c].quantile(.99) for c in ["ToolCount","AIToolCount"]}
    for c,limit in caps.items():Xt[c]=Xt[c].clip(upper=limit)
    medians=Xt[numeric].median(); Xt[numeric]=Xt[numeric].fillna(medians)
    Xt[single]=Xt[single].fillna("Missing")
    multi_parts=[]
    for col in multi:
        dummy=Xt[col].fillna("").str.get_dummies(sep=";")
        keep=dummy.columns[dummy.sum()>=50]
        part=dummy[keep].copy()
        part.columns=[f"{col}::{v}" for v in keep]
        part[f"{col}::Missing"]=Xt[col].isna().astype(int)
        multi_parts.append(part.astype("Sparse[int8]"))
    multi_matrix=pd.concat(multi_parts,axis=1)
    encoder=OneHotEncoder(handle_unknown="ignore",min_frequency=25)
    onehot=encoder.fit_transform(Xt[single])
    all_names=numeric+encoder.get_feature_names_out(single).tolist()+multi_matrix.columns.tolist()
    all_X=hstack([csr_matrix(Xt[numeric].to_numpy()),onehot,multi_matrix.sparse.to_coo()]).tocsr()
    positions=[i for i,name in enumerate(all_names) if any(name==g or name.startswith(g+"_") or name.startswith(g+"::") for g in SELECTED)]
    chosen_names=[all_names[i] for i in positions]
    chosen_X=all_X[:,positions]
    scaler=ColumnTransformer([("scale",StandardScaler(with_mean=False),[0,1,2])],remainder="passthrough")
    pipe=Pipeline([("preprocessor",scaler), ("model",CumulativeOrdinal())])
    pipe.fit(chosen_X,yt)
    supports={name:int(multi_matrix[name].sum()) for name in multi_matrix.columns}
    return {"model":pipe,"train":Xt,"medians":medians,"caps":caps,"numeric":numeric,
            "single":single,"multi":multi,"encoder":encoder,"multi_names":multi_matrix.columns.tolist(),
            "positions":positions,"selected_names":chosen_names,"supports":supports,"N":len(Xt)}


def original_to_X(answer,b):
    raw=pd.DataFrame([answer]); cols=b["numeric"]+b["single"]+b["multi"]
    row=pd.DataFrame(index=raw.index,columns=cols)
    for col in raw.columns:
        if col in row.columns:row[col]=raw[col].values
    for col in b["numeric"]:
        row[col]=pd.to_numeric(row[col],errors="coerce")
        if col in b["caps"]:row[col]=row[col].clip(upper=b["caps"][col])
        row[col]=row[col].fillna(b["medians"][col])
    onehot=b["encoder"].transform(row[b["single"]].fillna("Missing"))
    index={name:i for i,name in enumerate(b["multi_names"])}
    mc=lil_matrix((1,len(index)),dtype=np.int8)
    for col in b["multi"]:
        val=row.at[0,col]
        keys=[f"{col}::Missing"] if pd.isna(val) else [f"{col}::{x.strip()}" for x in str(val).split(";")]
        for key in keys:
            if key in index:mc[0,index[key]]=1
    full=hstack([csr_matrix(row[b["numeric"]].to_numpy(dtype=float)),onehot,mc.tocsr()]).tocsr()
    return full[:,b["positions"]].tocsr()


def rank_skills(answer,b,min_support=0):
    x=original_to_X(answer,b)
    names=b["selected_names"]
    model=b["model"]
    baseline=model.predict_proba(x)[0]
    ranks=np.arange(1,16)
    results=[]
    for j,name in enumerate(names):
        if "::" not in name or x[0,j] != 0:continue
        group,tech=name.split("::",1)
        if group not in TECH_GROUPS or tech=="Missing":continue
        support=b["supports"].get(name,0)
        if support < min_support:continue
        answered=x.tolil(copy=True)
        missing_name=f"{group}::Missing"
        if missing_name in names: answered[0,names.index(missing_name)]=0
        p_answered=model.predict_proba(answered.tocsr())[0]
        answered[0,j]=1
        p_new=model.predict_proba(answered.tocsr())[0]
        results.append({"Technology":tech,"Question":group,
                        "Skill-only improvement (bucket ranks)":float(ranks@(p_new-p_answered)),
                        "Full-profile improvement (bucket ranks)":float(ranks@(p_new-baseline)),
                        "Training respondents":support,
                        "Training support (%)":round(100*support/b["N"],2)})
    return pd.DataFrame(results).sort_values("Skill-only improvement (bucket ranks)",ascending=False)


def main():
    import streamlit as st
    import matplotlib.pyplot as plt
    st.set_page_config(page_title="Developer salary-bucket explorer",layout="wide")
    st.title("Developer salary-bucket explorer")
    st.caption("MIE1624 Assignment 2 bonus | Original survey answers, 15 ordered salary brackets, trained cumulative-threshold logistic regression")
    st.info("Educational model, not a salary quotation. Predicted associations are not causal wage effects.")
    path=next((fn for fn in ["processed_data_assignment2_2026.csv","processed_data_assignment2_2026(1).csv"] if os.path.isfile(fn)),None)
    if path:
        st.caption(f"Using unchanged local survey data: {path}")
        @st.cache_resource(show_spinner="Training the 14 threshold models...")
        def cache_path(p):return train_bundle(pd.read_csv(p,low_memory=False))
        bundle=cache_path(path)
    else:
        upload=st.file_uploader("Upload unchanged processed_data_assignment2_2026.csv",type="csv")
        if upload is None:st.stop()
        @st.cache_resource(show_spinner="Training the 14 threshold models...")
        def cache_bytes(content):
            from io import BytesIO
            return train_bundle(pd.read_csv(BytesIO(content),low_memory=False))
        bundle=cache_bytes(upload.getvalue())
    train=bundle["train"]
    st.subheader("1. Original survey answers")
    a={}
    def categorical(col, label, default=None):
        vals=sorted(set(train[col].dropna().astype(str)))
        choices=["Not answered"]+vals
        k=choices.index(default) if default in choices else 0
        selected=st.selectbox(label,choices,index=k)
        a[col]=np.nan if selected=="Not answered" else selected
    left,right=st.columns(2)
    with left:
        categorical("Country","Country","Canada")
        a["YearsCode"]=st.number_input("Years of coding experience",0,60,2)
        a["WorkExp"]=st.number_input("Years of professional experience",0,60,3)
        tool_unknown=st.checkbox("ToolCount unanswered",value=True)
        a["ToolCount"]=np.nan if tool_unknown else st.number_input("Tool count",0,200,7)
        categorical("Role","Role","Academic researcher")
        categorical("RoleFunctions","Role functions")
    with right:
        categorical("ICorPM","Individual contributor or manager","Individual contributor")
        categorical("OrgSize","Organization size")
        categorical("Education","Highest education")
        for g,label in [("LanguagesHaveWorkedWith","Programming languages"),
                        ("DatabasesHaveWorkedWith","Databases"),
                        ("CloudPlatHaveWorkedWith","Cloud and build platforms")]:
            missing=st.checkbox(f"{label}: unanswered",value=True,key=g+"_unknown")
            options=sorted([v.split("::",1)[1] for v in bundle["selected_names"] if v.startswith(g+"::") and not v.endswith("::Missing")])
            chosen=st.multiselect(label,options,disabled=missing,key=g+"_values")
            a[g]=np.nan if missing else ";".join(chosen)
    x=original_to_X(a,bundle);p=bundle["model"].predict_proba(x)[0]
    assert np.isclose(p.sum(),1) and np.all(p>=0)
    st.subheader("2. Predicted salary distribution")
    col1,col2=st.columns([1,2])
    with col1:
        st.metric("Most likely annual salary bucket",ORDER[p.argmax()])
        st.metric("Probability of modal bracket",f"{100*p.max():.1f}%")
        st.metric("Expected ordered bucket index",f"{np.arange(1,16)@p:.2f} / 15")
    with col2:
        fig,ax=plt.subplots(figsize=(10,3.3))
        ax.bar(np.arange(15),100*p)
        ax.set_xticks(np.arange(15),ORDER,rotation=55,ha="right",fontsize=8)
        ax.set_ylabel("Predicted probability (%)")
        fig.tight_layout();st.pyplot(fig);plt.close(fig)
    st.subheader("3. Skills and technology scenario ranking")
    min_count=st.slider("Minimum respondents reporting the technology in training (sensitivity only)",0,500,0,25)
    ranking=rank_skills(a,bundle,min_support=min_count)
    if ranking.empty:st.warning("No eligible technology indicators meet this support threshold.")
    else:
        st.dataframe(ranking.head(15).round(3),hide_index=True,use_container_width=True)
        st.caption("Main ranking is the skill-only expected bucket-index shift; the full-profile shift also includes the move from unanswered to answered. Brackets have unequal dollar widths. Counts are support diagnostics, not confidence intervals.")
    st.divider()
    st.caption("Model inputs are strictly the 12 original survey question groups selected in the assignment. Training preprocessing and ordinal models are fitted in memory; the survey CSV remains unchanged.")

if __name__=="__main__":main()

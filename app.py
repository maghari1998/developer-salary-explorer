"""MIE1624 Assignment 2: public Streamlit bonus dashboard.
This deployment loads a fitted model but never exposes the raw survey CSV.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from assignment2_bonus_dashboard import ORDER, original_to_X, rank_skills

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title='Developer Salary Explorer | MIE1624',
                   page_icon='📊', layout='wide')

@st.cache_resource(show_spinner='Loading trained salary model...')
def load_bundle():
    model_path = HERE / 'model_bundle.joblib'
    if not model_path.exists():
        raise FileNotFoundError('model_bundle.joblib not found. Add all deployment files to the GitHub repository root.')
    return joblib.load(model_path)

b = load_bundle()
st.title('Developer Salary Explorer')
st.caption('MIE1624 · Assignment 2 · 2026 Stack Overflow Developer Survey · Ordinal Logistic Regression')
st.info('Educational predictions, not employment or salary advice. Associations do not establish that learning a technology causes higher pay.')

with st.expander('How to use this dashboard', expanded=False):
    st.write('Enter your **original survey responses** below. The model will predict a probability across all 15 annual USD salary brackets and show hypothetical technology scenarios. You can leave questions unanswered.')
    st.write('The model was fitted on employed / self-employed survey respondents, and is less reliable for unusual profiles and rare salary brackets. The output is not a personalized salary offer.')

st.subheader('1. Your survey answers')
a={}

def categorical(col, label, default=None):
    options=['Not answered'] + b['choices'][col]
    index=options.index(default) if default in options else 0
    answer=st.selectbox(label, options, index=index)
    a[col]=np.nan if answer=='Not answered' else answer

left,right=st.columns(2)
with left:
    categorical('Country','Country','Canada')
    a['YearsCode']=st.number_input('Years of coding experience',min_value=0,max_value=60,value=2,step=1)
    a['WorkExp']=st.number_input('Years of professional work experience',min_value=0,max_value=60,value=3,step=1)
    unknown_count=st.checkbox('Tool count unanswered', value=True)
    if unknown_count:
        a['ToolCount']=np.nan
    else:
        a['ToolCount']=st.number_input('Number of tools used',min_value=0,max_value=200,value=7,step=1)
    categorical('Role','Primary role','Academic researcher')
    # Role functions are a true multi-select survey question (not a single category).
    role_unanswered=st.checkbox('Role functions: unanswered',value=True,key='RoleFunctions_unknown')
    role_options=sorted({name.split('::',1)[1] for name in b['selected_names']
                         if name.startswith('RoleFunctions::') and not name.endswith('::Missing')})
    role_selected=st.multiselect('Role functions',role_options,disabled=role_unanswered,key='RoleFunctions_selected')
    a['RoleFunctions']=np.nan if role_unanswered else ';'.join(role_selected)
with right:
    categorical('ICorPM','Individual contributor or manager','Individual contributor')
    categorical('OrgSize','Organization size')
    categorical('Education','Highest education',"Master's degree (M.A., M.S., M.Eng., MBA, etc.)")
    for col,label in [('LanguagesHaveWorkedWith','Programming languages'),
                      ('DatabasesHaveWorkedWith','Databases'),
                      ('CloudPlatHaveWorkedWith','Cloud/build platforms')]:
        unanswered=st.checkbox(f'{label}: unanswered',value=True,key=col+'_unknown')
        options=sorted({name.split('::',1)[1] for name in b['selected_names']
                        if name.startswith(col+'::') and not name.endswith('::Missing')})
        selected=st.multiselect(label,options,disabled=unanswered,key=col+'_selected')
        a[col]=np.nan if unanswered else ';'.join(selected)

x=original_to_X(a,b)
p=b['model'].predict_proba(x)[0]
if not np.isclose(p.sum(),1) or np.any(p<0):
    st.error('Invalid predicted probabilities. Please check the model bundle.')
    st.stop()

st.subheader('2. Salary-bucket prediction')
metric_col, plot_col=st.columns([1,2],gap='large')
with metric_col:
    st.metric('Most likely annual salary bracket (USD)',ORDER[int(np.argmax(p))])
    st.metric('Probability of that bracket',f'{100*np.max(p):.1f}%')
    st.metric('Expected ordered salary-bucket rank',f'{np.arange(1,16)@p:.2f} / 15')
    st.caption('Expected rank is not an expected salary in dollars. Salary brackets have different widths.')
with plot_col:
    fig,ax=plt.subplots(figsize=(11,3.7))
    ax.bar(np.arange(15),100*p,color='#3976aa')
    ax.set_xticks(np.arange(15),ORDER,rotation=60,ha='right',fontsize=8)
    ax.set_ylabel('Predicted probability (%)')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y',alpha=.15)
    fig.tight_layout()
    st.pyplot(fig,clear_figure=True)
    plt.close(fig)

st.subheader('3. Skill & technology recommendations')
st.write('The ranking simulates adding **one** technology to the answers above and measures the model-predicted change in expected salary-bucket rank. This is not a causal estimate.')
min_support=st.slider('Minimum number of training respondents reporting a technology',0,500,0,25)
ranking=rank_skills(a,b,min_support=min_support)
if ranking.empty:
    st.warning('No eligible technologies meet this support threshold. Lower the threshold or adjust the answered technologies.')
else:
    shown=ranking.head(15).copy()
    st.dataframe(shown.round(3),hide_index=True,use_container_width=True)
    st.caption('“Skill-only” change compares answered-without-skill vs. answered-with-skill. “Full-profile” change includes replacing an unanswered question. Training counts measure support, not statistical confidence.')
    top=shown.head(3)
    st.markdown('**Top three under the selected criterion:** '+', '.join(str(s) for s in top['Technology'].tolist()))

st.divider()
st.caption('Predictions use the same selected original survey question groups as the submitted analysis. No raw survey CSV or individual survey records are included in this hosted application.')

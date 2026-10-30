import pandas as pd, numpy as np
A='../results/'
K=['Coverage fine (%)','Overlap fine (%)','EE fine (%)']
def rd(x,s): return pd.read_excel(A+x,sheet_name=s)
p2=rd('robustness_studies/p2_results_studyA.xlsx','all_layouts'); p2r=pd.read_csv(A+'robustness_studies/p2_runs_studyA.csv')
p3=rd('robustness_studies/p3_results_studyB.xlsx','all_layouts'); p3r=pd.read_csv(A+'robustness_studies/p3_runs_studyB.csv')
p4=rd('robustness_studies/p4_results_studyC.xlsx','all_layouts'); p4r=pd.read_csv(A+'robustness_studies/p4_runs_studyC.csv')
p5=pd.read_csv(A+'main_setting/p5_runs.csv')
p9a=pd.read_csv(A+'robustness_studies/p9a_layout_metrics.csv'); p9ar=pd.read_csv(A+'robustness_studies/p9a_runs.csv')
p9b=pd.read_csv(A+'robustness_studies/p9bc_layout_metrics.csv'); p9br=pd.read_csv(A+'robustness_studies/p9bc_runs.csv')
def tag(L,R,study,src,keys=('Method','Seed')):
    L=L.copy(); R=R.copy()
    if R is L: m=L.copy()
    else: m=L.drop(columns=['Runtime (s)'],errors='ignore').merge(R[list(keys)+['Runtime (s)']],on=list(keys),how='left')
    m['Study']=study; m['Source']=src
    if 'Wasted sensing (% of field)' not in m: m['Wasted sensing (% of field)']=np.nan
    return m[['Study','Method','Seed']+K+['Wasted sensing (% of field)','Runtime (s)','Source']]
parts=[]
# Study A
parts.append(tag(p2[~p2.Method.str.startswith('PROPOSED_M')],p2r,'A','p2'))
q=p5[(p5.Method!='PROPOSED')&p5.Seed.between(20,39)]
parts.append(tag(q,q,'A','p5'))
parts.append(tag(p9a,p9ar,'A','p9a'))
# Study B
parts.append(tag(p3,p3r,'B','p3'))
parts.append(tag(p4[p4.Field=='R'],p4r[p4r.Field=='R'],'B','p4'))
parts.append(tag(p9b[p9b.Study=='B'],p9br[p9br.Study=='B'],'B','p9bc'))
# Study C
parts.append(tag(p4[p4.Field=='I'],p4r[p4r.Field=='I'],'C','p4'))
parts.append(tag(p9b[p9b.Study=='C'],p9br[p9br.Study=='C'],'C','p9bc'))
M=pd.concat(parts,ignore_index=True)
dup=M[M.duplicated(['Study','Method','Seed'],keep=False)].sort_values(['Study','Method','Seed'])
M=M.drop_duplicates(['Study','Method','Seed'],keep='first')
M['Feasible fine']=(M['Coverage fine (%)']>98)&(M['Overlap fine (%)']<10)&(M['EE fine (%)']>97)
M.to_csv('merged_18.csv',index=False)
print('duplicates (kept first source):'); print(dup[['Study','Method','Seed','EE fine (%)','Source']].to_string())
SEEDS={'A':range(20,40),'B':range(20,35),'C':range(20,30)}
for s,r in SEEDS.items():
    d=M[M.Study==s]; t=d.pivot_table(index='Method',columns='Seed',values='EE fine (%)',aggfunc='count')
    print(s,'methods',d.Method.nunique(), 'complete:', all(t.reindex(columns=list(r)).notna().all(axis=1)), 'extra seeds', sorted(set(d.Seed)-set(r)))
    miss=t.reindex(columns=list(r)).isna(); print([ (m,list(miss.columns[miss.loc[m]])) for m in miss.index if miss.loc[m].any()])

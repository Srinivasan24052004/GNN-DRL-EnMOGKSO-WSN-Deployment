import pandas as pd, numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
M=pd.read_csv('merged_18.csv')
LAB={"PROPOSED":"Proposed (GNN-DQN)","DQN_ONLY":"DQN only (no GNN)","CNN_DQN":"CNN-DQN","QLEARN":"Tabular Q-learning",
 "RANDOM_MOVE":"Random move type","BANDIT_UCB":"UCB bandit","ORIGINAL":"Hand-crafted ARLS","SA_PLAIN":"SA (hand-crafted)",
 "SA_LEARNED":"SA (GNN-DQN)","CVT":"Lloyd-CVT","PSO":"PSO","GA":"GA","DE":"DE","GWO":"GWO","WOA":"WOA","VFA":"VFA",
 "GKSO":"GKSO","WOA_LFGA":"WOA-LFGA"}
MET=[('EE fine (%)',True),('Overlap fine (%)',False),('Coverage fine (%)',True)]
def holm(p):
    p=np.asarray(p); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for k,i in enumerate(o):
        run=max(run,(m-k)*p[i]); adj[i]=min(1,run)
    return adj
out={}
for st in 'ABC':
    d=M[M.Study==st]
    g=d.groupby('Method')
    desc=pd.DataFrame({'Runs':g.Seed.count(),
        'C mean':g['Coverage fine (%)'].mean(),'C sd':g['Coverage fine (%)'].std(),
        'O mean':g['Overlap fine (%)'].mean(),'O sd':g['Overlap fine (%)'].std(),
        'EE mean':g['EE fine (%)'].mean(),'EE sd':g['EE fine (%)'].std(),'EE best':g['EE fine (%)'].max(),
        'Wasted':g['Wasted sensing (% of field)'].mean(),'Feasible':g['Feasible fine'].sum(),'Runtime':g['Runtime (s)'].mean()})
    piv={k:d.pivot(index='Seed',columns='Method',values=k) for k,_ in MET}
    E=piv['EE fine (%)']
    R=np.vstack([rankdata(-row) for row in E.values]); desc['Mean rank (EE)']=pd.Series(R.mean(0),index=E.columns)
    fr=friedmanchisquare(*[E[c].values for c in E.columns])
    desc=desc.sort_values('EE mean',ascending=False); desc.insert(0,'Label',[LAB[m] for m in desc.index])
    rows=[]
    for k,hi in MET:
        P=piv[k]; ps=[]; tmp=[]
        for m in desc.index:
            if m=='PROPOSED': continue
            diff=P['PROPOSED']-P[m]; better=(diff>1e-9) if hi else (diff<-1e-9); worse=(diff<-1e-9) if hi else (diff>1e-9)
            try: p=wilcoxon(P['PROPOSED'],P[m]).pvalue
            except ValueError: p=1.0
            tmp.append([k,m,LAB[m],len(diff),P['PROPOSED'].mean(),P[m].mean(),diff.mean(),f"{int(better.sum())}/{int((~better&~worse).sum())}/{int(worse.sum())}",p]); ps.append(p)
        adj=holm(ps)
        for t,a in zip(tmp,adj):
            sig=a<0.05; good=(t[6]>0) if hi else (t[6]<0)
            t+= [a, ('Proposed better' if good else 'Proposed worse') if sig else 'No significant difference']; rows.append(t)
    tests=pd.DataFrame(rows,columns=['Metric','Method','Label','Seeds','Mean proposed','Mean other','Mean diff','Proposed better/tie/worse','p','p Holm','Verdict'])
    out[st]=(desc,tests,fr)
    print(f'\n=== STUDY {st}: seeds {E.index.min()}-{E.index.max()} (n={len(E)}), Friedman chi2={fr.statistic:.1f}, p={fr.pvalue:.2e}')
    print(desc.round(2).to_string())
    print(tests[tests.Metric=='EE fine (%)'][['Label','Mean diff','Proposed better/tie/worse','p','p Holm','Verdict']].round(4).to_string())
    print(tests[tests.Metric!='EE fine (%)'][['Metric','Label','Mean diff','p Holm','Verdict']].query("Verdict!='Proposed better'").round(4).to_string())
with pd.ExcelWriter('Studies_ABC_18methods_results.xlsx') as w:
    M.to_excel(w,sheet_name='all_runs',index=False)
    for st,(desc,tests,fr) in out.items():
        desc.to_excel(w,sheet_name=f'{st}_descriptive'); tests.to_excel(w,sheet_name=f'{st}_wilcoxon_holm',index=False)
        pd.DataFrame({'Chi2':[fr.statistic],'p':[fr.pvalue]}).to_excel(w,sheet_name=f'{st}_friedman',index=False)
import pickle; pickle.dump(out,open('stats.pkl','wb'))

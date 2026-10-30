import os; os.makedirs('figs', exist_ok=True)
import pandas as pd, numpy as np, pickle
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Liberation Serif","DejaVu Serif"],
 "font.size":8.5,"axes.linewidth":0.8,"xtick.direction":"in","ytick.direction":"in","ytick.right":True,"savefig.dpi":600})
M=pd.read_csv('merged_18.csv')
ORDER=["PROPOSED","DQN_ONLY","CNN_DQN","QLEARN","RANDOM_MOVE","BANDIT_UCB","ORIGINAL","SA_PLAIN","SA_LEARNED",
       "CVT","VFA","PSO","GA","DE","GWO","WOA","GKSO","WOA_LFGA"]
LAB={"PROPOSED":"Proposed (GNN-DQN)","DQN_ONLY":"DQN only (no GNN)","CNN_DQN":"CNN-DQN","QLEARN":"Q-learning",
 "RANDOM_MOVE":"Random move type","BANDIT_UCB":"UCB bandit","ORIGINAL":"Hand-crafted ARLS","SA_PLAIN":"SA (hand-crafted)",
 "SA_LEARNED":"SA (GNN-DQN)","CVT":"Lloyd-CVT","PSO":"PSO","GA":"GA","DE":"DE","GWO":"GWO","WOA":"WOA","VFA":"VFA",
 "GKSO":"GKSO","WOA_LFGA":"WOA-LFGA"}
COL={"PROPOSED":"#2a78d6","DQN_ONLY":"#0d4f99","CNN_DQN":"#c0392b","QLEARN":"#16a085","RANDOM_MOVE":"#8a8984",
 "BANDIT_UCB":"#1baf7a","ORIGINAL":"#eb6834","SA_PLAIN":"#eda100","SA_LEARNED":"#17becf","CVT":"#52514e","VFA":"#e34948",
 "PSO":"#4a3aa7","GA":"#e87ba4","DE":"#008300","GWO":"#7a5c2e","WOA":"#9b2d2d","GKSO":"#7a3e9d","WOA_LFGA":"#b8860b"}
def boxes(ax,L,keys,col,ylab):
    data=[L[L.Method==k][col].dropna().values for k in keys]
    bp=ax.boxplot(data,positions=range(len(keys)),widths=0.55,showfliers=True,flierprops=dict(marker="o",ms=2.6,mfc="white",mew=0.6))
    for i,k in enumerate(keys):
        c=COL[k]; bp["boxes"][i].set(color=c,lw=1.0); bp["medians"][i].set(color=c,lw=1.2)
        for part in ("whiskers","caps"): bp[part][2*i].set(color=c); bp[part][2*i+1].set(color=c)
        bp["fliers"][i].set(markeredgecolor=c); ax.plot(i,np.mean(data[i]),marker="D",ms=2.6,color=c,zorder=3)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([LAB[k] for k in keys],fontsize=6,rotation=45,ha="right")
    ax.set_ylabel(ylab); ax.set_xlim(-0.6,len(keys)-0.4)
for st,name in [('A','Fig_StudyA_18methods'),('B','Fig_StudyB_18methods'),('C','Fig_StudyC_18methods')]:
    L=M[M.Study==st]
    fig,axs=plt.subplots(2,2,figsize=(9.5,6.2))
    boxes(axs[0,0],L,ORDER,'EE fine (%)','EE fine (%)'); axs[0,0].set_title('(a) EE, all 18 methods',loc='left',fontsize=9)
    boxes(axs[0,1],L,ORDER[:9],'EE fine (%)','EE fine (%)'); axs[0,1].set_title('(b) EE, the nine EnMOGKSO variants (zoom)',loc='left',fontsize=9)
    boxes(axs[1,0],L,ORDER,'Overlap fine (%)','Overlap fine (%)'); axs[1,0].set_title('(c) Overlap',loc='left',fontsize=9)
    boxes(axs[1,1],L,ORDER,'Coverage fine (%)','Coverage fine (%)'); axs[1,1].set_title('(d) Coverage',loc='left',fontsize=9)
    fig.tight_layout(h_pad=1.4,w_pad=1.6); fig.savefig(f'figs/{name}.png',dpi=600,facecolor='white',bbox_inches='tight'); plt.close(fig)
# Friedman mean ranks
out=pickle.load(open('stats.pkl','rb'))
R=pd.DataFrame({st:out[st][0]['Mean rank (EE)'] for st in 'ABC'}).reindex(ORDER)
R=R.loc[R.mean(axis=1).sort_values().index]
fig,ax=plt.subplots(figsize=(6.6,5.4))
y=np.arange(len(R)); h=0.26
for j,(st,c,lab) in enumerate([('A','#2a78d6','Study A (main setting, 20 seeds)'),('B','#eb6834','Study B (strict scoring, 15 seeds)'),('C','#1baf7a','Study C (irregular field, 10 seeds)')]):
    ax.barh(y+(j-1)*h,R[st].values,height=h,color=c,label=lab,edgecolor='none')
ax.set_yticks(y); ax.set_yticklabels([LAB[k] for k in R.index],fontsize=7.5); ax.invert_yaxis()
for t,k in zip(ax.get_yticklabels(),R.index):
    if k=='PROPOSED': t.set_fontweight('bold')
ax.set_xlabel('Friedman mean rank on EE (1 = best, 18 = worst)'); ax.set_xlim(0,18.5); ax.tick_params(axis='y',right=False)
ax.xaxis.grid(True,lw=0.4,color='#dddddd'); ax.set_axisbelow(True)
ax.legend(frameon=False,fontsize=7.5,loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=2); ax.set_xticks(range(0,19,2))
fig.tight_layout(); fig.savefig('figs/Fig_Studies_ABC_ranks.png',dpi=600,facecolor='white',bbox_inches='tight'); plt.close(fig)
print(R.round(2))

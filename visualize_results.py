import json
import matplotlib.pyplot as plt
import numpy as np

with open('results.json','r') as f:
    results=json.load(f)

feature_types=list(results.keys())
classifiers=['Naive Bayes','Decision Tree','Random Forest']
metrics=['accuracy','precision','recall','f1_score']

fig,axes=plt.subplots(2,2,figsize=(14,10))
fig.suptitle('Classifier Performance Comparison',fontsize=16)

for idx,metric in enumerate(metrics):
    row=idx//2
    col=idx%2
    ax=axes[row,col]
    
    x_pos=np.arange(len(feature_types))
    width=0.25
    
    for i,clf in enumerate(classifiers):
        values=[]
        for ft in feature_types:
            values.append(results[ft][clf][metric])
        
        offset=width*i
        ax.bar(x_pos+offset,values,width,label=clf)
    
    ax.set_xlabel('Feature Type')
    ax.set_ylabel(metric.replace('_',' ').title())
    ax.set_title(metric.replace('_',' ').title())
    ax.set_xticks(x_pos+width)
    ax.set_xticklabels(feature_types,rotation=15)
    ax.legend()
    ax.set_ylim([0.85,1.05])
    ax.grid(axis='y',alpha=0.3)

plt.tight_layout()

fig2,ax2=plt.subplots(figsize=(10,6))
training_times=[]
labels=[]

for ft in feature_types:
    for clf in classifiers:
        time_val=results[ft][clf]['training_time']
        training_times.append(time_val)
        labels.append(f"{ft}\n{clf}")

colors=['#FF6B6B','#4ECDC4','#45B7D1','#FFA07A','#98D8C8','#6C5CE7','#A29BFE','#FD79A8','#FDCB6E']
bars=ax2.bar(range(len(training_times)),training_times,color=colors[:len(training_times)])
ax2.set_xlabel('Feature Type and Classifier')
ax2.set_ylabel('Training Time (seconds)')
ax2.set_title('Training Time Comparison')
ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(labels,rotation=45,ha='right')
ax2.grid(axis='y',alpha=0.3)

plt.tight_layout()

print("\nDetailed Results Summary:")
print("="*60)
for ft in feature_types:
    print(f"\n{ft.upper()}")
    print("-"*60)
    for clf in classifiers:
        print(f"  {clf}:")
        print(f"    Accuracy:  {results[ft][clf]['accuracy']:.4f}")
        print(f"    F1-Score:  {results[ft][clf]['f1_score']:.4f}")
        print(f"    Time:      {results[ft][clf]['training_time']:.4f}s")
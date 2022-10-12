import numpy as np
from utils.scoring_utils import score_dataset
import matplotlib.pyplot as plt

score = np.load('/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation/test_high/nofilterbbox/stc/Jan14_2322/checkpoints/0_reco_loss.npy')
metadata = np.load('/home/yaboliu/work/research/gepc/gepc_new1/work_dir_ablation/test_high/nofilterbbox/stc/Jan14_2322/checkpoints/test_dataset_meta.npy')
# print(score[:100], metadata[:100])
auc, shift, sigma, fpr, tpr = score_dataset(score, metadata, level='high')
# fpr, tpr = plot_material
# roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('auc_test.jpg')
plt.close()

exit(-1)
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings;warnings.filterwarnings('ignore')
dataset = load_breast_cancer()
data = dataset.data
target = dataset.target
X_train,X_test,y_train,y_test = train_test_split(data,target,test_size=0.2)
rf = RandomForestClassifier(n_estimators=5)
rf.fit(X_train,y_train)
pred = rf.predict_proba(X_test)[:,1]
#############画图部分
print(y_test)
print(pred)
fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)
plt.figure(figsize=(6,6))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('auc_test.jpg')
# plt.show()

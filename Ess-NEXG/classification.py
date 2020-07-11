import numpy as np
import xgboost as xgb
from itertools import chain
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


protein_x_0 = np.load("unweighted/BioGRID/bio_protein_emb_PCC.npy")
protein_label_0 = np.load("unweighted/BioGRID/bio_protein_label.npy")  # 存0和1，和bio_protein_emb.npy中的节点对应

protein_x_1 = np.load("data3/xgboost/bio_protein_emb_PCC(4).npy")
protein_label_1 = np.load("data3/xgboost/bio_protein_label(4).npy")

#print(protein_label)
#print(protein_x.shape, protein_label.shape)
train_number = int(0.8*len(protein_x_0))  # 80%的节点作为训练集
#print(train_number)

dataset_train = protein_x_0[:train_number]  # 训练集
datalabels_train = protein_label_0[:train_number]
dataset_test = protein_x_0[train_number:]  # 测试集
datalabels_test = protein_label_0[train_number:]

dataset_xunlian = protein_x_1[:train_number]  # 训练集
datalabels_xunlian = protein_label_1[:train_number]
dataset_ceshi = protein_x_1[train_number:]  # 测试集
datalabels_ceshi = protein_label_1[train_number:]

# SVM
clf = SVC(probability=True, kernel='rbf')  # probability：是否采用概率估计，默认为False, kernel：核函数
clf.fit(dataset_train, datalabels_train)  # 用训练数据拟合分类器模型
#print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#print(clf.support_vectors_)  # 支持向量
#print(len(clf.support_))  # clf.support_  支持向量是哪几个(下标)
support_index = clf.support_
support_index .tolist()
# for ind in range(len(support_index)):
#     print(support_index[ind])
#print(clf.n_support_)  # 每一类中有几个支持向量

predicted = clf.predict(dataset_test)  # 测试数据
# plot ROC
probas_ = clf.predict_proba(dataset_test)  # 返回预测属于某标签的概率,两个类
# SVM的ROC曲线
fpr_SVM, tpr_SVM, thresholds_SVM = metrics.roc_curve(datalabels_test, probas_[:, 1])  # probas_[:, 1]:取第二列的数据
roc_auc_SVM = metrics.auc(fpr_SVM, tpr_SVM)
#plt.plot(fpr_SVM, tpr_SVM, lw=1, label='SVM ')

result = clf.decision_function(dataset_test)
print('svm accuracy:', clf.score(dataset_test, datalabels_test))
print(metrics.classification_report(datalabels_test, predicted,))
print("ROC:", roc_auc_SVM)
# print(metrics.confusion_matrix(datalabels_test, predicted))


# DT
print("this is result of decision tree:")
clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit(dataset_train, datalabels_train)
predicted1 = clf1.predict(dataset_test)
probas_ = clf1.predict_proba(dataset_test)
fpr_DT, tpr_DT, thresholds_DT = metrics.roc_curve(datalabels_test, probas_[:, 1])
roc_auc_DT = metrics.auc(fpr_DT, tpr_DT)
#plt.plot(fpr_DT, tpr_DT, lw=1, label='Decision Tree ')

print("roc_auc_DT", roc_auc_DT)
fpr, tpr, thresholds = metrics.roc_curve(datalabels_test, probas_[:,1])

print('decision tree accuracy:', clf1.score(dataset_test, datalabels_test))
print(metrics.classification_report(datalabels_test, predicted1,))
print(metrics.confusion_matrix(datalabels_test, predicted1))



# random forest
print("this is result of random forest:")
clf2 = RandomForestClassifier(n_estimators=10, min_samples_leaf =3)
clf2 = clf2.fit(dataset_train, datalabels_train)
predicted2 = clf2.predict(dataset_test)
probas_ = clf2.predict_proba(dataset_test)
fpr_RF, tpr_RF, thresholds_DRF = metrics.roc_curve(datalabels_test, probas_[:, 1])
roc_auc_RF = metrics.auc(fpr_RF, tpr_RF)
#plt.plot(fpr_RF, tpr_RF, lw=1, label='Random Forest' )

print("roc_auc_RF", roc_auc_RF)
print('random forest accuracy:', clf2.score(dataset_test, datalabels_test))
print(metrics.classification_report(datalabels_test, predicted2,))
# print(metrics.confusion_matrix(datalabels_test, predicted2))


# adaboost
print("this is result of adaboost:")
clf3 = AdaBoostClassifier(n_estimators=100)
clf3 = clf3.fit(dataset_train, datalabels_train)
predicted3 = clf3.predict(dataset_test)
probas_ = clf3.predict_proba(dataset_test)
fpr_ADA, tpr_ADA, thresholds_ADA = metrics.roc_curve(datalabels_test, probas_[:, 1])
roc_auc_ADA = metrics.auc(fpr_ADA, tpr_ADA)
#plt.plot(fpr_ADA, tpr_ADA, lw=1, label='Adaboost' )

print("roc_auc_ADA", roc_auc_ADA)
print('adaboost accuracy:', clf3.score(dataset_test, datalabels_test))
print(metrics.classification_report(datalabels_test, predicted3,))
# print(metrics.confusion_matrix(datalabels_test, predicted3))

#  naive bayes
print("this is result of naive bayes:")
dataset_train_float = []
dataset_test_float = []
for i in range(len(dataset_train)):
    ans = []
    for j in dataset_train[i]:
        ans.append(float(j))
    dataset_train_float.append(ans)

for i in range(len(dataset_test)):
    ans = []
    for j in dataset_test[i]:
        ans.append(float(j))
    dataset_test_float.append(ans)

clf4 = GaussianNB()
clf4 = clf4.fit(dataset_train_float, list(chain.from_iterable(datalabels_train)))

predicted4 = clf4.predict(dataset_test_float)
probas_ = clf4.predict_proba(dataset_test_float)
fpr_NB, tpr_NB, thresholds_NB = metrics.roc_curve(datalabels_test, probas_[:, 1])
roc_auc_NB = metrics.auc(fpr_NB, tpr_NB)
#plt.plot(fpr_NB, tpr_NB, lw=1, label='naive bayes' )


print("roc_auc_NB", roc_auc_NB)
print('naive bayes accuracy:', clf4.score(dataset_test_float, datalabels_test))
print(metrics.classification_report(datalabels_test, predicted4,))
# print(metrics.confusion_matrix(datalabels_test, predicted4))


#  xgboost
print("Xgboost")
seed = 2
test_size = 0.2
train_x, test_x, train_y, test_y = train_test_split(protein_x_0, protein_label_0, test_size=test_size, random_state=seed)

dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x)

# 参数设置
params = {'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 4,
    'lambda': 3,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'min_child_weight': 4,
    'eta': 0.1414,
    'seed': 0,
    'nthread': 6,
     'silent': 1}

bst = xgb.train(params, dtrain, num_boost_round=100)
ypred = bst.predict(dtest)
y_pred = (ypred >= 0.5)*1

# ROC曲线下与坐标轴围成的面积
print('AUC: %.4f' % metrics.roc_auc_score(test_y, ypred))
# 准确率
print('ACC: %.4f' % metrics.accuracy_score(test_y, y_pred))
print(metrics.classification_report(test_y, y_pred))

fpr_DT, tpr_DT, thresholds_DT = metrics.roc_curve(list(chain.from_iterable(test_y)), ypred)
#plt.plot(fpr_DT, tpr_DT, lw=1, label='xgboost')




# SVM
clf5 = SVC(probability=True, kernel='rbf')
clf5.fit(dataset_xunlian, datalabels_xunlian)
support_index = clf5.support_
support_index .tolist()
predicted5 = clf5.predict(dataset_ceshi)
probas_ = clf5.predict_proba(dataset_ceshi)
# SVM的ROC曲线
fpr_1, tpr_1, thresholds_1 = metrics.roc_curve(datalabels_ceshi, probas_[:, 1])  # probas_[:, 1]:取第二列的数据
roc_auc_1 = metrics.auc(fpr_1, tpr_1)
plt.plot(fpr_1, tpr_1, lw=1, label='SVM_1 ')

result = clf5.decision_function(dataset_test)



# DT
print("this is result of decision tree:")
clf6 = tree.DecisionTreeClassifier()
clf6 = clf6.fit(dataset_xunlian, datalabels_xunlian)
predicted6 = clf6.predict(dataset_ceshi)
probas_ = clf6.predict_proba(dataset_ceshi)
fpr_2, tpr_2, thresholds_2 = metrics.roc_curve(datalabels_ceshi, probas_[:, 1])
roc_auc_2 = metrics.auc(fpr_2, tpr_2)
plt.plot(fpr_2, tpr_2, lw=1, label='Decision Tree_1 ')



# random forest
print("this is result of random forest:")
clf7 = RandomForestClassifier(n_estimators=10, min_samples_leaf =3)
clf7 = clf7.fit(dataset_xunlian, datalabels_xunlian)
predicted7 = clf7.predict(dataset_ceshi)
probas_ = clf7.predict_proba(dataset_ceshi)
fpr_3, tpr_3, thresholds_3 = metrics.roc_curve(datalabels_ceshi, probas_[:, 1])
roc_auc_3 = metrics.auc(fpr_3, tpr_3)
plt.plot(fpr_3, tpr_3, lw=1, label='Random Forest_1' )



# adaboost
print("this is result of adaboost:")
clf8 = AdaBoostClassifier(n_estimators=100)
clf8 = clf8.fit(dataset_xunlian, datalabels_xunlian)
predicted8 = clf8.predict(dataset_ceshi)
probas_ = clf8.predict_proba(dataset_ceshi)
fpr_4, tpr_4, thresholds_4 = metrics.roc_curve(datalabels_ceshi, probas_[:, 1])
roc_auc_4 = metrics.auc(fpr_4, tpr_4)
plt.plot(fpr_4, tpr_4, lw=1, label='Adaboost_1' )


#  naive bayes
print("this is result of naive bayes:")
dataset_train_float = []
dataset_test_float = []
for i in range(len(dataset_xunlian)):
    ans = []
    for j in dataset_xunlian[i]:
        ans.append(float(j))
    dataset_train_float.append(ans)

for i in range(len(dataset_ceshi)):
    ans = []
    for j in dataset_ceshi[i]:
        ans.append(float(j))
    dataset_test_float.append(ans)

clf9 = GaussianNB()
clf9 = clf9.fit(dataset_train_float, list(chain.from_iterable(datalabels_train)))

predicted9 = clf9.predict(dataset_test_float)
probas_ = clf9.predict_proba(dataset_test_float)
fpr_5, tpr_5, thresholds_5 = metrics.roc_curve(datalabels_ceshi, probas_[:, 1])
roc_auc_5 = metrics.auc(fpr_5, tpr_5)
plt.plot(fpr_5, tpr_5, lw=1, label='naive bayes_1' )





#  xgboost
seed = 2
test_size = 0.2
train_1, test_1, train_2, test_2 = train_test_split(protein_x_1, protein_label_1, test_size=test_size, random_state=seed)

xtrain = xgb.DMatrix(train_1, label=train_2)
xtest = xgb.DMatrix(test_1)

# 参数设置
params = {'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 4,
    'lambda': 3,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'min_child_weight': 4,
    'eta': 0.1414,
    'seed': 0,
    'nthread': 6,
     'silent': 1}

bst_1 = xgb.train(params, xtrain, num_boost_round=100)
ypred_1 = bst.predict(xtest)
y_pred_1 = (ypred_1 >= 0.5)*1



fpr_6, tpr_6, thresholds_6 = metrics.roc_curve(list(chain.from_iterable(test_2)), ypred)
plt.plot(fpr_6, tpr_6, lw=1, label='xgboost_1')



plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


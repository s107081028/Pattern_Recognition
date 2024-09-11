import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTEENN
import random
random.seed(42)

def icbhi_score(cm):
    specificity = cm[1][1]/np.sum(cm[1])
    sensitivity = (cm[0][0]+cm[2][2])/(np.sum(cm[0])+np.sum(cm[2]))
    score = (sensitivity + specificity) / 2
    print(f"Specificity: {specificity}")
    print(f"Sensitivity: {sensitivity}")
    print(f"ICBHI score: {score}")
    
def draw_cm(cm, title):
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(cm_norm, cmap=plt.get_cmap('Blues'))
    ax.xaxis.set_ticklabels(['','Coarse', 'Normal', 'Wheeze'], fontdict = {'fontsize': 7})
    ax.yaxis.set_ticklabels(['','Coarse', 'Normal', 'Wheeze'], fontdict = {'fontsize': 7})
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            ax.text(x=j, y=i, s=round(cm_norm[i][j], 2), va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=10)
    plt.ylabel('Doctor-Labeled', fontsize=10)
    plt.title(title, fontsize=10)
    plt.show()
    plt.savefig(f"{title}.png")
    
for j in range(1, 6):
    file = open(f"labels_train_{j}.pkl",'rb')
    labels_train = pickle.load(file)
    file = open(f"labels_val_{j}.pkl",'rb')
    labels_val = pickle.load(file)
    file = open(f"embeddings_train_{j}.pkl",'rb')
    embeddings_train = pickle.load(file)
    file = open(f"embeddings_val_{j}.pkl",'rb')
    embeddings_val = pickle.load(file)
    X_train, X_val, y_train, y_val = embeddings_train.cpu(), embeddings_val.cpu(), labels_train, labels_val
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    print(X_train_scaled.shape)
    
    y_train = np.argmax(y_train,axis=1)
    y_val = np.argmax(y_val,axis=1)

    # PCA
    pca = PCA(n_components = 0.8)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    print(X_train_pca.shape)

    # LDA
    lda = LinearDiscriminantAnalysis('eigen', shrinkage=0.2)
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_val_lda = lda.transform(X_val_scaled)
    print(X_train_lda.shape)

    # DecisionTree
    rf = RandomForestClassifier(n_estimators=100, max_depth = 30, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_decision = rf.predict(X_train_scaled)
    importances = rf.feature_importances_
    forest_importances = pd.Series(importances)
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    X_train_dt = X_train_scaled[:,sorted(range(len(importances)), key=lambda k: importances[k], reverse=True)[:10]]
    X_val_dt = X_val_scaled[:,sorted(range(len(importances)), key=lambda k: importances[k], reverse=True)[:10]]
    print(X_train_dt.shape)
    
    sm = SMOTEENN()
    X_train_smote, y_train_smote = sm.fit_resample(X_train_scaled, y_train)
    sm = SMOTEENN()
    X_train_pca_smote, y_train_pca_smote = sm.fit_resample(X_train_pca, y_train)
    sm = SMOTEENN()
    X_train_lda_smote, y_train_lda_smote = sm.fit_resample(X_train_lda, y_train)
    sm = SMOTEENN()
    X_train_dt_smote, y_train_dt_smote = sm.fit_resample(X_train_dt, y_train)
    
    # ALL
    svm_model = svm.SVC()
    svm_model.fit(X_train_scaled, y_train)
    predicted = svm_model.predict(X_val_scaled)
    cm = confusion_matrix(y_val, predicted)
    print('Accuracy: %f' % accuracy_score(y_val, predicted))
    print(cm)
    icbhi_score(cm)
    draw_cm(np.array(cm), "SVM w/o SMOTE")
    
    # ALL SMOTE
    svm_model = svm.SVC()
    svm_model.fit(X_train_smote, y_train_smote)
    predicted = svm_model.predict(X_val_scaled)
    cm = confusion_matrix(y_val, predicted)
    print('Accuracy: %f' % accuracy_score(y_val, predicted))
    print(cm)
    icbhi_score(cm)
    draw_cm(np.array(cm), "SVM w/ SMOTE")
    
    # PCA
    svm_model = svm.SVC()
    svm_model.fit(X_train_pca, y_train)
    predicted = svm_model.predict(X_val_pca)
    cm = confusion_matrix(y_val, predicted)
    print('Accuracy: %f' % accuracy_score(y_val, predicted))
    print(cm)
    icbhi_score(cm)
    draw_cm(np.array(cm), "SVM PCA w/o SMOTE")

    # PCA SMOTE
    svm_model = svm.SVC()
    svm_model.fit(X_train_pca_smote, y_train_pca_smote)
    predicted = svm_model.predict(X_val_pca)
    cm = confusion_matrix(y_val, predicted)
    print('Accuracy: %f' % accuracy_score(y_val, predicted))
    print(cm)
    icbhi_score(cm)
    draw_cm(np.array(cm), "SVM PCA w/ SMOTE")

    # LDA
    svm_model = svm.SVC()
    svm_model.fit(X_train_lda, y_train)
    predicted = svm_model.predict(X_val_lda)
    cm = confusion_matrix(y_val, predicted)
    print('Accuracy: %f' % accuracy_score(y_val, predicted))
    print(cm)
    icbhi_score(cm)
    draw_cm(np.array(cm), "SVM LDA w/o SMOTE")

    # LDA SMOTE
    svm_model = svm.SVC()
    svm_model.fit(X_train_lda_smote, y_train_lda_smote)
    predicted = svm_model.predict(X_val_lda)
    cm = confusion_matrix(y_val, predicted)
    print('Accuracy: %f' % accuracy_score(y_val, predicted))
    print(cm)
    icbhi_score(cm)
    draw_cm(np.array(cm), "SVM LDA w/ SMOTE")

    # Decision Tree
    svm_model = svm.SVC()
    svm_model.fit(X_train_dt, y_train)
    predicted = svm_model.predict(X_val_dt)
    cm = confusion_matrix(y_val, predicted)
    print('Accuracy: %f' % accuracy_score(y_val, predicted))
    print(cm)
    icbhi_score(cm)
    draw_cm(np.array(cm), "SVM DT w/o SMOTE")

    # Decision Tree SMOTE
    svm_model = svm.SVC()
    svm_model.fit(X_train_dt_smote, y_train_dt_smote)
    predicted = svm_model.predict(X_val_dt)
    cm = confusion_matrix(y_val, predicted)
    print('Accuracy: %f' % accuracy_score(y_val, predicted))
    print(cm)
    icbhi_score(cm)
    draw_cm(np.array(cm), "SVM DT w/ SMOTE")
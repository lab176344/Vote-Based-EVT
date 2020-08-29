from sklearn.metrics import precision_recall_fscore_support
import scipy.io as sio
import numpy as np
score=[]
ThresholdScore=[]
ThresMacro=[]
FolderName='CIFAR_OpenSet\\'
score=[]
macroScore=[]
for j in range(1,6):
    loadName=FolderName+'fcheck'+str(j)+'.mat'
    dict=sio.loadmat(loadName)
    testy_PredMac=dict['outputknown']
    testy_GTMac=dict['outputTrue']
    testy_PredMac=np.transpose(testy_PredMac)
    testy_PredMac=np.reshape(testy_PredMac,[testy_PredMac.shape[0]])
    testy_GTMac=np.transpose(testy_GTMac)
    testy_GTMac=np.reshape(testy_GTMac,[testy_GTMac.shape[0]])
    precision,recall,fscore,_=precision_recall_fscore_support(testy_GTMac,testy_PredMac)
    print(np.mean(fscore))
    macroScore.append(np.mean(fscore))
print('Macro-F Score', np.mean(macroScore))

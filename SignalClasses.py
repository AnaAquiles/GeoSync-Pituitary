import numpy as np
from sklearn.cluster import KMeans

"""
        1)  SIGNAL CLASSIFICATION
       
                
"""

PowerSpec = spec    
Frequencies = freq      
cells = cells


def FrequencyCovariation_E(PowerSpec, Frequencies, cells):

    CovElec = []
    for i in range(cells):
        CovElec.append(np.cov(PowerSpec[i,:,:].T))
        
    CovElec = np.array(CovElec)

    #PCA per electrode
    eigval = np.zeros((cells,60))
    eigvec = np.zeros((cells,60,60))

    for i in range(cells):
        eigval[i,:], eigvec[i,:,:] = np.linalg.eigh(CovElec[i,:,:])

    indexes = np.flip(np.argsort(eigval[0,:]))
    eigval = eigval[:,indexes]
    eigvec = eigvec[:,:,indexes]

    maximum = np.max(np.abs(eigvec))
    minimum = np.min(np.abs(eigvec))
    Maximum = np.max(np.abs(CovElec))

    eigvecE = eigvec[:,:,0].T ## PC1
    eigvecE2 = eigvec[:,:,1].T ## PC2

    MatPC1 = eigvecE * Frequencies [:,None]    #eigval, frequency 
    MatPC2 = eigvecE2 * Frequencies[:,None]    #eigval, frequency 

    CovMat = []
    for i in range(cells):
        for j in range(cells):
            CovMat.append(MatPC1[:,i]* MatPC1[:,j])
        
    CovMat = np.array(CovMat)
    CovMat = np.array([CovMat[i:i+cells] for i in range(0,len(CovMat),cells)]) 
    CovMat = np.mean(CovMat, axis=2)

    eigval_r, eigvec_r= np.linalg.eigh(CovMat)
    indexes = np.flip(np.argsort(eigval_r))
    eigval_r = eigval_r[indexes]
    eigvec_r = eigvec_r[:, indexes]
    maximum = np.max(np.abs(eigvec_r))

    return eigval_r, eigvec_r

eigval, eigvec = FrequencyCovariation_E(PowerSpec, Frequencies, cells)

d = {'cells': [], 'number' :[]}      #
dataF = pd.DataFrame(d)                  #
dataF['PC1'] = np.abs(eigvec[:,0].T)     #    
dataF['PC2'] = np.abs(eigvec[:,1].T)     # 

"""
        2)  SIGNAL CLASSES EXTRACTION
       
                
"""

####       K-MEANS CLUSTERING

X = np.array((dataF['PC1'].values, dataF['PC2'].values))
y_pred = KMeans(n_clusters=2, n_init=10).fit_predict(X.T)

Xpred = y_pred[y_pred == 0]
Ypred = y_pred[y_pred != 1]

### Take the cluster indexes 
Cluster1 = np.array(np.where(y_pred == 0))              
Cluster2 = np.array(np.where(y_pred == 1)) 

### Extract the cluster indexes from the treated signal
Act1 = DataFiltBPT[Cluster1] 
Act2 = DataFiltBPT[Cluster2]

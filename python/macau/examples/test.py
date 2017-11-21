
import macau
import scipy.io
import numpy as np
import math
np.set_printoptions(threshold=np.nan)

ic50 = scipy.io.mmread("chembl-IC50-346targets.mm")
ecfp = scipy.io.mmread("chembl-IC50-compound-feat.mm")
import random

cens_num=[15]
loop_fac=1
#Matrix to store the test RMSE of the experiments
rmse_mat=np.zeros((len(cens_num),loop_fac))
rmse_mat2=np.zeros((len(cens_num),loop_fac))

random.seed(94)
np.random.seed(94)
for [idx_i,cens_i] in enumerate(cens_num): #Number of censoring thresholds
    for loop in range(loop_fac): #Repeat the experiment several times.



        drug_num=10
        assay_num=5
        latents=5
        num_cens=cens_i #censoring limit
        num_0=20
        num_test=10

        U=np.random.randn(drug_num,latents)
        V =np.random.randn(assay_num,latents)
        Y=np.dot(U,V.T)
        #print(Y)
        grid=np.indices((drug_num,assay_num))
        rows_idx=grid[0].flatten()
        col_idx=grid[1].flatten()
        import random
        idx_sel=random.sample(range(len(rows_idx)),(num_test+num_cens+num_0))
        row_sel=rows_idx[idx_sel]
        col_sel=col_idx[idx_sel]

        Y[row_sel[0:num_cens],col_sel[0:num_cens]]=np.ceil(Y[row_sel[0:num_cens],col_sel[0:num_cens]])+1.5*np.random.random(Y[row_sel[0:num_cens],col_sel[0:num_cens]].shape)
        Y[row_sel[num_cens:(num_cens+num_0)],col_sel[num_cens:(num_cens+num_0)]]=0
        val_tests=Y[row_sel[(num_cens+num_0):],col_sel[(num_cens+num_0):]].flatten()
        Y[row_sel[(num_cens+num_0):],col_sel[(num_cens+num_0):]]=0

        D=scipy.sparse.coo_matrix(Y)
        Dtest=scipy.sparse.coo_matrix((val_tests,(row_sel[(num_cens+num_0):],col_sel[(num_cens+num_0):])),shape=D.shape)
        #print("TRAINING MAT")
        #print(D)
        #print("TEST MAT")
        #print(Dtest)
        #print(val_tests)



        idx=[row_sel[0:num_cens],col_sel[0:num_cens]]
        cens=[1]*len(idx[0])
        C_mat=scipy.sparse.coo_matrix((cens,(idx[0],idx[1])),shape=D.shape)
        #print("Censoring MAT")
        #print(C_mat)

        side=np.random.randn(U.shape[0],20)
        side_inf=scipy.sparse.coo_matrix(side)

        #idx=[ic50.row,ic50.col]
        #cens=[1]*len(idx[0])
        #C_mat=scipy.sparse.coo_matrix((cens,(idx[0],idx[1])))
        C_mat0=scipy.sparse.coo_matrix(([cens[0]],([idx[0][0]],[idx[1][0]])),shape=D.shape)

        result=macau.macau(Y=D,Ytest=Dtest,side=[None,None], num_latent=latents,precision=5.0,burnin=1000,nsamples=3000,C=C_mat)
        result2=macau.macau(Y=D,Ytest=Dtest,side=[None,None],num_latent=latents,precision=5.0,burnin=1000,nsamples=3000)

        rmse_mat[idx_i,loop]=result.rmse_test
        rmse_mat2[idx_i,loop]=result2.rmse_test
        print(result.prediction)
        print(result)

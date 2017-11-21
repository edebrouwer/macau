from __future__ import division
import macau
import scipy.io
import numpy as np
import math
np.set_printoptions(threshold=np.nan)


import random

cens_lim=[-2,-1,0,1,2,3]
loop_fac=10
rmse_mat=np.zeros((len(cens_lim),loop_fac))
rmse_mat2=np.zeros((len(cens_lim),loop_fac))
cens_numbers=np.zeros((len(cens_lim),loop_fac))
random.seed(94)
np.random.seed(94)
for [idx_i,cens_i] in enumerate(cens_lim):
    for loop in range(loop_fac):



        drug_num=500
        assay_num=150
        latents=5
        cens_threshold=cens_i
        nums_0=60000
        test_per=0.2

        U=np.random.randn(drug_num,5)
        V =np.random.randn(assay_num,5)
        Y=np.dot(U,V.T)
        #print(Y)
        grid=np.indices((drug_num,assay_num))
        rows_idx=grid[0].flatten()
        col_idx=grid[1].flatten()
        import random
        idx_sel=random.sample(range(len(rows_idx)),(nums_0))
        row_sel=rows_idx[idx_sel]
        col_sel=col_idx[idx_sel]

        # Set some samples to 0
        Y[row_sel[0:nums_0],col_sel[0:nums_0]]=0

        #Training
        D=scipy.sparse.coo_matrix(Y)
        #Test
        D, Dtest = macau.make_train_test(D, test_per)
        #Censor below censoring threshold:
        a=D.data
        cens=[1]*sum(D.data<cens_threshold)
        C_mat=scipy.sparse.coo_matrix((cens,(D.row[D.data<cens_threshold],D.col[D.data<cens_threshold])),shape=D.shape)
        D.data[a<cens_threshold]=cens_threshold

        cens_numbers[idx_i,loop]=len(C_mat.data)/len(D.data)

        result=macau.macau(Y=D,Ytest=Dtest,side=[None,None], num_latent=latents,precision=5.0,burnin=2000,nsamples=2000,C=C_mat)
        result2=macau.macau(Y=D,Ytest=Dtest,side=[None,None],num_latent=latents,precision=5.0,burnin=2000,nsamples=2000)

        rmse_mat[idx_i,loop]=result.rmse_test
        rmse_mat2[idx_i,loop]=result2.rmse_test

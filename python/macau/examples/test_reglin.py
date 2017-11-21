from __future__ import division
import macau
import scipy.io
import numpy as np
import math
np.set_printoptions(threshold=np.nan)


import random

cens_lim=[-10,-5,-1,1,5,10]
loop_fac=5
rmse_mat=np.zeros((len(cens_lim),loop_fac))
rmse_mat2=np.zeros((len(cens_lim),loop_fac))
cens_numbers=np.zeros((len(cens_lim),loop_fac))
random.seed(94)
np.random.seed(94)
for [idx_i,cens_i] in enumerate(cens_lim):
    for loop in range(loop_fac):



        drug_num=500
        assay_num=1
        latents=1
        cens_threshold=cens_i
        nums_0=0
        test_per=0.2

        covariates=np.random.randn(drug_num,5)
        beta=[1,3,2,4,2]
        Y=np.dot(covariates,beta)+0.5*np.random.randn(drug_num)

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

        result=macau.macau(Y=D,Ytest=Dtest,side=[None,covariates], num_latent=latents,precision=5.0,burnin=2000,nsamples=2000,C=C_mat)
        result2=macau.macau(Y=D,Ytest=Dtest,side=[None,covariates],num_latent=latents,precision=5.0,burnin=2000,nsamples=2000)

        rmse_mat[idx_i,loop]=result.rmse_test
        rmse_mat2[idx_i,loop]=result2.rmse_test

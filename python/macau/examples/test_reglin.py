from __future__ import division
import macau
import scipy.io
import numpy as np
import math
np.set_printoptions(threshold=np.nan)

import random

mode="prop_censoring" #Mode for censoring ("global_cens" or other)
matrix_size=(500,20) #Size of simulated matrix
true_latent=5
cov_dim=5 #dimension of the side information
lat=5 #Number of latent dimensions
nums_0=0 #Number of missing values
test_per=0.8 #Proportion of test_set
cens_lim=[-10,-5,-1,1,5,10] # censoring limits ( used in "global_cens" mode)
cens_prop=[0.1,0.3,0.5,0.8] #censoring proportion (used in "other" mode)
loop_fac=1 #Number of loops for each configuration
sd_vec=[1,3,5]
#sd_vec=[1]

#Choose censoring mode.
if mode=="global_cens":
    cens_vec=cens_lim
    rmse_mat=np.zeros((len(cens_vec),loop_fac))
    rmse_mat2=np.zeros((len(cens_vec),loop_fac))
    cens_numbers=np.zeros((len(cens_vec),loop_fac))
    sd_vec=[1]
else:
    cens_vec=cens_prop
    rmse_mat=np.zeros((len(cens_vec),len(sd_vec),loop_fac))
    rmse_mat2=np.zeros((len(cens_vec),len(sd_vec),loop_fac))
    cens_numbers=np.zeros((len(cens_vec),loop_fac))

random.seed(94)
np.random.seed(94)

for [idx_i,cens_i] in enumerate(cens_vec):
    for [sd_i,sd_val] in enumerate(sd_vec):
        for loop in range(loop_fac):


            drug_num=matrix_size[0]
            assay_num=matrix_size[1]
            latents=lat
            cens_threshold=cens_i


            covariates=np.random.randn(drug_num,cov_dim)
            beta=np.random.randn(cov_dim,lat)
            #Y=(np.dot(covariates,beta)+0.5*np.random.randn(drug_num,true_latent))
            Y=np.dot((np.dot(covariates,beta)+0.5*np.random.randn(drug_num,true_latent)),0.5*np.random.randn(true_latent,assay_num))


            #Training
            D=scipy.sparse.coo_matrix(Y)
            #Test
            D, Dtest = macau.make_train_test(D, test_per)

            #Censor below censoring threshold:
            a=D.data
            if (mode=="global_cens"):
                cens=[1]*sum(D.data<cens_threshold)#vector of ones. length is number of censored samples
                C_mat=scipy.sparse.coo_matrix((cens,(D.row[D.data<cens_threshold],D.col[D.data<cens_threshold])),shape=D.shape)
                D.data[a<cens_threshold]=cens_threshold
            else:
                cens=[1]*int(cens_i*len(D.data)) #cens_prop mode.
                cens_samples=random.sample(range(len(D.data)),int(cens_i*len(D.data)))
                C_mat=scipy.sparse.coo_matrix((cens,(D.row[cens_samples],D.col[cens_samples])),shape=D.shape)
                D.data[cens_samples]=D.data[cens_samples]+abs(sd_val*np.random.randn(len(cens_samples)))


            result=macau.macau(Y=D,Ytest=Dtest,side=[covariates,None], num_latent=latents,precision=5.0,burnin=2000,nsamples=2000,C=C_mat)
            result2=macau.macau(Y=D,Ytest=Dtest,side=[covariates,None],num_latent=latents,precision=5.0,burnin=2000,nsamples=2000)

            if (mode=="global_cens"):
                rmse_mat[idx_i,loop]=result.rmse_test
                rmse_mat2[idx_i,loop]=result2.rmse_test
                cens_numbers[idx_i,loop]=len(C_mat.data)/len(D.data) #Store censoring proportion
            else:
                rmse_mat[idx_i,sd_i,loop]=result.rmse_test
                rmse_mat2[idx_i,sd_i,loop]=result2.rmse_test
                cens_numbers[idx_i,loop]=cens_i

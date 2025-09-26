#!/usr/bin/env python
import numpy as np
import csv
import pandas as pd
from general_function import *
import os
    
##################PARAMETERS#############
n = 100 #number of cells
k_on_1, k_on_2 = 5, 5
d_M_1, d_M_2 = 10, 10    #degradation rate ARN
d_P_1, d_P_2 = 5, 5    #degradation rate protein
s_1, s_2 = 10, 10    #formation rate protein
pa_1, pa_2 = 10, 10    #activation parameters
pi_1, pi_2 = -10, -10    #inhibition parameters
parameters = [k_on_1, k_on_2, d_M_1, d_M_2, d_P_1, d_P_2, s_1, s_2, pa_1, pa_2, pi_1, pi_2]
CI=(0,0)     #initial condition

G = 2    #number of gene
tms_st = np.linspace(0, 100, 100) #time in stationary state
tms_tr = np.linspace(0, 1, 100) #time in transient state

t1_st = 40 #t1 in stationary state
t1_tr = 10 #t1 in transient state
div_st = 1
div_tr = 100

n_simu = 100
epsilon = 0.001 #diffusion coeficient

#####################MAIN######################################
def main(timegap,M,n,G, parameters,CI,tms,t1,n_simu,epsilon,div,state):
    Mentrop = []
    Sentrop = []
    Mentroppdmp = []
    Sentroppdmp = []
    Mentrop1, Mentrop2, Mentrop3, Mentrop4, Mentrop5 = [], [], [], [], []
    Sentrop1, Sentrop2, Sentrop3, Sentrop4, Sentrop5 = [], [], [], [], []
    Mentroppdmp1, Mentroppdmp2, Mentroppdmp3, Mentroppdmp4, Mentroppdmp5 = [], [], [], [], []
    Sentroppdmp1, Sentroppdmp2, Sentroppdmp3, Sentroppdmp4, Sentroppdmp5 = [], [], [], [], []

    for tmpt in timegap:
        moy = []
        moypdmp = []
        moy1, moy2, moy3, moy4, moy5 = [], [], [], [], []
        moypdmp1, moypdmp2, moypdmp3, moypdmp4, moypdmp5 = [], [], [], [], []

        for m in range(M):
            S = simu(n, toggle_switch(G, parameters), CI, tms)
            mu, nu, mu_n, nu_n = distribution(S, t1, tmpt)
            parameters1=[k_on_1, k_on_2, d_M_1, d_M_2,d_P_1, d_P_2,s_1,s_2,pa_1,pa_2,pi_1,0]
            parameters2=[k_on_1, k_on_2, d_M_1, d_M_2,d_P_1, d_P_2,s_1,s_2,pa_1,pa_2,pi_1,pi_2+5]
            parameters3=[k_on_1, k_on_2, d_M_1, d_M_2,d_P_1, d_P_2,s_1,s_2,pa_1,0,pi_1,pi_2]
            parameters4=[k_on_1, k_on_2, d_M_1, d_M_2,d_P_1, d_P_2,s_1,s_2,pa_1,2*pa_2,pi_1,pi_2]
            parameters5=[k_on_1, k_on_2, d_M_1, d_M_2,d_P_1, d_P_2,s_1,s_2,0.5*pa_1,0.5*pa_2,pi_1,pi_2]

            PDMP = PDMP_ref_n(mu_n, nu_n, n, n_simu, toggle_switch(G, parameters), t1, tmpt, tms, np.max([np.max(mu), np.max(nu)]) + 0.1)
            PDMP1 = PDMP_ref_n(mu_n, nu_n, n, n_simu, toggle_switch(G, parameters1), t1, tmpt, tms, np.max([np.max(mu), np.max(nu)]) + 0.1)
            PDMP2 = PDMP_ref_n(mu_n, nu_n, n, n_simu, toggle_switch(G, parameters2), t1, tmpt, tms, np.max([np.max(mu), np.max(nu)]) + 0.1)
            PDMP3 = PDMP_ref_n(mu_n, nu_n, n, n_simu, toggle_switch(G, parameters3), t1, tmpt, tms, np.max([np.max(mu), np.max(nu)]) + 0.1)
            PDMP4 = PDMP_ref_n(mu_n, nu_n, n, n_simu, toggle_switch(G, parameters4), t1, tmpt, tms, np.max([np.max(mu), np.max(nu)]) + 0.1)
            PDMP5 = PDMP_ref_n(mu_n, nu_n, n, n_simu, toggle_switch(G, parameters5), t1, tmpt, tms, np.max([np.max(mu), np.max(nu)]) + 0.1)
            Diff = Diff_ref(mu_n, nu_n, epsilon, S[0].t[t1], S[0].t[tmpt])


            a, b = np.ones((n,)) / n, np.ones((n,)) / n
            PDMP_sch = sinkhorn(a, b, PDMP, epsilon)[0]
            PDMP1_sch = sinkhorn(a, b, PDMP1, epsilon)[0]
            PDMP2_sch = sinkhorn(a, b, PDMP2, epsilon)[0]
            PDMP3_sch = sinkhorn(a, b, PDMP3, epsilon)[0]
            PDMP4_sch = sinkhorn(a, b, PDMP4, epsilon)[0]
            PDMP5_sch = sinkhorn(a, b, PDMP5, epsilon)[0]

            Diff_sch = sinkhorn(a, b, Diff, epsilon)[0]


            ent_diff = entropy(Diff_sch, PDMP_sch)
            ent_diff1 = entropy(Diff_sch, PDMP1_sch)
            ent_diff2 = entropy(Diff_sch, PDMP2_sch)
            ent_diff3 = entropy(Diff_sch, PDMP3_sch)
            ent_diff4 = entropy(Diff_sch, PDMP4_sch)
            ent_diff5 = entropy(Diff_sch, PDMP5_sch)

            ent_pdmp = entropy(PDMP, PDMP_sch)
            ent_pdmp1 = entropy(PDMP1, PDMP1_sch)
            ent_pdmp2 = entropy(PDMP2, PDMP2_sch)
            ent_pdmp3 = entropy(PDMP3, PDMP3_sch)
            ent_pdmp4 = entropy(PDMP4, PDMP4_sch)
            ent_pdmp5 = entropy(PDMP5, PDMP5_sch)

            moy.append(ent_diff)
            moy1.append(ent_diff1)
            moy2.append(ent_diff2)
            moy3.append(ent_diff3)
            moy4.append(ent_diff4)

            moy5.append(ent_diff5)
            moypdmp.append(ent_pdmp)
            moypdmp1.append(ent_pdmp1)
            moypdmp2.append(ent_pdmp2)
            moypdmp3.append(ent_pdmp3)
            moypdmp4.append(ent_pdmp4)
            moypdmp5.append(ent_pdmp5)
        Mentrop.append(np.mean(moy))
        Sentrop.append(np.std(moy))
        Mentroppdmp.append(np.mean(moypdmp))
        Sentroppdmp.append(np.std(moypdmp))
        Mentrop1.append(np.mean(moy1))
        Sentrop1.append(np.std(moy1))
        Mentrop2.append(np.mean(moy2))
        Sentrop2.append(np.std(moy2))
        Mentrop3.append(np.mean(moy3))
        Sentrop3.append(np.std(moy3))
        Mentrop4.append(np.mean(moy4))
        Sentrop4.append(np.std(moy4))
        Mentrop5.append(np.mean(moy5))
        Sentrop5.append(np.std(moy5))

        Mentroppdmp1.append(np.mean(moypdmp1))
        Sentroppdmp1.append(np.std(moypdmp1))
        Mentroppdmp2.append(np.mean(moypdmp2))
        Sentroppdmp2.append(np.std(moypdmp2))
        Mentroppdmp3.append(np.mean(moypdmp3))
        Sentroppdmp3.append(np.std(moypdmp3))
        Mentroppdmp4.append(np.mean(moypdmp4))
        Sentroppdmp4.append(np.std(moypdmp4))
        Mentroppdmp5.append(np.mean(moypdmp5))
        Sentroppdmp5.append(np.std(moypdmp5))

    with open(f'Results_files/Entropybytimegap_PDMPs_{state}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ["timegap", 
           "MoyEntrop", "stdEntrop", "MoyEntropPDMP", "stdEntropPDMP",
           "MoyEntrop1", "stdEntrop1", "MoyEntropPDMP1", "stdEntropPDMP1",
           "MoyEntrop2", "stdEntrop2", "MoyEntropPDMP2", "stdEntropPDMP2",
           "MoyEntrop3", "stdEntrop3", "MoyEntropPDMP3", "stdEntropPDMP3",
           "MoyEntrop4", "stdEntrop4", "MoyEntropPDMP4", "stdEntropPDMP4",
           "MoyEntrop5", "stdEntrop5", "MoyEntropPDMP5", "stdEntropPDMP5"]

        writer.writerow(headers)
        for i in range(len(timegap)):
            data = [timegap[i]/div,
            Mentrop[i], Sentrop[i], Mentroppdmp[i], Sentroppdmp[i],
            Mentrop1[i], Sentrop1[i], Mentroppdmp1[i], Sentroppdmp1[i],
            Mentrop2[i], Sentrop2[i], Mentroppdmp2[i], Sentroppdmp2[i],
            Mentrop3[i], Sentrop3[i], Mentroppdmp3[i], Sentroppdmp3[i],
            Mentrop4[i], Sentrop4[i], Mentroppdmp4[i], Sentroppdmp4[i],
            Mentrop5[i], Sentrop5[i], Mentroppdmp5[i], Sentroppdmp5[i]]
            writer.writerow(data)

M=100
timegap_st = np.arange(t1_st+5,65,5)# np.arange(t1_st+5,60,5)
timegap_tr = np.arange(t1_tr+20,70,10)# np.arange(t1_tr+10,100,10)
#main(timegap_st,M,n,G,parameters,CI,tms_st,t1_st,n_simu,epsilon,div_st,"st")  
main(timegap_tr,M,n,G,parameters,CI,tms_tr,t1_tr,n_simu,epsilon,div_tr,"tr")                    

#

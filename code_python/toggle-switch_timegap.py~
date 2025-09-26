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
tms_tr = np.linspace(0, 1.5, 150) #time in transient state

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
    minentrop = []
    maxentrop = []
    minentroppdmp = []
    maxentroppdmp = []
    for tmpt in timegap:
        moy = []
        moypdmp = []
        for m in range(M):
            S = simu(n, toggle_switch(G, parameters), CI, tms)
            mu, nu, mu_n, nu_n = distribution(S, t1, tmpt)

            PDMP = PDMP_ref_n(mu_n, nu_n, n, n_simu, toggle_switch(G, parameters), t1, tmpt, tms, np.max([np.max(mu), np.max(nu)]) + 0.1)
            Diff = Diff_ref(mu_n, nu_n, epsilon, S[0].t[t1], S[0].t[tmpt])

            a, b = np.ones((n,)) / n, np.ones((n,)) / n #uniform vector
            PDMP_sch = sinkhorn(a, b, PDMP, epsilon)[0] 
            Diff_sch = sinkhorn(a, b, Diff, epsilon)[0]

            ent_diff = entropy(Diff_sch, PDMP_sch)
            ent_pdmp = entropy(PDMP, PDMP_sch)

            moy.append(ent_diff)
            moypdmp.append(ent_pdmp)
        Mentrop.append(np.mean(moy))
        Sentrop.append(np.std(moy))
        Mentroppdmp.append(np.mean(moypdmp))
        Sentroppdmp.append(np.std(moypdmp))
        minentrop.append(np.min(moy))
        maxentrop.append(np.max(moy))
        minentroppdmp.append(np.min(moypdmp))
        maxentroppdmp.append(np.max(moypdmp))

    with open(f'Results_files/Entropybytimegap_{state}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timegap", "MoyEntrop", "stdEntrop","minEntrop","maxEntrop", "MoyEntropPDMP", "stdEntropPDMP","minEntropPDMP","maxEntropPDMP"])
        for i in range(len(timegap)):
            data = [timegap[i]/div, Mentrop[i], Sentrop[i],minentrop[i],maxentrop[i], Mentroppdmp[i], Sentroppdmp[i],minentroppdmp[i],maxentroppdmp[i]]
            writer.writerow(data)

M=100
timegap_st = np.arange(t1_st+5,65,5)#np.arange(t1_st+5,60,5)
timegap_tr = np.arange(t1_tr+20,70,5)#np.arange(t1_tr+10,100,10)
main(timegap_st,M,n,G,parameters,CI,tms_st,t1_st,n_simu,epsilon,div_st,"st")  
#main(timegap_tr,M,n,G,parameters,CI,tms_tr,t1_tr,n_simu,epsilon,div_tr,"tr")                    


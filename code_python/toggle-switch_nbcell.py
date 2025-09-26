#!/usr/bin/env python
import numpy as np
import csv
import pandas as pd
from general_function import *
import os
    
##################PARAMETERS#############
k_on_1, k_on_2 = 5, 5
d_M_1, d_M_2 = 10, 10    #degradation rate ARN
d_P_1, d_P_2 = 5, 5    #degradation rate protein
s_1, s_2 = 10, 10    #formation rate protein
pa_1, pa_2 = 10, 10    #activation parameters
pi_1, pi_2 = -10, -10    #inhibition parameters
parameters = [k_on_1, k_on_2, d_M_1, d_M_2, d_P_1, d_P_2, s_1, s_2, pa_1, pa_2, pi_1, pi_2]
CI=(0,0)     #initial condition

G = 2    #number of gene
tms = np.linspace(0, 1, 100)
t1 = 10
t2 = 50
n_simu = 100
epsilon = 0.001


#####################MAIN######################################
def main(N_CELL,M,G,parameters,CI,tms,t1,t2,n_simu,epsilon):
    Mentrop = []
    Sentrop = []
    Mentroppdmp = []
    Sentroppdmp = []
    minentrop = []
    maxentrop = []
    minentroppdmp = []
    maxentroppdmp = []

    for N in N_CELL:
        moy = []
        moypdmp = []


        for m in range(40):
            S = simu(N, toggle_switch(G, parameters), [0, 0], tms)
            mu, nu, mu_n, nu_n = distribution(S, t1, t2)

            PDMP = PDMP_ref_n(mu_n, nu_n, N, n_simu, toggle_switch(G, parameters), t1, t2, tms, np.max([np.max(mu), np.max(nu)]) + 0.1)
            Diff = Diff_ref(mu_n, nu_n, epsilon, S[0].t[t1], S[0].t[t2])
            a, b = np.ones((N,)) / N, np.ones((N,)) / N
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
    with open('../visualisation/Results_files/Entropybcellumber.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["nbcell", "MoyEntrop", "stdEntrop","minEntrop","maxEntrop", "MoyEntropPDMP", "stdEntropPDMP","minEntropPDMP","maxEntropPDMP"])
        for i in range(len(N_CELL)):
            data = [N_CELL[i], Mentrop[i], Sentrop[i],minentrop[i],maxentrop[i], Mentroppdmp[i], Sentroppdmp[i],minentroppdmp[i],maxentroppdmp[i]]
            writer.writerow(data)


M=100
N_CELL = [25, 64, 100, 144, 256, 400]

main(N_CELL,M,G,parameters,CI,tms,t1,t2,n_simu,epsilon)            
                    




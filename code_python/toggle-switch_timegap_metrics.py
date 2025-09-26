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
    Mfrob = []
    Sfrob = []
    Mtv = []
    Stv = []
    Mjs = []
    Sjs = []
    Mfrob_pdmp = []
    Sfrob_pdmp = []
    Mtv_pdmp = []
    Stv_pdmp = []
    Mjs_pdmp = []
    Sjs_pdmp = []

    for tmpt in timegap:
        moy = []
        moypdmp = []
        timegap_str = str(tmpt)
        print(tmpt)
        frob_vals = []
        tv_vals = []
        kl_vals = []
        js_vals = []
        frob_pdmp_vals = []
        tv_pdmp_vals = []
        kl_pdmp_vals = []
        js_pdmp_vals = []
        for m in range(M):
            S = simu(n, toggle_switch(G, parameters), CI, tms)
            mu, nu, mu_n, nu_n = distribution(S, t1, tmpt)
            PDMP = PDMP_ref_n(mu_n, nu_n, n, n_simu, toggle_switch(G, parameters), t1, tmpt, tms, np.max([np.max(mu), np.max(nu)]) + 0.1)
            Diff = Diff_ref(mu_n, nu_n, epsilon, S[0].t[t1], S[0].t[tmpt])

            a, b = np.ones((n,)) / n, np.ones((n,)) / n
            PDMP_sch = sinkhorn(a, b, PDMP, epsilon)[0]
            Diff_sch = sinkhorn(a, b, Diff, epsilon)[0]

            ent_diff = entropy(Diff_sch, PDMP_sch)
            ent_pdmp = entropy(PDMP, PDMP_sch)
            fro_diff = frobenius_distance(Diff_sch, PDMP_sch)
            tv_diff = total_variation(Diff_sch, PDMP_sch)
            js_diff = js_divergence(Diff_sch, PDMP_sch)
            fro_pdmp = frobenius_distance(PDMP, PDMP_sch)
            tv_pdmp = total_variation(PDMP, PDMP_sch)
            js_pdmp = js_divergence(PDMP, PDMP_sch)

            frob_pdmp_vals.append(fro_pdmp)
            tv_pdmp_vals.append(tv_pdmp)
            js_pdmp_vals.append(js_pdmp)

            moy.append(ent_diff)
            moypdmp.append(ent_pdmp)
            frob_vals.append(fro_diff)
            tv_vals.append(tv_diff)
            js_vals.append(js_diff)
        Mentrop.append(np.mean(moy))
        Sentrop.append(np.std(moy))
        Mentroppdmp.append(np.mean(moypdmp))
        Sentroppdmp.append(np.std(moypdmp))
        Mfrob.append(np.mean(frob_vals))
        Sfrob.append(np.std(frob_vals))
        Mtv.append(np.mean(tv_vals))
        Stv.append(np.std(tv_vals))
        Mjs.append(np.mean(js_vals))
        Sjs.append(np.std(js_vals))
        Mfrob_pdmp.append(np.mean(frob_pdmp_vals))
        Sfrob_pdmp.append(np.std(frob_pdmp_vals))
        Mtv_pdmp.append(np.mean(tv_pdmp_vals))
        Stv_pdmp.append(np.std(tv_pdmp_vals))
        Mjs_pdmp.append(np.mean(js_pdmp_vals))
        Sjs_pdmp.append(np.std(js_pdmp_vals))

    with open(f'../visualisation/Results_files/Entropybytimegap_metrics_{state}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        headers = [
           "timegap", "MoyEntrop", "stdEntrop", "MoyEntropPDMP", "stdEntropPDMP",
           "MoyFrobenius", "stdFrobenius",
           "MoyTV", "stdTV",
           "MoyJS", "stdJS","MoyFrobenius_PDMP", "stdFrobenius_PDMP",
            "MoyTV_PDMP", "stdTV_PDMP",
            "MoyJS_PDMP", "stdJS_PDMP"
        ]
        writer.writerow(headers)
        for i in range(len(timegap)):
            data = [
                timegap[i]/div, Mentrop[i], Sentrop[i], Mentroppdmp[i], Sentroppdmp[i],
                Mfrob[i], Sfrob[i],
                Mtv[i], Stv[i],
                Mjs[i], Sjs[i],
                Mfrob_pdmp[i], Sfrob_pdmp[i],
                Mtv_pdmp[i], Stv_pdmp[i],
                Mjs_pdmp[i], Sjs_pdmp[i]
            ]
            writer.writerow(data)
M=100 
timegap_st = np.arange(t1_st+5,60,20)
timegap_tr =  np.arange(t1_tr+20,70,5)
#main(timegap_st,M,n,G, parameters,CI,tms_st,t1_st,n_simu,epsilon,div_st,"st")  
main(timegap_tr,M,n,G, parameters,CI,tms_tr,t1_tr,n_simu,epsilon,div_tr,"tr")


#!/usr/bin/env python
import numpy as np
import csv
import pandas as pd
from general_function import *
import os

##################PARAMETERS#############
n=100 #number of cells
k_on_1, k_on_2, k_on_3 = 5,5,5
d_M= 10    #degradation rate ARN
d_P = 5   #degradation rate protein

pa_11,pa_22,pa_33= 10,10,10   #activation parameters
pa_12,pa_23,pa_31= -10, -10,-10    #inhibition parameters
parameters=[k_on_1, k_on_2 ,k_on_3, d_M,d_P, pa_11,pa_22,pa_33,pa_12,pa_23,pa_31]

G = 3   #nombre de genes 
div=100
tms =np.linspace(0,1.5,150)
t1=10
CI=(0,0,0)     #initial condition

n_simu=100       
epsilon=0.001

def main (timegap,M,n,G, parameters,CI,tms,t1,n_simu,epsilon,div):
    Mentroppdmp=[]
    Sentroppdmp=[]
    Mentrop=[]
    Sentrop =[]
    MINpdmp=[]
    MAXpdmp=[]

    MIN=[]
    MAX=[]
    for tmpt in (timegap):
        moy=[]
        moypdmp=[]
        for m in range(M):
            S=simu_3G(n,Network3genes(G,parameters),CI,tms)
            mu,nu,mu_n,nu_n=distribution_3G(S,t1,tmpt)
            PDMP=PDMP_ref_n_3G(mu_n,nu_n,n,n_simu,Network3genes(G,parameters),t1,tmpt,tms,np.max([np.max(mu),np.max(nu)])+0.1)
            Diff=Diff_ref(mu_n,nu_n,epsilon,S[0].t[t1],S[0].t[tmpt])

            a, b = np.ones((n,)) / n, np.ones((n,)) / n  

            element_max = np.vectorize(max)
            PDMP_sch=sinkhorn(a,b,PDMP,epsilon)[0]
            Diff_sch=sinkhorn(a,b,Diff,epsilon)[0]

            moy.append(entropy(Diff_sch,PDMP_sch))
            moypdmp.append(entropy(PDMP,PDMP_sch))
        timegap_str=str(tmpt/div)
        Mentrop.append(np.mean(moy))
        Sentrop.append(np.std(moy))
        Mentroppdmp.append(np.mean(moypdmp))
        Sentroppdmp.append(np.std(moypdmp))
        MIN.append(np.min(moy))
        MAX.append(np.max(moy))
        MINpdmp.append(np.min(moypdmp))
        MAXpdmp.append(np.max(moypdmp))
    with open('../visualisation/Results_files/Entropybytimegap_repressilator.csv', 'w') as f:
        writer = csv.writer(f)
        headers = ["timegap", "MoyEntrop","stdEntrop","minEntrop","maxEntrop", "MoyEntropPDMP","stdEntropPDMP","minEntropPDMP","maxEntropPDMP"]
        writer.writerow(headers)
        for i in range (len(timegap)):
            data=[timegap[i]/div,Mentrop[i],Sentrop[i],MIN[i],MAX[i],Mentroppdmp[i],Sentroppdmp[i],MINpdmp[i],MAXpdmp[i]]
            writer.writerow(data)

M=100
timegap = np.arange(30,70,5)
main(timegap,M,n,G,parameters,CI,tms,t1,n_simu,epsilon,div)                    

       

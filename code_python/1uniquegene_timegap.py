#!/usr/bin/env python
import numpy as np
from harissa import NetworkModel #https://github.com/ulysseherbach/harissa/tree/main
from harissa.models import BurstyBase
import ot #https://pythonot.github.io/index.html
import csv
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
from general_function import entropy
from sinkhorn_bridge import relative_entropy, sinkhorn_bridge

def MODEL():
    # Create a model
    model = BurstyBase()
    model.burst_size = 0.5
    model.degradation_rate = 0.8
    return model

def x1x2(model,n_cells,kon_new):
    x1 = np.zeros(n_cells)
    x2 = np.zeros(n_cells)
    kon_old = 1 # "Old" burst frequency
    kon_new =2.5 # "New" burst frequency
    long_time = 10 / model.degradation_rate # Sufficient time to reach steady state
    delta_t = 0.1 / model.degradation_rate # Time delay between the two snapshots

    # Snapshot 1: steady-state cells with "old" burst frequency
    model.burst_frequency = kon_old
    for k in range(n_cells):
        x1[k] = model.simulate(long_time).x

    # Snapshot 2: simulate previous cells with "new" burst frequency
    model.burst_frequency = kon_new
    for k in range(n_cells):
        x2[k] = model.simulate(delta_t, init_state=x1[k]).x
    return x1,x2


def K_ref(x1,x2,delta_t,n_cells,kon_new,model):
    # Assume we know the correct model
    
    model.burst_frequency = kon_new

    # Conditional distribution of x2 with respect to x1
    prob_cond_x1 = model.distribution(x2, x1, delta_t, discrete=True)

    # Probability measures related to x1 and x2
    prob_x1 = np.ones((n_cells, 1)) / n_cells
    prob_x2 = np.ones((1, n_cells)) / n_cells

    # Reference coupling for Schrödinger bridge (use broadcasting)
    p_ref = prob_cond_x1 * prob_x1
    return p_ref
def simu(n,model,VI,tms):
    S=[]
    for i in range(n):  # pour chaque cellule
        sim = model.simulate(tms, init_state=VI)   # simulation avec harissa
        S.append(sim)
    return S
def distribution_gauss(S, t1, t2):
    mu=[]
    nu=[] 
    for sim in S:
        cell_t1=[sim.x[int(t1)]] #vecteur P1 et P2 au temps 1
        cell_t2=[sim.x[int(t2)]]
        mu.append(cell_t1)
        nu.append(cell_t2)
    mu=np.array(mu)
    nu=np.array(nu)
    values = np.vstack([nu[:, g] for g in range(0, 1)])
    return mu, nu, values
def PDMP_ref(mu,nu,n,n_simu,model,t1,t2,tms):
    P=np.zeros((int(n), int(n)))
    for i in range(len(mu)):
        sims=simu(n_simu,model,mu[i],tms)#,[mu_n[i]*maxi])
        nu_tilde=distribution_gauss(sims,0,t2-t1)[2]

        kernel = st.gaussian_kde(nu_tilde) # kernel of estimate density with gaussian distrib (lissage des donnée pour ne pas avoir de 0)
        for k in range(len(nu)):

            P[i,k]=kernel(nu[k])[0] 
    return P/np.sum(P) 
def Diff_ref(mu_n,nu_n,epsilon,t1,t2,n_cells):
    D=np.exp(-ot.dist(mu_n.reshape((n_cells, 1)),nu_n.reshape((n_cells, 1)))/(epsilon*(t2-t1))) #distance euclidienne en noyau de transition
    return D/np.sum(D)

def plot_process(diff_sch,pdmp_sch,K_sch):
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4 , figsize=(10,10))

    im = ax1.imshow(diff_sch.T, interpolation='nearest',cmap="Blues", origin='lower', vmin=0, vmax=np.max(K_sch))
    ax1.set_title("$Diff^{sch}$")
    ax1.set_xlabel("index cells at $t_1$") 
    ax1.set_ylabel("index cells at $t_2$")        
    im = ax2.imshow(PDMP_sch.T, interpolation='nearest',cmap="Reds", origin='lower', vmin=0, vmax=np.max(K_sch))
    ax2.set_title("$PDMP^{sch}$")
    ax2.set_xlabel("index cells at $t_1$") 
    im = ax3.imshow(K_sch.T, interpolation='nearest',cmap="Greens", origin='lower', vmin=0, vmax=np.max(K_sch))
    ax3.set_title("$K^{sch}$")
    ax3.set_xlabel("index cells at $t_1$") 
    im = ax4.imshow(np.identity(n_cells).T/n_cells, interpolation='nearest',cmap="Greys", origin='lower')
    plt.title("ground truth")
    ax4.set_xlabel("index cells at $t_1$") 
    
def graph(df,t1):
    plt.plot(df.timegap, df.MoyEntropDK,c='lightskyblue', label='HDK',marker='o')
    plt.errorbar(
        df.timegap, df.MoyEntropDK, 
        yerr=df.stdEntropDK,
        fmt='none', ecolor='lightskyblue',elinewidth=1.5, capsize=3, alpha=0.8)
    plt.plot(df.timegap , df.minEntropDK,'--',  c='lightskyblue')
    plt.plot(df.timegap , df.maxEntropDK,'--',  c='lightskyblue')
    
    plt.plot(df.timegap, df.MoyEntropKK,c='darkseagreen', label='HKK',marker='o')
    plt.errorbar(
        df.timegap, df.MoyEntropKK, 
        yerr=df.stdEntropKK,
        fmt='none', ecolor='darkseagreen',elinewidth=1.5, capsize=3, alpha=0.8)
    plt.plot(df.timegap , df.minEntropKK,'--',  c='darkseagreen')
    plt.plot(df.timegap , df.maxEntropKK,'--',  c='darkseagreen')
    
    plt.plot(df.timegap, df.MoyEntropPK,c='lightcoral', label='HPK',marker='o')
    plt.errorbar(
        df.timegap, df.MoyEntropPK, 
        yerr=df.stdEntropPK,
        fmt='none', ecolor='lightcoral',elinewidth=1.5, capsize=3, alpha=0.8)
    plt.plot(df.timegap , df.minEntropPK,'--',  c='lightcoral')
    plt.plot(df.timegap , df.maxEntropPK,'--',  c='lightcoral')
    plt.axvline(t1,label="default value")
    plt.xlabel("$t_2$")
    plt.ylabel("Entropy mean ± std")
    plt.grid()
    plt.show()

def main(n_cells,M,n_simu,t1,tms,timegap):
    MentropDK = []
    SentropDK = []
    MentropKK = []
    SentropKK = []
    MentropPK = []
    SentropPK = []
    minentropDK = []
    maxentropDK = []
    minentropKK = []
    maxentropKK = []
    minentropPK = []
    maxentropPK = []
    for tmpt in timegap:
        moyDK = []
        moyKK = []
        moyPK = []
        for m in range(M):
            kon_new = 2.5
            model=MODEL()
            x1,x2=x1x2(model,n_cells,kon_new)
            K=K_ref(x1,x2,tmpt-t1,n_cells,kon_new,model)
            diff= Diff_ref(x1,x2,0.001,t1,tmpt,n_cells)
            pdmp= PDMP_ref(x1,x2,n_cells,n_simu,model,t1,tmpt,tms)
            lambd = 1e-3
            epsilon=0.001   
            
            a = np.ones((n_cells, 1)) / n_cells
            b = np.ones((1, n_cells)) / n_cells
            PDMP_sch =sinkhorn_bridge(pdmp, a,b, tol=1e-5, verb=True)[0]
            K_sch = sinkhorn_bridge(K, a,b, tol=1e-5, verb=True)[0]
            diff_sch =sinkhorn_bridge(diff, a,b, tol=1e-5, verb=True)[0]
            ent_diffK = entropy(diff_sch, K_sch)
            ent_pdmpK = entropy(PDMP_sch,K_sch)
            ent_kk = entropy(K,K_sch)
            moyDK.append(ent_diffK)
            moyKK.append(ent_kk)
            moyPK.append(ent_pdmpK)

        MentropDK.append(np.mean(moyDK))
        SentropDK.append(np.std(moyDK))
        MentropKK.append(np.mean(moyKK))
        SentropKK.append(np.std(moyKK))
        MentropPK.append(np.mean(moyPK))
        SentropPK.append(np.std(moyPK))
        minentropDK.append(np.min(moyDK))
        maxentropDK.append(np.max(moyDK))
        minentropPK.append(np.min(moyPK))
        maxentropPK.append(np.max(moyPK))        
        minentropKK.append(np.min(moyKK))
        maxentropKK.append(np.max(moyKK))

    with open('../visualisation/Results_files/Entropybytimegap_1GENE.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ["timegap", "MoyEntropDK", "stdEntropDK","minEntropDK","maxEntropDK",
                   "MoyEntropKK", "stdEntropKK","minEntropKK","maxEntropKK",
                   "MoyEntropPK", "stdEntropPK","minEntropPK","maxEntropPK"]
        writer.writerow(headers)
        for i in range(len(timegap)):
            data = [timegap[i]/100, MentropDK[i], SentropDK[i],minentropDK[i],maxentropDK[i],
                    MentropKK[i], SentropKK[i],minentropKK[i],maxentropKK[i],
                    MentropPK[i], SentropPK[i],minentropPK[i],maxentropPK[i]] 

            writer.writerow(data)


M=50     
t=np.arange(20,60,10)#np.arange(150,250,50)#np.arange(150,300,50)
main(100,M,100,10,np.linspace(0,1,100),t)



    

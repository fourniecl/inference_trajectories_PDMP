#!/usr/bin/env python
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys; sys.path += ['../']
import ot #https://pythonot.github.io/index.html
from harissa import NetworkModel 
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
import pandas as pd

####################MODEL WITH 2 GENES#########################
def toggle_switch(G,parameters):
        model = NetworkModel(G) #init model
        model.basal[1] = parameters[0] # k_on_1
        model.basal[2] = parameters[1] #k_on_2
        ### theta ###
        model.inter[1,1] = parameters[8] #theta_11
        model.inter[2,2] = parameters[9] #theta_22
        model.inter[1,2] = parameters[10] #theta_12
        model.inter[2,1] = parameters[11] #theta_21
        model.d[0] = parameters[2] #degradation rate for mRNA
        model.d[1] = parameters[4] #degradation rate for protein

        return model
def Network3genes(G,parameters):
        model = NetworkModel(G) #init model
        model.basal[1] = parameters[0] # k_on_1
        model.basal[2] = parameters[1] #k_on_2
        model.basal[3] = parameters[2] #k_on_3
        ### theta ###
        model.inter[1,1] = parameters[5] #theta_11
        model.inter[2,2] = parameters[6] #theta_22
        model.inter[3,3] = parameters[7] #theta_33
        model.inter[1,2] = parameters[8] #theta_12
        model.inter[2,3] = parameters[9] #theta_23
        model.inter[3,1] = parameters[10] #theta_31
        model.d[0] = parameters[3] #d_M_1 et d_M_1
        model.d[1] = parameters[4] #d_P_1 et d_P_1

        return model

################SIMULATIONS#################################
def simu(n,model,VI,tms):
    S=[]
    for i in range(n):  # for each cell
        sim = model.simulate(tms, use_numba=True,P0=[0,VI[0],VI[1]])	   #simulation
        S.append(sim)
    return S
def simu_RNA(n,model,VI,tms):
    S=[]
    for i in range(n):  # pour chaque cellule
        sim = model.simulate(tms, use_numba=True,M0=[0,VI[0],VI[1]])	   #simulation
        S.append(sim)
    return S

def simu_3G(n,model,VI,tms):
    S=[]
    for i in range(n):  # pour chaque cellule
        sim = model.simulate(tms, use_numba=True,P0=[0,VI[0],VI[1],VI[2]])	 #simulation
        S.append(sim)
    return S

def distribution(S, t1, t2):
    mu=[]
    nu=[] 
    for sim in S:
        cell_t1=[sim.p[:,0][t1],sim.p[:,1][t1]] #vector P1 and P2 at t1
        cell_t2=[sim.p[:,0][t2],sim.p[:,1][t2]] #vector P1 and P2 at t2
        mu.append(cell_t1)
        nu.append(cell_t2)
    mu=np.array(mu)
    nu=np.array(nu)
    return mu, nu,mu/(np.max([np.max(mu),np.max(nu)])+0.1), nu/(np.max([np.max(mu),np.max(nu)])+0.1)
def distribution_RNA(S, t1, t2):
    mu=[]
    nu=[] 
    for sim in S:
        cell_t1=[sim.m[:,0][t1],sim.m[:,1][t1]] #vector M1 and M2 at t1
        cell_t2=[sim.m[:,0][t2],sim.m[:,1][t2]] #vector M1 and M2 at t2
        mu.append(cell_t1)
        nu.append(cell_t2)
    mu=np.array(mu)
    nu=np.array(nu)
    return mu, nu,mu/(np.max([np.max(mu),np.max(nu)])+0.1), nu/(np.max([np.max(mu),np.max(nu)])+0.1)
def distribution_3G(S, t1, t2):
    mu=[]
    nu=[] 
    for sim in S:
        cell_t1=[sim.p[:,0][t1],sim.p[:,1][t1],sim.p[:,2][t1]] #vecteur P1 et P2 au temps 1
        cell_t2=[sim.p[:,0][t2],sim.p[:,1][t2],sim.p[:,2][t2]]
        mu.append(cell_t1)
        nu.append(cell_t2)
    mu=np.array(mu)
    nu=np.array(nu)
    return mu, nu,mu/(np.max([np.max(mu),np.max(nu)])+0.1), nu/(np.max([np.max(mu),np.max(nu)])+0.1)

####################PROCESSES#################################
def PDMP_ref_n(mu_n,nu_n,n,n_simu,model,t1,t2,tms,maxi):
    P=np.zeros((int(n), int(n)))
    for i in range(len(mu_n)):
        sims=simu(n_simu,model,[mu_n[i,0]*maxi,mu_n[i,1]*maxi],tms)
        nu_=distribution(sims,0,t2-t1)[3]
        nu_tilde=np.vstack([nu_[:, g] for g in range(0, 2)])
        kernel = st.gaussian_kde(nu_tilde) # kernel of estimate density with gaussian distrib 
        for k in range(len(nu_n)):
            P[i,k]=kernel(nu_n[k])[0] 
    if np.sum(P)!=0:
        return P/np.sum(P)
    else:
        return P
    
def PDMP_ref_n_RNA(mu_n,nu_n,n,n_simu,model,t1,t2,tms,maxi):
    P=np.zeros((int(n), int(n)))
    for i in range(len(mu_n)):
        sims=simu_RNA(n_simu,model,[mu_n[i,0]*maxi,mu_n[i,1]*maxi],tms)
        nu_=distribution_RNA(sims,0,t2-t1)[3]
        nu_tilde=np.vstack([nu_[:, g] for g in range(0, 2)])
        if np.linalg.matrix_rank(nu_tilde) < nu_tilde.shape[0]: #add noize in case nu_tilde is degenarated
            nu_tilde += 1e-8 * np.random.randn(*nu_tilde.shape)
        kernel = st.gaussian_kde(nu_tilde) # kernel of estimate density with gaussian distrib 
        for k in range(len(nu_n)):
            P[i,k]=kernel(nu_n[k])[0] 
    if np.sum(P)!=0:
        return P/np.sum(P)
    else:
        return P
def PDMP_ref_n_3G(mu_n,nu_n,n,n_simu,model,t1,t2,tms,maxi):
    P=np.zeros((int(n), int(n)))
    for i in range(len(mu_n)):
        sims=simu_3G(n_simu,model,[mu_n[i,0]*maxi,mu_n[i,1]*maxi,mu_n[i,2]*maxi],tms)
        nu_=distribution_3G(sims,0,t2-t1)[3]
        nu_tilde=np.vstack([nu_[:, g] for g in range(0, 3)])
        if np.linalg.matrix_rank(nu_tilde) < nu_tilde.shape[0]: #add noize in case nu_tilde is degenarated
            nu_tilde += 1e-8 * np.random.randn(*nu_tilde.shape)
        kernel = st.gaussian_kde(nu_tilde) # kernel of estimate density with gaussian distrib 
        for k in range(len(nu_n)):
            P[i,k]=kernel(nu_n[k])[0]     
    if np.sum(P)!=0:
        return P/np.sum(P)
    else:
        return P

   

def Diff_ref(mu_n,nu_n,epsilon,t1,t2):
    D=np.exp(-ot.dist(mu_n,nu_n)/(epsilon*(t2-t1))) #euclidean distance into transition kernel
    if np.sum(D)!=0:
        return D/np.sum(D)
    else:
        return D
###########################OT###############################
eps = 1e-1
element_max = np.vectorize(max)
def sinkhorn( a, b, K, epsilon, precision=eps): #https://lucyliu-ucsb.github.io/posts/Sinkhorn-algorithm/
    C=K
    a = a.reshape((C.shape[0], 1))
    b = b.reshape((C.shape[1], 1))
    u = np.ones((C.shape[0], 1))
    v = np.ones((C.shape[1], 1))
    P = np.diag(u.flatten()) @ K @ np.diag(v.flatten())
    p_norm = np.trace(P.T @ P)
    print(np.sum(P),p_norm)
    while True:
        u = a/element_max((K @ v), 1e-300) # avoid divided by zero
        v = b/element_max((K.T @ u), 1e-300)
        P = np.diag(u.flatten()) @ K @ np.diag(v.flatten())
        if abs((np.trace(P.T @ P) - p_norm)/p_norm) < precision:
            break
        p_norm = np.trace(P.T @ P)
    return P, np.trace(C.T @ P)
def entropy (P,R):
    entrop=0.0
    for j in range (0,len(R)):
        for k in range (0,len(R[j])):
            if R[j][k]==0:  
                entrop+=0
            elif P[j][k] ==0:
                entrop+=0
            else: 
                entrop+=P[j][k] * np.log(P[j][k] / R[j][k])
    return entrop


def frobenius_distance(A, B):
    return np.linalg.norm(A - B, ord='fro')

def total_variation(A, B):
    return 0.5 * np.sum(np.abs(A - B))

def kl_divergence(P, Q):
    kl = 0.0
    for p_row, q_row in zip(P, Q):
        mask = (p_row > 0) & (q_row > 0)
        kl += np.sum(rel_entr(p_row[mask], q_row[mask]))
    return kl

def js_divergence(P, Q):
    js = 0.0
    for p_row, q_row in zip(P, Q):
        m = 0.5 * (p_row + q_row)
        mask = (p_row > 0) & (q_row > 0)
        js += 0.5 * np.sum(rel_entr(p_row[mask], m[mask])) + 0.5 * np.sum(rel_entr(q_row[mask], m[mask]))
    return js

def save_matrix(mat, name):
    with open(name, 'w', newline='') as matf:
        writer_mat = csv.writer(matf)
        writer_mat.writerows(mat)
################PLOT#################################    
def basic_plot(sim,G):
    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(2,1)
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    # Plot proteins
    for i in range(G):
        ax1.plot(sim.t, sim.p[:,i], label=f'$P_{{{i+1}}}$')
        ax1.set_xlim(sim.t[0], sim.t[-1])
        ax1.set_ylim(0, np.max([1.2*np.max(sim.p), 1]))
        ax1.tick_params(axis='x', labelbottom=False)
        ax1.legend(loc='upper left', ncol=4, borderaxespad=0, frameon=False)
    # Plot mRNA
    for i in range(G):
        ax2.plot(sim.t, sim.m[:,i], label=f'$M_{{{i+1}}}$')
        ax2.set_xlim(sim.t[0], sim.t[-1])
        ax2.set_ylim(0, 1.2*np.max(sim.m))
        ax2.legend(loc='upper left', ncol=4, borderaxespad=0, frameon=False)
        
def plot_distrib(mu,nu):
    fig,  (ax1) = plt.subplots(1,1, figsize=(6,6))
    ax1.scatter(mu[0],mu[1], c="darkgreen")   
    ax1.scatter(nu[0],nu[1], c="purple")   

    ax1.set_xlabel("Protein 1")
    ax1.set_ylabel("Protein 2")     
    ax1.set_xlim(0.0,np.max([np.max(mu),np.max(nu)]))  
    ax1.set_ylim(0.0,np.max([np.max(mu),np.max(nu)]))
    ax1.set_title('Distributions in protein space')

    
def plot_process(diff_sch,PDMP_sch,K_sch):
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2 , figsize=(10,10))

    im = ax1.imshow(diff_sch.T, interpolation='nearest',cmap="Blues", origin='lower', vmin=0, vmax=np.max(K_sch))
    ax1.set_title(f'A.', loc='left', fontweight='bold')
    ax1.set_xlabel("index cells at $t_1$") 
    ax1.set_ylabel("index cells at $t_2$")        
    #cb =fig.colorbar(im, ax=ax1,orientation= 'horizontal')#, label='ratio of cells per cases')

    im = ax2.imshow(PDMP_sch.T, interpolation='nearest',cmap="Reds", origin='lower', vmin=0, vmax=np.max(K_sch))
    ax2.set_title(f'B.', loc='left', fontweight='bold')
    ax2.set_xlabel("index cells at $t_1$") 
    #cb =fig.colorbar(im, ax=ax2,orientation=  'horizontal')

    im = ax3.imshow(K_sch.T, interpolation='nearest',cmap="Greens", origin='lower', vmin=0, vmax=np.max(K_sch))
    ax3.set_title(f'C.', loc='left', fontweight='bold')
    ax3.set_xlabel("index cells at $t_1$") 
    ax3.set_ylabel("index cells at $t_2$")        

    #cb =fig.colorbar(im, ax=ax3,orientation='horizontal')#, anchor=(0, 0.3),boundaries=[0,np.max(K_ref)])

    im = ax4.imshow(np.identity(np.shape(diff_sch)[0]).T/np.shape(diff_sch)[0], interpolation='nearest',cmap="Greys", origin='lower')#, vmin=0, vmax=np.max(K_sch))
    ax4.set_title(f'D.', loc='left', fontweight='bold')
    ax4.set_xlabel("index cells at $t_1$") 
    #fig.colorbar(im, ax=ax4)#,orientation='horizontal')#, anchor=(0, 0.3),boundaries=[0,np.max(K_ref)])

def graph1G(df,t1):
    
    plt.plot(df.timegap-t1, df.MoyEntropDK,c='blue', label='HDK',marker='o')
    plt.errorbar(
        df.timegap-t1, df.MoyEntropDK, 
        yerr=df.stdEntropDK,
        fmt='none', ecolor='blue',elinewidth=1.5, capsize=3, alpha=0.8)
    plt.plot(df.timegap-t1 , df.minEntropDK,'--',  c='lightblue')
    plt.plot(df.timegap-t1 , df.maxEntropDK,'--',  c='lightblue')
    
    plt.plot(df.timegap-t1, df.MoyEntropKK,c='limegreen', label='HKK',marker='o')
    plt.errorbar(
        df.timegap-t1, df.MoyEntropKK, 
        yerr=df.stdEntropKK,
        fmt='none', ecolor='limegreen',elinewidth=1.5, capsize=3, alpha=0.8)
    plt.plot(df.timegap-t1 , df.minEntropKK,'--',  c='lightgreen')
    plt.plot(df.timegap-t1 , df.maxEntropKK,'--',  c='lightgreen')
    
    plt.plot(df.timegap-t1, df.MoyEntropPK,c='red', label='HPK',marker='o')
    plt.errorbar(
        df.timegap-t1, df.MoyEntropPK, 
        yerr=df.stdEntropPK,
        fmt='none', ecolor='red',elinewidth=1.5, capsize=3, alpha=0.8)
    plt.plot(df.timegap-t1 , df.minEntropPK,'--',  c='pink')
    plt.plot(df.timegap-t1 , df.maxEntropPK,'--',  c='pink')
    #plt.axvline(t1,label="default value",color='grey')
    plt.xlabel("$t_{2}-t_{1}$")
    plt.ylabel("Entropy")
    #plt.grid()
    plt.show()        
def graph(df,t1):
    plt.plot(df.timegap, df.MoyEntrop,c='blue', label='HDP',marker='o')
    plt.errorbar(
        df.timegap, df.MoyEntrop, 
        yerr=df.stdEntrop,
        fmt='none', ecolor='blue',elinewidth=1.5, capsize=3, alpha=0.8)
    plt.plot(df.timegap , df.minEntrop,'--',  c='lightblue')
    plt.plot(df.timegap , df.maxEntrop,'--',  c='lightblue')

    plt.plot(df.timegap, df.MoyEntropPDMP, c='red',marker='o',label=r'$HPP$')
    plt.errorbar(df.timegap, df.MoyEntropPDMP, yerr=df.stdEntropPDMP,
        fmt='none', ecolor='red',elinewidth=1.5, capsize=3, alpha=0.8)
    plt.plot(df.timegap , df.minEntropPDMP,'--',  c='pink')
    plt.plot(df.timegap , df.maxEntropPDMP,'--',  c='pink')
    #plt.plot(t1,df.minEntropPDMP,'--',color='grey',label='min and max')
    plt.axvline(t1,label="default value",color='grey')
    #plt.legend(bbox_to_anchor=(1.0,1.0))
    plt.xlabel("$t_2$")
    plt.ylabel("Entropy")
    plt.grid()
    plt.show()
def grapht2t1(df,t1,legend):
    plt.plot(df.timegap-t1, df.MoyEntrop,c='blue', label='HDP',marker='o')
    plt.errorbar(
        df.timegap-t1, df.MoyEntrop, 
        yerr=df.stdEntrop,
        fmt='none', ecolor='blue',elinewidth=1.5, capsize=3, alpha=0.8)
    plt.plot(df.timegap-t1 , df.minEntrop,'--',  c='lightblue')
    plt.plot(df.timegap-t1 , df.maxEntrop,'--',  c='lightblue')

    plt.plot(df.timegap-t1, df.MoyEntropPDMP, c='red',marker='o',label=r'$HPP$')
    plt.errorbar(df.timegap-t1, df.MoyEntropPDMP, yerr=df.stdEntropPDMP,
        fmt='none', ecolor='red',elinewidth=1.5, capsize=3, alpha=0.8)
    plt.plot(df.timegap-t1 , df.minEntropPDMP,'--',  c='pink')
    plt.plot(df.timegap-t1 , df.maxEntropPDMP,'--',  c='pink')
    #plt.plot(t1,df.minEntropPDMP,'--',color='grey',label='min and max')
    #plt.axvline(t1,label="default value",color='grey')
    #plt.legend(bbox_to_anchor=(1.0,1.0))
    plt.xlabel(legend)
    plt.ylabel("Entropy")
    #plt.grid()
    plt.show()
    
def graph_nbcell(entropnbcell):
    plt.plot(entropnbcell.nbcell, entropnbcell.MoyEntrop,c='blue', label='HDP (mean ± std)',marker='o')
    plt.errorbar(
        entropnbcell.nbcell, entropnbcell.MoyEntrop, 
        yerr=entropnbcell.stdEntrop,
        fmt='none', ecolor='blue',elinewidth=1.5, capsize=3, alpha=0.8)
    plt.plot(entropnbcell.nbcell , entropnbcell.minEntrop,'--',  c='lightblue')
    plt.plot(entropnbcell.nbcell , entropnbcell.maxEntrop,'--',  c='lightblue')

    plt.plot(entropnbcell.nbcell, entropnbcell.MoyEntropPDMP, c='red', 
             label=r'HPP (mean ± std)',marker='o')
    plt.errorbar(
        entropnbcell.nbcell, entropnbcell.MoyEntropPDMP, 
        yerr=entropnbcell.stdEntropPDMP,
        fmt='none', ecolor='red',elinewidth=1.5, capsize=3, alpha=0.8)
    plt.plot(entropnbcell.nbcell , entropnbcell.minEntropPDMP,'--',  c='pink')
    plt.plot(entropnbcell.nbcell , entropnbcell.maxEntropPDMP,'--',  c='pink')
    plt.plot(25,0,"--",c='grey',label="min and max")
    #plt.ylim(0,2)
    plt.axvline(100,label="default value",color='grey')
    #plt.legend(bbox_to_anchor=(1.0,0.3))
    plt.xlabel("number of cells")
    plt.ylabel("Entropy")
    #plt.grid()
    
def graph_PDMPs(entroptimegapPDMPs_tr,t1,legend):
    PDMPs = ["","PDMP","PDMP1", "PDMP2", "PDMP3", "PDMP4", "PDMP5"]
    colors = ['blue' ,'red','orange',"green",'purple', 'hotpink','gold']
    #moys = [ "MoyEntrop","MoyEntropPDMP","MoyEntropPDMP1", "MoyEntropPDMP2", "MoyEntropPDMP3", "MoyEntropPDMP4", "MoyEntropPDMP5"]
    #stds = ["stdEntrop","stdEntropPDMP","stdEntropPDMP1", "stdEntropPDMP2", "stdEntropPDMP3", "stdEntropPDMP4", "stdEntropPDMP5"]
    moys = [ "MoyEntrop","MoyEntropPDMP","MoyEntrop1", "MoyEntrop2", "MoyEntrop3", "MoyEntrop4", "MoyEntrop5"]
    stds = ["stdEntrop","stdEntropPDMP","stdEntrop1", "stdEntrop2", "stdEntrop3", "stdEntrop4", "stdEntrop5"]
    for pdmp, color,moy,std in zip(PDMPs, colors,moys,stds):

        plt.plot(entroptimegapPDMPs_tr['timegap']-t1, entroptimegapPDMPs_tr[moy],marker="o", c=color, label=f"{pdmp}")#{label}{coef})
        plt.errorbar(
            entroptimegapPDMPs_tr.timegap-t1, entroptimegapPDMPs_tr[moy], 
            yerr=entroptimegapPDMPs_tr[std],
            fmt='none', ecolor=color,elinewidth=1.5, capsize=3, alpha=0.8)

    #plt.legend(bbox_to_anchor=(1.0,0.3))
    plt.xlabel(legend)
    plt.ylabel("Entropy ")
    #plt.grid(True)
    plt.show()
    
def graph_metrics(entroptimegapmetriq,t1,legend):
    colors = [ 'blue' ,"green",'purple', 'hotpink','gold']
    # Entropy
    plt.plot(entroptimegapmetriq.timegap-t1, entroptimegapmetriq.MoyEntrop, c='blue', label='Entropy',marker='o')
    plt.errorbar(
        entroptimegapmetriq.timegap-t1, entroptimegapmetriq.MoyEntrop, 
        yerr=entroptimegapmetriq.stdEntrop,
        fmt='none', ecolor='blue', elinewidth=1.5, capsize=3, alpha=0.8)
    
    # TV
    plt.plot(entroptimegapmetriq.timegap-t1, entroptimegapmetriq.MoyTV, c='orange', label='Total Variation Distance',marker='o')
    plt.errorbar(
        entroptimegapmetriq.timegap-t1, entroptimegapmetriq.MoyTV, 
        yerr=entroptimegapmetriq.stdTV,
        fmt='none', ecolor='orange', elinewidth=1.5, capsize=3, alpha=0.8)
    # JS
    plt.plot(entroptimegapmetriq.timegap-t1, entroptimegapmetriq.MoyJS, c='purple', label='Jensen-Shannon Divergence',marker='o')
    plt.errorbar(
        entroptimegapmetriq.timegap-t1, entroptimegapmetriq.MoyJS, 
        yerr=entroptimegapmetriq.stdJS,
        fmt='none', ecolor='purple', elinewidth=1.5, capsize=3, alpha=0.8)
    # HPP / PDMP
    plt.plot(entroptimegapmetriq.timegap-t1, entroptimegapmetriq.MoyEntropPDMP,"--", c='blue',marker='o')#, label=r'$HPP$')
    plt.errorbar(
        entroptimegapmetriq.timegap-t1, entroptimegapmetriq.MoyEntropPDMP, 
        yerr=entroptimegapmetriq.stdEntropPDMP,
        fmt='none', ecolor='blue',elinewidth=1.5, capsize=3, alpha=0.8)
    
    plt.plot(entroptimegapmetriq.timegap-t1, entroptimegapmetriq.MoyTV_PDMP,"--", c='orange',marker='o')#, label=r'$HPP$')
    plt.errorbar(
        entroptimegapmetriq.timegap-t1, entroptimegapmetriq.MoyTV_PDMP, 
        yerr=entroptimegapmetriq.stdTV_PDMP,
        fmt='none', ecolor='orange',elinewidth=1.5, capsize=3, alpha=0.8)
    plt.plot(entroptimegapmetriq.timegap-t1, entroptimegapmetriq.MoyJS_PDMP,"--", c='purple',marker='o')#, label=r'$HPP$')
    plt.errorbar(
        entroptimegapmetriq.timegap-t1, entroptimegapmetriq.MoyJS_PDMP, 
        yerr=entroptimegapmetriq.stdJS_PDMP,
        fmt='none', ecolor='purple',elinewidth=1.5, capsize=3, alpha=0.8)
    plt.plot(entroptimegapmetriq.timegap-t1, entroptimegapmetriq.MoyFrobenius_PDMP,"--", c='green',marker='o')#, label=r'$HPP$')
    plt.errorbar(
        entroptimegapmetriq.timegap-t1, entroptimegapmetriq.MoyFrobenius_PDMP, 
        yerr=entroptimegapmetriq.stdFrobenius_PDMP,
        fmt='none', ecolor='green',elinewidth=1.5, capsize=3, alpha=0.8)
    # Frobenius
    plt.plot(entroptimegapmetriq.timegap-t1, entroptimegapmetriq.MoyFrobenius, c='green', label='Frobenius Norm',marker='o')
    plt.errorbar(
        entroptimegapmetriq.timegap-t1, entroptimegapmetriq.MoyFrobenius, 
        yerr=entroptimegapmetriq.stdFrobenius,
        fmt='none', ecolor='green', elinewidth=1.5, capsize=3, alpha=0.8)
    
    #plt.legend(bbox_to_anchor=(1.0, 0.3))
    plt.xlabel(legend)
    plt.ylabel("Metrics")

    #plt.grid(True)
    plt.show()
def graph_GRNs(df00_,t1,legend):
    coefs = [ "GRN1", "GRN2","GRN3", "GRN4", "GRN5"]
    colors = [ 'blue' ,"green",'purple', 'hotpink','gold']
    labels = [ 'MoyEntropGRN1', 'MoyEntropGRN2','MoyEntropGRN3', 'MoyEntropGRN4', 'MoyEntropGRN5']
    stds = ['stdEntropGRN1', 'stdEntropGRN2','stdEntropGRN3','stdEntropGRN4', 'stdEntropGRN5']
    MIN = [ "minGRN1", "minGRN2","minGRN3", "minGRN4", "minGRN5"]
    MAX = [ "maxGRN1", "maxGRN2","maxGRN3", "maxGRN4", "maxGRN5"]
    for coef, color, label, std,min_col, max_col in zip(coefs, colors, labels,stds,MIN, MAX):
        plt.plot(df00_['timegap']-t1 , df00_[label], label=f"{coef}", c=color,marker="o")#{label} ({coef})

        plt.errorbar(
        df00_['timegap']-t1, df00_[label], 
        yerr=df00_[std],
        fmt='none', ecolor=color,elinewidth=1.5, capsize=3, alpha=0.8)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel(legend)
    plt.ylabel("Entropy")
    #plt.grid(True)
    #plt.axvline(t1,label="$t_1$",color='grey')

    plt.tight_layout()

def graph_traj(csv):
    # Recharger les données sauvegardées
    df_all = pd.read_csv(csv)

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))

    # Palette de couleurs pour les lignes
    trajectoires = df_all["Trajectory"].unique()
    colors = plt.cm.plasma(np.linspace(0, 1, len(trajectoires)))

    for idx, traj_id in enumerate(trajectoires):
        data = df_all[df_all["Trajectory"] == traj_id]

        P1 = data["Protein1"].values
        P2 = data["Protein2"].values

        # Tracer la ligne
        ax1.plot(P1, P2, color="grey", label=f"Trajectoire {traj_id}")

        # Marquer le premier et dernier point
        ax1.scatter(P1[0], P2[0], color="purple", s=40, zorder=3)
        ax1.scatter(P1[-1], P2[-1], color="darkgreen", s=40, zorder=3)

    ax1.set_xlabel("Protein 1")
    ax1.set_ylabel("Protein 2")
    ax1.set_xlim(0.0,1.9)  
    ax1.set_ylim(0.0,1.25)

    plt.show()
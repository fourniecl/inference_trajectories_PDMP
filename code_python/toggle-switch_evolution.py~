import general_function 

##################PARAMETERS#############
n = 100
k_on_1, k_on_2 = 5, 5
d_M_1, d_M_2 = 10, 10
d_P_1, d_P_2 = 5, 5
s_1, s_2 = 10, 10
pa_1, pa_2 = 10, 10
pi_1, pi_2 = -10, -10
parameters = [k_on_1, k_on_2, d_M_1, d_M_2, d_P_1, d_P_2, s_1, s_2, pa_1, pa_2, pi_1, pi_2]

G = 2
tms = np.linspace(0, 100, 100)
t1 = 40
t2 = 60

CI=(0,0)
#########################PLOT##############
S = simu(n, toggle_switch(G, parameters), CI, tms)
basic_plot(S[0])
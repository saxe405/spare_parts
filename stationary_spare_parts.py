import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

lambda_1 = 0.2 #rate of a machine going from state 1 to 2
lambda_2 = 0.3 # rate of a machine going from state 2 to 3
mu = 0.3 # rate of delivery of spare parts
N = 10 # number of spareparts
M = 40 # number of machines to serve

def matrix_F_n(lambda_1, M, n, N):
	Fn = np.zeros((M+1,M+1))	
	for k in range(1,M+1):
		Fn[k][k-1] = k*lambda_1
	if(n>N):
		i = n-N				
		for k in range(M-i,M+1):
			Fn[k][k-1] = (M-i)*lambda_1			
	return Fn

def matrix_B_n(mu,n,M):
	return mu*n*np.eye(M+1)

def matrix_L_n(lambda_2,lambda_1,mu,M,n,N):
	Fn = matrix_F_n(lambda_1,M,n,N)
	Bn = matrix_B_n(mu,n,M)
	Ln = -1*(np.diag(Fn.sum(axis=1)+Bn.sum(axis=1)))  #row sum = 0
	for k in range(1,M+1):
		Ln[k-1][k] = (M-k+1)*lambda_2
		Ln[k-1][k-1] -= (M-k+1)*lambda_2	
	return Ln


def R_k(lambda_2,lambda_1,mu,M,N,k):
	R = -1*np.dot(matrix_F_n(lambda_1,M,M+N-1,N),np.linalg.inv(matrix_L_n(lambda_2,lambda_1,mu,M,M+N,N)))  # this is R_{M+N}
	if k > 1:
		for j in reversed(range(k,M+N)):		
			R = -1*np.dot(matrix_F_n(lambda_1,M,j-1,N),np.linalg.inv(matrix_L_n(lambda_2,lambda_1,mu,M,j,N)+ np.dot(R,matrix_B_n(mu,j+1,M)) ))
		return R
	if k==0:
		return -1*np.dot(matrix_B_n(mu,1,M),np.linalg.inv(matrix_L_n(lambda_2,lambda_1,mu,M,0,N)))					# R0 = -B_1L0^-1
	return	np.eye(M+1)			#R_1 = I

def pi(lambda_2,lambda_1,mu,M,N):
	F0 = matrix_F_n(lambda_1,M,0,N)
	L1 = matrix_L_n(lambda_2,lambda_1,mu,M,1,N)
	B2 = matrix_B_n(mu,2,M)
	R0 = R_k(lambda_2,lambda_1,mu,M,N,0)
	R2 = R_k(lambda_2,lambda_1,mu,M,N,2)
	e = np.ones((M+1,1))

	A = np.dot(R0,F0) + L1 + np.dot(R2,B2)

	normalizing_vec = np.dot(R0,e)
	R = np.eye(M+1)
	for j in range(1,M+N+1):
		R = np.dot(R,R_k(lambda_2,lambda_1,mu,M,N,j))
		normalizing_vec += np.dot(R,e)
	A = np.block([
			[A],
			[normalizing_vec.transpose()]
		])
	b = np.zeros((1,M+2)).transpose()
	b[M+1][0] = 1
	pi1 = np.linalg.lstsq(A,b,rcond=None)[0].transpose() #solving linear system of equations
	pi_vec = np.block([
				[np.dot(pi1,R_k(lambda_2,lambda_1,mu,M,N,0))]
			])

	#print(R_k(lambda_2,lambda_1,mu,M,N,0))
	for j in range(1,M+N):
		pi1 = np.dot(pi1,R_k(lambda_2,lambda_1,mu,M,N,j))
		pi_vec = np.block([
				[pi_vec],
				[pi1]
			])
	return(pi_vec)

def prob_sparepart_not_available(lambda_2,lambda_1,mu,M,N):
	Pr = 0
	pi_vector = pi(lambda_2,lambda_1,mu,M,N)
	for i in range(N,M+N):
		Pr += sum( [ pi_vector[i][j] for j in range(M-(i-N),M+1) ])
	return Pr

#print(prob_sparepart_not_available(lambda_2, lambda_1,mu,M,N))
# check the dependence of this probability on the parameters N,M,mu, 
# others we can't change but we can consider case 1: low lambda_1, case 2: high lambda_1 et
mu_vals = [0.3,0.4,0.5]
N_vals = [x for x in range(5, 50, 4)]
results = [[] for x in range(len(mu_vals)) ]
results_log = [[] for x in range(len(mu_vals)) ]
#mu_vals = [x/100 for x in range(40,100,5)]	
for i in range(len(mu_vals)):
	for N in N_vals:
		mu = mu_vals[i]
		res = prob_sparepart_not_available(lambda_2, lambda_1,mu,M,N)
		results_log[i].append(math.log(res,10))
		results[i].append(res)
#prob_mat = np.zeros((len(N_vals),len(mu_vals)))
#for i in range(len(N_vals)):
#	for j in range(len(mu_vals)):
#		prob_mat[i][j] = prob_sparepart_not_available(lambda_2, lambda_1,mu_vals[j],M,N_vals[i])

#ax = sns.heatmap(prob_mat)
plt.figure(1)
fig, ax = plt.subplots()
plt.plot(N_vals, results_log[0], 'b-', label = 'mu = 0.3' )
plt.plot(N_vals, results_log[1], 'b--', label = 'mu = 0.4' )
plt.plot(N_vals, results_log[2], 'b-.', label = 'mu = 0.5' )
plt.xlabel('Starting inventory (N)')
plt.ylabel('log( Prob spare part not available )')
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-small')
legend.get_frame().set_facecolor('#00FFCC')
plt.tight_layout()
plt.savefig("C:/Users/sapoorv/Downloads/CODE/python/SP management/prob_NA_N_log.png")

plt.figure(2)
fig, ax = plt.subplots()
plt.plot(N_vals,results[0], 'b-', label = 'mu = 0.3')
plt.plot(N_vals,results[1], 'b--', label = 'mu = 0.4')
plt.plot(N_vals,results[2], 'b-.', label = 'mu = 0.5')
plt.xlabel('Starting inventory (N)')
plt.ylabel('Prob spare part not available')
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-small')
legend.get_frame().set_facecolor('#00FFCC')
plt.tight_layout()
plt.savefig("C:/Users/sapoorv/Downloads/CODE/python/SP management/prob_NA_N.png")
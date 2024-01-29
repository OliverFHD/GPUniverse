try:
    import cupy as np
    try:
        print(np.cuda.Device())
    except:
        import numpy as np
except ImportError:
    import numpy as np
import sys


data_type = np.cdouble
data_type_r = np.double



#
# Cliffort_generators
# -- computes an array of generators for the Clifford algebra, via the iterative scheme described in
#    
#
# two_n (int): total number of generators (which also determines the dimension of the representation)
#
def Cliffort_generators(two_n):
    #
    if two_n%2 != 0:
        print("WARNING: the function Cliffort_generators(n) is implemented only for even two_n.")
        return -99
    #
    n = int(two_n/2+0.5)
    N = 2**n
    dummy = np.zeros((N,N), dtype=data_type)
    generators = []
    #
    if n == 1:
        M = 0.0+dummy
        M[0,0] = 1.0j
        M[1,1] = -1.0j
        generators.append(np.copy(M))
        M = 0.0+dummy
        M[0,1] = 1.0
        M[1,0] = -1.0
        generators.append(np.copy(M))
        P = 0.0+dummy
        P[0,1] = 1.0j
        P[1,0] = 1.0j
        generators.append(np.copy(P))
        return generators
    #
    N_lower = 2**(n-1)
    generators_lower = Cliffort_generators(two_n-2)
    for i in range(0,2*(n-1)):
        M = 0.0+dummy
        M[:N_lower,:N_lower] = generators_lower[i]
        M[N_lower:,N_lower:] = -generators_lower[i]
        generators.append(np.copy(M))
    #
    P_times_iPOWn = generators_lower[2*(n-1)]*1.0j**n
    M = 0.0+dummy
    M[:N_lower,:N_lower] = P_times_iPOWn
    M[N_lower:,N_lower:] = -P_times_iPOWn
    generators.append(np.copy(M))
    #
    M = 0.0+dummy
    M[:N_lower,N_lower:] = np.diag(np.ones(N_lower, dtype=data_type))
    M[N_lower:,:N_lower] = -np.diag(np.ones(N_lower, dtype=data_type))
    generators.append(np.copy(M))
    #
    P_squared_times_iPOWn = np.diag(np.ones(N_lower, dtype=data_type))*1.0j**n
    if (two_n-2)%4 == 2:
        P_squared_times_iPOWn *= -1.0
    M = 0.0+dummy
    M[:N_lower,N_lower:] = P_squared_times_iPOWn
    M[N_lower:,:N_lower] = P_squared_times_iPOWn
    generators.append(np.copy(M))
    #
    return generators

#
# simulate_overlapping_qubits
# -- returns the Hamiltonian operators, the total occupation number operator, and the occupation number operator & raising operator of
#    one individual qubit in a system with several overlapping qubit degrees of freedom
#
# seed (int): random seed
# N_qubit (int): number of overlapping qubits
# n_effective (int): log2 of dimension of Hilbert space
# Hamiltonian_mode (0,1): 0 == fiducial Hamiltonian ; 1 == alternative Hamiltonian
#
# if \hat N_i is the occupation number operator of the i-th qubit, then the fiducial Hamiltonian is
#      \hat H = 0.5 \sum_{i=1}^{N_qubit} (2\hat N_i - 1)
#
def simulate_overlapping_qubits(N_qubit, n_effective, Hamiltonian_mode=0, seed=5, Cliffort_gens=[]):
    #
    np.random.seed(seed)
    if len(Cliffort_gens) == 0:
        Cliffort_gens = Cliffort_generators(2*n_effective)
    #
    N_H = int(2**n_effective+0.1) # <-- dimension of Hilbertspace
    two_N_qubit = int(2*N_qubit+0.1)
    #
    Unit = np.identity(N_H).astype(data_type)
    #
    imbedding_bases = np.random.normal(0.0, 1.0, size=(two_N_qubit, 2*n_effective)).astype(data_type_r)
    imbedding_bases = (imbedding_bases.T/np.linalg.norm(imbedding_bases, axis=1)).T
    #
    imbedding_bases[1::2] -= np.transpose(np.sum(imbedding_bases[::2]*imbedding_bases[1::2], axis=1)*np.transpose(imbedding_bases[::2]))
    imbedding_bases[1::2] = (imbedding_bases[1::2].T/np.linalg.norm(imbedding_bases[1::2], axis=1)).T
    #
    #
    Hamiltonian = np.zeros((N_H, N_H)).astype(data_type)
    E0 = np.zeros((N_H, N_H)).astype(data_type)
    F0 = np.zeros((N_H, N_H)).astype(data_type)
    #
    N_total = np.zeros((N_H, N_H)).astype(data_type)
    N_first_mode = np.zeros((N_H, N_H)).astype(data_type)
    H_first_mode = np.zeros((N_H, N_H)).astype(data_type)
    Cdagger_first_mode = np.zeros((N_H, N_H)).astype(data_type)
    #
    if Hamiltonian_mode == 1:
        C = np.zeros((N_H, N_H)).astype(data_type)
        Cdagger = np.zeros((N_H, N_H)).astype(data_type)
        for i in range(0,int(N_qubit+0.1)):
            E0 = np.tensordot(imbedding_bases[2*i,:], Cliffort_gens[:-1], axes=([0],[0]))
            F0 = np.tensordot(imbedding_bases[2*i+1,:], Cliffort_gens[:-1], axes=([0],[0]))
            C = 0.5*(E0 + 1.0j*F0)
            zi_dagger = imbedding_bases[2*i,:] - 1.0j*imbedding_bases[2*i+1,:]
            for j in range(i,int(N_qubit+0.1)):
                if i != j:
                    E0 = np.tensordot(imbedding_bases[2*j,:], Cliffort_gens[:-1], axes=([0],[0]))
                    F0 = np.tensordot(imbedding_bases[2*j+1,:], Cliffort_gens[:-1], axes=([0],[0]))
                Cdagger = 0.5*(E0 - 1.0j*F0)
                zj = imbedding_bases[2*j,:] + 1.0j*imbedding_bases[2*j+1,:]
                #
                prod = np.sum(zi_dagger*zj)
                addition = prod*(Cdagger.dot(C) - 0.5*Unit)
                Hamiltonian += addition
                if i != j:
                    Hamiltonian += addition.T.conj()
                #
            #
        #
        Hamiltonian /= 2.0*(1.0+(N_qubit - 1.0)/n_effective)
    #
    #
    #
    for i in range(0,int(N_qubit+0.1)):
        E0 = np.tensordot(imbedding_bases[2*i,:], Cliffort_gens[:-1], axes=([0],[0]))
        F0 = np.tensordot(imbedding_bases[2*i+1,:], Cliffort_gens[:-1], axes=([0],[0]))
        N_total += E0.dot(F0)
        #
        if i == 0:
            H_first_mode = 0.25j*(E0.dot(F0) - F0.dot(E0))
            N_first_mode = H_first_mode + 0.5*Unit
            Cdagger_first_mode  = 0.5*(E0 + 1.0j*F0)
            #
        #
    #
    #
    #
    N_total = 0.25j*(N_total - np.transpose(N_total).conj())
    if Hamiltonian_mode != 1:
        Hamiltonian = N_total.copy()
    #
    N_total = N_total + 0.5*N_qubit*Unit
    #
    #
    return Hamiltonian, N_total, N_first_mode, Cdagger_first_mode



#
# return_Emin_and_d2N_first_dt2
# -- returns minimum energy, estimate for lifetime of plane wave, and plane wave state vector
#
# Hamiltonian (matrix): Hamiltonian operator
# N_first_mode (matrix): occupation number operator of the 1st qubit mode
#
def return_Emin_and_d2N_first_dt2(Hamiltonian, N_first_mode):
    #
    eigenvalues_N_first, eigenvectors_N_first = np.linalg.eigh(N_first_mode)
    indxs = np.where(eigenvalues_N_first > 0.5)[0]
    N_excited = len(indxs)
    excited_block = eigenvectors_N_first[:,indxs]
    #
    eigenvalues_H_JL, eigenvectors_H_JL = np.linalg.eigh(Hamiltonian)
    evec_min_E = eigenvectors_H_JL[:, 0]
    E_min = eigenvalues_H_JL[0]
    #
    #
    dN_first_dt = 1.0j*(Hamiltonian.dot(N_first_mode) - N_first_mode.dot(Hamiltonian))
    d2N_first_dt2 = 1.0j*(Hamiltonian.dot(dN_first_dt) - dN_first_dt.dot(Hamiltonian))
    #
    sub_d2N_first_dt2 = excited_block.T.conj().dot(d2N_first_dt2)
    sub_d2N_first_dt2 = sub_d2N_first_dt2.dot(excited_block)
    sub_H_JL = excited_block.T.conj().dot(Hamiltonian)
    sub_H_JL = sub_H_JL.dot(excited_block)
    eigenvalues_sub_d2N_first_dt2, eigenvectors_sub_d2N_first_dt2 = np.linalg.eigh(sub_d2N_first_dt2)
    #
    indxs = np.where(eigenvalues_sub_d2N_first_dt2 < np.mean(eigenvalues_sub_d2N_first_dt2))[0]
    excited_block_sub = eigenvectors_sub_d2N_first_dt2[:,indxs]
    sub_sub_H_JL = excited_block_sub.T.conj().dot(sub_H_JL)
    sub_sub_H_JL = sub_sub_H_JL.dot(excited_block_sub)
    eigenvalues_sub_sub_H_JL, eigenvectors_sub_sub_H_JL = np.linalg.eigh(sub_sub_H_JL)
    #
    evec_plane_wave_step_1 = np.tensordot(excited_block_sub, eigenvectors_sub_sub_H_JL[:,0], axes=([1],[0]))
    evec_plane_wave = np.tensordot(excited_block, evec_plane_wave_step_1, axes=([1],[0]))
    #
    d2N_first_dt2_expec = np.max(eigenvalues_sub_d2N_first_dt2)
    d2N_first_dt2_expec_2 = np.einsum('i, ij, j', evec_plane_wave.conj(), d2N_first_dt2, evec_plane_wave)
    if np.abs(d2N_first_dt2_expec - d2N_first_dt2_expec_2)/d2N_first_dt2_expec > 0.001:
        print("potential inconsistency in return_Emin_and_d2N_first_dt2:")
        print("d2N_first_dt2_expec=",d2N_first_dt2_expec)
        print("d2N_first_dt2_expec_2=",d2N_first_dt2_expec_2)
    #
    T_scramble = np.sqrt(-1.0/d2N_first_dt2_expec)
    #
    return E_min, T_scramble, evec_plane_wave


N_qubit = int(sys.argv[1])
n_effective = int(sys.argv[2])
Hamiltonian_mode = int(sys.argv[3])
seed = int(sys.argv[4])


Cliffort_gens = np.array(Cliffort_generators(2*n_effective))

Hamiltonian, N_total, N_first_mode, Cdagger_first_mode = simulate_overlapping_qubits(N_qubit, n_effective, Hamiltonian_mode, seed, Cliffort_gens)

E_min, T_scramble, evec_plane_wave = return_Emin_and_d2N_first_dt2(Hamiltonian, N_first_mode)

print("N_qubit =",N_qubit)
print("n_effective =",n_effective)
print("Hamiltonian_mode =",Hamiltonian_mode)
print("")
print("E_min =",E_min)
print("E_min/(-0.5 N_qubit) =",E_min/(-0.5*N_qubit))
print("T_scramble =",T_scramble)
print("<d2N_dt2> =",-1.0/T_scramble**2)
print("")
print("WARNING: these results are only from a single simulation. Average over many, in order to improve precision.")


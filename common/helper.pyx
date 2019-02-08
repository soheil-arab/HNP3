
# import numpy as np
cimport numpy as np
from libc.math cimport exp,log

cpdef dict compute_omega_u(list active_set,double time,np.ndarray betas):
    cdef dict sum_omega_u = dict()
    cdef int user
    cdef int k
    cdef double beta_k
    cdef double t

    # This function returns a dict that for each pair of user u returns the
    # sum_{e\inD_u}1/beta_{z_e}(1-exp(beta_{z_e}(time-t_e))) which is used in computing
    # the likelihood of time of events in the Hawkes process.
    for event in active_set:
        user = event['user']
        k = event["topic"]
        beta_k = betas[k]
        t = event["time"]
        sum_omega_u[user] = sum_omega_u.get(user, 0) + 1 / beta_k * (
            1 - exp(-beta_k * (time - t)))
        # if sum_omega_u[(user_s, user)]<0:
            # print('K_s is %f and beta_k is %f'%(k_s,beta_k))
    return sum_omega_u

cpdef double document_log_likelihood(dict etas,
                                     int doc_words_num, dict doc_words, double eta_sum,
                                     np.ndarray[double, mode="c", ndim=1] default_eta):
    cdef double likelihood = 0
    for l in range(doc_words_num):
        likelihood -= log(eta_sum + l)

    for w,c in doc_words.items():
        for l in range(c):
            likelihood += log(default_eta[w]+etas.get(w,0) + l)
    return likelihood

cpdef double compute_beta_log_likelihood(list active_set, double time, np.ndarray[double, mode="c", ndim=1] betas, int user_num, double a, double b
                                         , np.ndarray[long, mode="c", ndim=2] adjacency_matrix, np.ndarray[long, mode="c", ndim=2] counts):

    cdef dict sum_omega_u = compute_omega_u(active_set,time,betas)
    cdef double result = 0.0
    for user in range(user_num):
        if not user in sum_omega_u:
            continue
        for user_prime in range(user_num):
            if adjacency_matrix[user][user_prime]!=0:
                result-=(a + counts[user][user_prime]) * log(b + sum_omega_u[user])
    return result
import numpy as np
import math
import numba
import pandas as pd


# parametry symulacji

path = 'D:\\Studia\\KMS\\KMS_2'
N = 100
d_tau = 0.0001
kappa = 2
omega = 4 * math.pi ** 2 / 2
n_steps = 10000


# def compute_hamiltonian(psi_r, psi_i):
#
#     h_real = []
#     h_imaginary = []
#     h_real[0], h_real[N], h_imaginary[0], h_imaginary[N] = 0, 0, 0, 0
#     h_real[1:N-1] = [-0.5 * (psi_r[k + 1] + psi_r[k - 1] - 2 * psi_r[k]) / (d_x ** 2) + kappa
#                      * (points[k] - 0.5) * psi_r[k] * math.sin(omega * tau) for k in range(1, N)]
#
#     h_imaginary[1:N-1] = [-0.5 * (psi_i[k + 1] + psi_i[k - 1] - 2 * psi_i[k]) / (d_x ** 2)
#                           + kappa * (points[k] - 0.5) * psi_i[k] * math.sin(omega * tau) for k in range(1, N)]
#
#     return h_real, h_imaginary

@numba.jit
def compute_h(psi):

    new_h = np.zeros(N+1)
    new_h[1:N] = [-0.5 * ((psi[k + 1] + psi[k - 1] - 2 * psi[k]) / (d_x ** 2)) + kappa * (points[k] - 0.5) * psi[k]
                    * math.sin(omega * tau) for k in range(1, N)]
    return np.array(new_h)


@numba.jit
def compute_psi_tmp(psi_re, h_im):
    return psi_re + (h_im * d_tau / 2)


@numba.jit
def compute_new_psi_im(psi_im, h_re):
    return psi_im - (h_re * d_tau)


@numba.jit
def compute_new_psi_re(psi_re, h_im):
    return psi_re + (h_im * d_tau / 2)


if __name__ == '__main__':
    tau = 0
    d_x = 1/N
    points = np.array([n*d_x for n in range(0, N+1)])

    n = 1
    psi_real = np.array([math.sqrt(2)*math.sin(n*math.pi*point) for point in points])
    psi_imaginary = np.array([0 for k in psi_real])
    h_real = compute_h(psi_real)
    h_imaginary = compute_h(psi_imaginary)

    for step in range(0, n_steps):
        tau += d_tau
        psi_real = compute_psi_tmp(psi_real, h_imaginary)
        h_real = compute_h(psi_real)
        psi_imaginary = compute_new_psi_im(psi_imaginary, h_real)
        h_imaginary = compute_h(psi_imaginary)
        psi_real = compute_new_psi_re(psi_real, h_imaginary)
        if step % 10 == 0:
            N_out = d_x * sum((psi_real ** 2) + (psi_imaginary ** 2))
            x_out = d_x * sum(points * ((psi_real ** 2) + (psi_imaginary ** 2)))
            eps_out = d_x * sum((psi_real * h_real) + (psi_imaginary * h_imaginary))
            proba_density = (psi_real ** 2) + (psi_imaginary ** 2)
            f = open(f"{path}//results_density.csv", "a")
            for ii in proba_density:
                f.write(f'{ii},')
            f.write('\n')
            f.close()
            f = open(f"{path}//results.csv", "a")
            f.write(f'{tau},{N_out},{x_out},{eps_out}\n')
            f.close()






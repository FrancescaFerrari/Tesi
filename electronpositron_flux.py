from numpy import pi, exp, log, sqrt, inf
from scipy.integrate import quad
import matplotlib.pyplot as plt


m = 0.511e-3  # GeV
c = 3.0e+10  # cm/s
sigmaT = 6.65246e-25  # cm^2
h = 2.0 * pi * 6.582e-25  # GeV*s


def main():

    N = 10000
    r = 8.33  # kpc
    rs = 0.25  # kpc


    Es = 1.0e+4  # GeV


    xmin = 1.0; xmax = 1.0e+4; dx = (xmax - xmin) / N
    x = []; y = []
    for i in range(0, N, 1):
        E = xmin + i*dx  # GeV
        flusso = phi(E, Es, r, rs)
        x.append(E)
        y.append(E**3 * flusso)

    plt.plot(x, y, label="burst-like", linewidth=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energia [GeV]")
    plt.ylabel("$E^{3}\phi_{e^{+}} [GeV^2/cm^2/s/sr]$")
    plt.ylim(1.0e-11, 1.0e-7)
    plt.grid()
    plt.legend()
    plt.show()





# coefficiente di diffusione
def K(E):
    E0 = 1.0  # GeV
    #K0 = 0.0967  # kpc^2/Myr
    K0 = 0.0967 / (3.154 * 1.0e+13)  # kpc^2/s
    delta = 0.408
    return K0 * (E / E0)**delta


# densità di fotoni iniziali per radiazione di corpo nero definita a meno della costante

# CMB
def photon_density(eps):
    T = 2.726  # K
    kB = 8.6167e-14  # GeV/K

    if eps / (kB * T) < 700 and eps / (kB * T) > 1.0e-6:
        return eps**2 / (exp(eps / (kB * T)) - 1.0)
    else:
        return 0.0


# rate di collisioni per ICS definito a meno della costante
def collision_rate(q, eps, E):
    Gamma = 4.0 * eps * E / m**2
    return (photon_density(eps) / eps) * \
           (2.0 * q * log(q) + 1.0 + q - 2.0 * q**2 + 0.5 * (Gamma * q)**2 * (1.0 - q) / (1.0 + Gamma * q))


# funzione da integrare prima in dq e poi in deps definita a meno della costante cost = cost1 * cost2
def f(q, eps, E):
    Gamma = 4.0 * eps * E / m**2
    return (E * Gamma)**2 / (1.0 + Gamma * q)**2 * (q / (1.0 + Gamma * q) - m**2 / (4.0 * E**2)) * \
           collision_rate(q, eps, E)


# integro in dq
def fint_dq(eps, E):
    res_dq, err_dq = quad(f, m**2 / (4.0 * E**2), 1.0, args=(eps, E))
    return res_dq

# integro in deps
def fint_deps(E):
    res_deps, err_deps = quad(fint_dq, 0.0, 1.0e-11, args=(E))
    return res_deps


# perdite di energia
def b(E):

    # costante CMB
    cost1 = 8.0 * pi / (h * c) ** 3

    # costante collision rate
    cost2 = 3.0 * sigmaT * c * m ** 2 / (4.0 * E**2)

    cost = cost1 * cost2

    return cost * fint_deps(E)


# funzione da integrare in dE per ottene la lunghezza tipica di propagazione
def g(E):
    return K(E)/b(E)


def gint(E, Es):
    res, err = quad(g, E, Es)
    return res


def propagation_scale(Eearth, Es):
    return sqrt(4.0 * gint(Eearth, Es))


def Q(E):
    Ec = 1.0e+6  # GeV
    E0 = 1.0  # GeV
    gamma = 1.8
    Q0 = 1.3450528572198825e+48  # 1/GeV
    return Q0 * (E / E0)**(-gamma) * exp(- E / Ec)

def N(E, Es, r, rs):
    return (b(Es)/b(E)) * (1/(pi * propagation_scale(E, Es)**2)**(3.0 / 2.0)) * \
           exp(- (abs(r - rs))**2 / propagation_scale(E, Es)**2) * Q(Es) / \
           (1.0e+63 * (3.086)**3)  # 1/cm^3/GeV


def phi(E, Es, r, rs):
    return (c / (4.0 * pi)) * N(E, Es, r, rs)  # 1/cm^2/GeV/sr/s

main()
import numpy as np
from matplotlib import colormaps
import matplotlib.pyplot as plt
import pint
from scipy.special import voigt_profile

plt.rcParams['figure.figsize'] = (8, 5)


# Set up units
ureg = pint.UnitRegistry()
z = 0
e = 4.8032e-10 * ureg.esu
m_e = ureg.electron_mass
c = ureg.speed_of_light



def voigt(v, b, gamma, lamda_ul):
    # input units: v (km/s), b (km/s), gamma (1/s), lamda_ul (angstrom)
    gamma = gamma / ureg.s
    b = b * ureg.km / ureg.s
    lamda_ul = lamda_ul * ureg.angstrom

    # Calculate Voigt parameters
    gamma_0 = gamma / (4 * np.pi) * lamda_ul
    sigma = b / np.sqrt(2)

    # Convert units to base units
    sigma = sigma.to('km/s').magnitude
    gamma_0 = gamma_0.to('km/s').magnitude

    # Calculate Voigt profile
    return voigt_profile(v, sigma, gamma_0) * ureg.s / ureg.km

def test_voigt():
    v = np.arange(-300000, 300000, 0.1)
    b = 10
    gamma = 100
    lamda_ul = 1000
    voigt_v = voigt(v, b, gamma, lamda_ul).magnitude

    assert np.allclose(np.trapz(voigt_v, v), 1, rtol=1e-4), fr"Integral of Voigt profile should is {np.trapz(voigt_v, v)} instead of 1"

def tau(v, N, f_lu, lamda_ul, b, gamma):
    # input units: N (cm^-2), f_lu (dimensionless), lamda_ul (angstrom), b (km/s), v_window (km/s)
    phi_v = voigt(v, b, gamma, lamda_ul)
    N = N * ureg.cm ** -2
    f_lu = f_lu * ureg.dimensionless
    lamda_ul = lamda_ul * ureg.angstrom
    b = b * ureg.km / ureg.s
    a_0 = np.pi * e ** 2 / (m_e * c) * N * f_lu * lamda_ul 
    tau_v = a_0 * phi_v
    tau_v = tau_v.to_base_units()
    assert tau_v.units == ureg.dimensionless, f"Units of tau_v is {tau_v.units}"
    
    return tau_v.magnitude

def calculate_W(N, f_lu, lamda_ul, b, gamma, approx=False, v_window=300000):
    # input units: N (cm^-2), f_lu (dimensionless), lamda_ul (angstrom), b (km/s), gamma (1/s), v_window (km/s)
    if approx:
        tau_0 = tau(0, N, f_lu, lamda_ul, b, gamma)
        b = b * ureg.km / ureg.s
        gamma = gamma / ureg.s
        lamda_ul = lamda_ul * ureg.angstrom

        if tau_0 < 1.25393:
            W_total = np.sqrt(np.pi) * b / c  * tau_0 / (1 + tau_0 / (2 * np.sqrt(2))) 
        else:
            W_total = np.sqrt((2 * b / c) ** 2 * np.log(tau_0 / np.log(2)) + \
                                (b / c) * (gamma * lamda_ul / c) * ((tau_0 - 1.25393) / np.sqrt(np.pi))
                            )
            W_total = W_total.to('dimensionless')
        W_total = W_total * lamda_ul
        W_total = W_total.to('angstrom').magnitude  # Convert to angstrom

    else:
        v = np.arange(-v_window, v_window, 0.1)  # Velocity grid
        tau_v = tau(v, N, f_lu, lamda_ul, b, gamma)  # Compute optical depth
        W_total = np.trapz(1-np.exp(-tau_v), v) * ureg.km / ureg.s  # Integrate over velocity to get total equivalent width
        W_total = W_total * lamda_ul * ureg.angstrom / ureg.c
        W_total = W_total.to('angstrom').magnitude  # Convert to angstrom
    
    return W_total



def plot_logW_vs_logN(N_list, f_lu, lamda_ul, b_list, gamma, approx=False, v_window=300000):
    plt.figure()

    color_indices = np.linspace(0, 1, len(b_list))  # Normalize for colormap
    colors = colormaps['plasma'](color_indices)  #

    # Loop over each N in N_list
    for i, b in enumerate(b_list):
        for j, N in enumerate(N_list):
            W_total = calculate_W(N, f_lu, lamda_ul, b, gamma, approx=approx, v_window=v_window)

            if j == 0:
                plt.plot(np.log10(N), np.log10(W_total), 'o', color=colors[i], label=fr"b = {b:.1f} km/s")
            else:
                plt.plot(np.log10(N), np.log10(W_total), 'o', color=colors[i])

    # Set plot labels and legend
    plt.title("Curve of Growth")
    plt.xlabel("log(N / cm$^{-2}$)")
    plt.ylabel(r"log(W$_{\lambda}$ / $\AA$)")
    plt.legend()
    plt.show()


def plot_F_vs_lamda_N(N_list, f_lu, lamda_ul, b, gamma, z=0, v_window=200):
    plt.figure()

    v = np.arange(-v_window, v_window, 0.1)
    lamda = v * ureg.km / ureg.s / c * lamda_ul * ureg.angstrom + lamda_ul * ureg.angstrom
    lamda = lamda.to('angstrom').magnitude

    color_indices = np.linspace(0, 1, len(N_list))  # Normalize for colormap
    colors = colormaps['plasma'](color_indices)  #

    for i, N in enumerate(N_list):
        tau_v = tau(v, N, f_lu, lamda_ul, b, gamma)
        # tau_0 = tau(0, N, f_lu, lamda_ul, b, gamma)
        # print(f"logN = {np.log10(N):.2f}, b = {b:.1f} km/s : tau_0 = {tau_0}")

        plt.plot(lamda*(1+z), np.exp(-tau_v), label=fr"logN = {np.log10(N):.1f}", color=colors[i])

    plt.title(fr"HI Ly$\alpha$ b = {b:.1f} km/s")
    plt.axhline(1, color='black', linestyle='--')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel(r"$\lambda$ (angstrom)")
    plt.ylabel("F")
    plt.legend()
    plt.show()

def plot_F_vs_lamda_b(N, f_lu, lamda_ul, b_list, gamma, z=0, v_window=200):
    plt.figure()

    v = np.arange(-v_window, v_window, 0.5)
    lamda = v * ureg.km / ureg.s / c * lamda_ul * ureg.angstrom + lamda_ul * ureg.angstrom
    lamda = lamda.to('angstrom').magnitude

    color_indices = np.linspace(0, 1, len(b_list))  # Normalize for colormap
    colors = colormaps['plasma'](color_indices)  #

    for i, b in enumerate(b_list):
        tau_v = tau(v, N, f_lu, lamda_ul, b, gamma)
        # tau_0 = tau(0, N, f_lu, lamda_ul, b, gamma)
        # print(f"logN = {np.log10(N):.2f}, b = {b:.1f} km/s : tau_0 = {tau_0}")

        plt.plot(lamda*(1+z), np.exp(-tau_v), label=fr"b = {b:.1f} km/s", color=colors[i])

    plt.title(fr"HI Ly$\alpha$ log N(HI)/cm$^{2}$ = {np.log10(N):.1f}")
    plt.axhline(1, color='black', linestyle='--')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel(r"$\lambda$ (angstrom)")
    plt.ylabel("F")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # HI Lyman alpha
    # f_lu = 0.4164
    # lamda_ul = 1215.7
    # gamma = 6.3e8
    # z = 0

    # # N_list = 10 ** np.arange(11, 22, 1, dtype=float)
    # # plot_F_vs_lamda_N(N_list, f_lu, lamda_ul, b, gamma, z, v_window=2000)

    # b_list = np.array([1, 2, 5, 10, 20, 30], dtype=float)
    # # plot_F_vs_lamda_b(10**12.5, f_lu, lamda_ul, b_list, gamma, z, v_window=200)
    # # plot_F_vs_lamda_b(10**16, f_lu, lamda_ul, b_list, gamma, z, v_window=200)
    # # # plot_F_vs_lamda_b(10**19, f_lu, lamda_ul, b_list, gamma, z, v_window=200)

    # N_list = 10 ** np.arange(10, 22, 0.5, dtype=float)
    # plot_logW_vs_logN(N_list, f_lu, lamda_ul, b_list, gamma, approx=True)


    # Fe II 2382.8
    approx = True
    f_lu = 0.320
    lamda_ul = 2382.7642
    gamma = 3.13e8
    z = 2

    W_rest_approx = calculate_W(10**20.3, f_lu, lamda_ul, 5, gamma, approx=approx)
    print(f"W_rest = {W_rest_approx:.3f} angstrom")

    b_list = np.array([1, 2, 3, 5, 10], dtype=float)
    N_list = 10 ** np.arange(12, 17, 0.2, dtype=float)
    plot_logW_vs_logN(N_list, f_lu, lamda_ul, b_list, gamma, approx=approx)

    # # Fe II 2249.9
    # approx = True
    # f_lu = 0.00182
    # lamda_ul = 2249.8768
    # gamma = 3.31e8
    # z = 2

    # W_rest_approx = calculate_W(10**20.3, f_lu, lamda_ul, 5, gamma, approx=approx)
    # print(f"W_rest = {W_rest_approx:.3f} angstrom")

    # b_list = np.array([1, 2, 3, 5, 10], dtype=float)
    # N_list = 10 ** np.arange(12, 17, 0.2, dtype=float)
    # plot_logW_vs_logN(N_list, f_lu, lamda_ul, b_list, gamma, approx=approx)

    # # C II 
    # approx = True
    # f_lu = 0.12780
    # lamda_ul = 1334.5323
    # gamma = 2.880e8
    # z = 2

    # W_rest_approx = calculate_W(10**20.3, f_lu, lamda_ul, 5, gamma, approx=approx)
    # print(f"W_rest = {W_rest_approx:.3f} angstrom")

    # b_list = np.array([1, 2, 3, 5, 10], dtype=float)
    # N_list = 10 ** np.arange(13, 18, 0.2, dtype=float)
    # plot_logW_vs_logN(N_list, f_lu, lamda_ul, b_list, gamma, approx=approx)








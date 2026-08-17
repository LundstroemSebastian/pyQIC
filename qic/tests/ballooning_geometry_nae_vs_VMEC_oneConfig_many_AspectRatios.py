"""
Test:
In this code we compute the relevant components for the ideal ballooning equation, by loading a qic configuration 
and using the function compute_ballooning_components_along_fl(stel, psi, alpha, varphi). 
 
In the end we would preferably also like to then store down the components and run the ideal ballooning solver with them.

Author: Sebastian Lundström
Created: 10.08.2026
Last modified: 13.08.2026
"""

# Imports:
import os
import numpy as np
import matplotlib.pyplot as plt

from qic import Qic
from simsopt.mhd import Vmec
from simsopt.mhd.vmec_diagnostics import vmec_fieldlines
import booz_xform as bx

from nae_configs_for_test import make_nae_stellarator
from function_defintions import compute_ballooning_coefficients_along_fl

_HERE = os.path.dirname(os.path.abspath(__file__))
_EQUILIBRIA = os.path.join(_HERE, '2nd_order_QI_basics', 'Equilibria')

# Create folders holding the VMEC runs  <-> And one with the corresponding name of the near-axis configuration
# (VMEC folder, near-axis configuration name)
CONFIGS = [
    ("QI_NFP2_Katia_std_buffer/",                           "QI NFP2 Katia"),
    ("QI_NFP2_Katia_smooth_buffer/",                        "QI NFP2 Katia smooth"),
    ("QI_NFP3_Katia_smooth_buffer_beta_shape/",             "QI NFP3 Katia smooth beta shape"),
    ("QI_NFP3_half_period_better_varphi_right_axis_extra/", "QI NFP3 half period better simp"),
    ("QI_opt_NFP3_Katia_smooth_buffer_hres/",               "QI NFP3 Katia opt"),
]

index = 2

config_folder, config_name = CONFIGS[index]
s     = 0.25                          # 0 = axis, 1 = boundary of THIS run
alpha = 0.0                           # field-line label
stel_config = make_nae_stellarator(config_name, nphi=501)

print('The configuration folder name:',config_folder)
print('\n')
print('The configuration name:', config_name)
aspects = []
all_errors = {'P_1': [], 'P_2': [], 'P_3': [],
              'Q_1': [], 'Q_2': [],
              'W_1': [], 'W_2': [], 'W_3': []}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 1.4,
    'lines.linewidth': 1.6,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.width': 1.4, 'ytick.major.width': 1.4,
    'xtick.major.size': 6, 'ytick.major.size': 6,
})

# ======================================================================
#  Choose the aspect ratio (i.e. which VMEC run)
# ======================================================================

# Their scan: 8 runs, boundaries at these radii
#N_r     = 8
#r_array = np.logspace(-2.5, -0.75, N_r)

# print("\nAvailable runs:")
runs = []
# for i in range(N_r):
#     path = os.path.join(_HERE, '..', 'Equilibria', config_folder,
#                         f'A_{i}', 'wout_temp.nc')
#     if not os.path.exists(path):
#         print(f"  A_{i}:  r_bdry = {r_array[i]:.5f}   (file missing)")
#         continue
#     try:
#         v = Vmec(path)
#         if v.wout.aspect > 1:
#             print(f"  A_{i}:  r_bdry = {r_array[i]:.5f}   A = {v.wout.aspect:7.2f}")
#             runs.append(i)
#         else:
#             print(f"  A_{i}:  r_bdry = {r_array[i]:.5f}   (did not converge)")
#     except Exception:
#         print(f"  A_{i}:  could not read")

print("\nAvailable runs:")
runs = []
for i in range(8):
    path = os.path.join(_EQUILIBRIA, config_folder, f'A_{i}', 'wout_temp.nc')
    if not os.path.exists(path):
        continue
    v = Vmec(path)
    if v.wout.aspect > 1:
        print(f"  A_{i}:   aspect ratio = {v.wout.aspect:8.2f}   "
              f"boundary at r = {v.wout.Aminor_p:.5f}")
        runs.append(i)
# --- pick one
#run_index = 1
for run_index in runs:
    # wout_file  = os.path.join(_EQUILIBRIA, config_folder,
    #                           f'A_{run_index}', 'wout_temp.nc')
    wout_file = os.path.join(_EQUILIBRIA, config_folder,
                            f'A_{run_index}', 'wout_temp.nc')
    v = Vmec(wout_file)
    r_boundary = Vmec(wout_file).wout.Aminor_p


    # ======================================================================
    #  Choose the surface and the field line
    # ======================================================================


    #r = r_boundary * np.sqrt(s)           # the physical radius of that surface
    # psi at the plasma boundary, from VMEC's toroidal flux
    # (v.wout.phi is the FLUX array, not an angle)
    psi_edge = v.wout.phi[-1] / (2*np.pi)

    # psi on the surface we want
    psi = s * psi_edge




    # ======================================================================
    #  Near-axis side
    # ======================================================================


    #psi = 0.5 * stel_config.Bbar * r**2

    nae = compute_ballooning_coefficients_along_fl(stel_config, psi, alpha,
                                                stel_config.varphi)
    # the near-axis flux label r corresponding to that psi
    r = np.sqrt(2*psi / stel_config.Bbar)
    print(f"\nrun A_{run_index}:  boundary at r = {r_boundary:.5f}")
    print(f"looking at s = {s}  ->  r = {r:.5f},  alpha = {alpha}")

    # ======================================================================
    #  VMEC side -- SAME physical points
    # ======================================================================
    # stel_config.varphi[i] and stel_config.phi[i] are the Boozer and
    # cylindrical angles of the SAME grid point i.  The near-axis function
    # was given varphi; simsopt wants phi.  Passing stel_config.phi here
    # means entry i of both results refers to the same place on the axis.

    #v  = Vmec(wout_file)
    fl = vmec_fieldlines(v, s=s, alpha=alpha, phi1d=stel_config.phi)
    bz = bx.Booz_xform()
    bz.read_boozmn(os.path.join(_EQUILIBRIA, config_folder,
                                f'A_{run_index}', 'boozmn_out.nc'))
    js = np.argmin(np.abs(np.array(bz.s_b) - s))
    G_booz    = bz.Boozer_G_all[js]
    I_booz    = bz.Boozer_I_all[js]
    iota_booz = bz.iota[js]

    def get(name):
        return getattr(fl, name)[0, 0, :]

    modB = get('modB')

    vmec = {
        'B0':                            modB,
        #'B_varphi_up':                   get('B_sup_phi'),
        # B^varphi in BOOZER phi = B^2/(G + iota I), not simsopt's cylindrical B_sup_phi
        #'B_varphi_up': modB**2 / (v.wout.bvco[1] + fl.iota[0]*v.wout.buco[1]), 
        'B_varphi_up': modB**2 / (G_booz + iota_booz*I_booz),
        'grad_psi_squared':              get('grad_psi_dot_grad_psi'),          
        'grad_alpha_squared':            get('grad_alpha_dot_grad_alpha'),      
        'grad_alpha_dot_grad_psi':       get('grad_alpha_dot_grad_psi'),
        # simsopt gives B x (...), you use b x (...) = (B x ...)/|B|
        'b_cross_grad_B_dot_grad_alpha': get('B_cross_grad_B_dot_grad_alpha') / modB,
        'b_cross_kappa_dot_grad_alpha':  get('B_cross_kappa_dot_grad_alpha') / modB,
        'b_cross_grad_B_dot_grad_psi':   get('B_cross_grad_B_dot_grad_psi') / modB,
        'b_cross_kappa_dot_grad_psi':    get('B_cross_kappa_dot_grad_psi')  / modB,
    }
    # vmec = {
    #     'B0':                            modB,
    #     'B_varphi_up':                   get('B_sup_phi'),
    #     'grad_psi_squared':              get('grad_psi_dot_grad_psi'),
    #     'grad_alpha_squared':            get('grad_alpha_dot_grad_alpha'),
    #     'grad_alpha_dot_grad_psi':       get('grad_alpha_dot_grad_psi'),
    #     # simsopt gives B x (...), you use b x (...) = (B x ...)/|B|
    #     'b_cross_grad_B_dot_grad_alpha': get('B_cross_grad_B_dot_grad_alpha') / modB,
    #     'b_cross_kappa_dot_grad_alpha':  get('B_cross_kappa_dot_grad_alpha')  / modB,
    #     'b_cross_grad_B_dot_grad_psi':   get('B_cross_grad_B_dot_grad_psi')   / modB,
    #     'b_cross_kappa_dot_grad_psi':    get('B_cross_kappa_dot_grad_psi')    / modB,
    # }


    # ======================================================================
    #  Assemble P, Q, W from either side with the same formulas
    # ======================================================================

    def make_PQW(d):
        B0  = d['B0']
        Bpu = d['B_varphi_up']
        return {
            'P_1': Bpu * d['grad_alpha_squared']      / B0**2,
            'P_2': Bpu * d['grad_alpha_dot_grad_psi'] / B0**2,
            'P_3': Bpu * d['grad_psi_squared']        / B0**2,
            'Q_1': d['b_cross_kappa_dot_grad_alpha'] / (Bpu * B0),
            'Q_2': d['b_cross_kappa_dot_grad_psi']   / (Bpu * B0),
            'W_1': d['grad_alpha_squared']      / (Bpu * B0**2),
            'W_2': d['grad_alpha_dot_grad_psi'] / (Bpu * B0**2),
            'W_3': d['grad_psi_squared']        / (Bpu * B0**2),
        }

    PQW_nae  = make_PQW(nae)
    PQW_vmec = make_PQW(vmec)
    aspects.append(v.wout.aspect)
    print(f"\nA = {v.wout.aspect:.2f}")
    for k in PQW_nae:
        a, b = PQW_nae[k], PQW_vmec[k]
        err = np.sqrt(np.sum((a-b)**2)) / np.sqrt(np.sum(b**2))
        all_errors[k].append(err)
        print(f"  {k}:  relative L2 error = {err:.3e}")
# for k in PQW_nae:
#     a, b = PQW_nae[k], PQW_vmec[k]
#     err = np.sqrt(np.sum((a-b)**2)) / np.sqrt(np.sum(b**2))
#     print(f"  {k}:  relative L2 error = {err:.3e}")

# ======================================================================
#  Diagnostic: does |B| agree?
# ======================================================================
# |B| is the same physical quantity in both codes, with no convention
# freedom.  At A = 150 the two should be nearly indistinguishable.
# If they are not, the problem is upstream of the coefficients:
# wrong configuration, wrong surface, or wrong field line.

# plt.figure(figsize=(7, 4))
# plt.plot(stel_config.varphi, nae['B0'],  label='near-axis')
# plt.plot(stel_config.varphi, vmec['B0'], '--', label='VMEC')
# plt.xlabel(r'$\varphi$')
# plt.ylabel(r'$|B|$')
# plt.legend()
# plt.title(f'{config_name},  A_{run_index},  s={s}')
# plt.tight_layout()
# plt.show()

# print("iota  =", stel_config.iota)
# print("iotaN =", stel_config.iotaN)
# print("N_helicity =", stel_config.N_helicity)
# print("VMEC iota at s:", np.interp(s, np.linspace(0,1,len(v.wout.iotas)), v.wout.iotas))

# print("iotaf[0] =", v.wout.iotaf[0])
# print("iotas[1] =", v.wout.iotas[1])
# print("simsopt fl.iota:", fl.iota)
# ======================================================================
#  Diagnostic 2: does the FIRST-ORDER part of |B| agree?
# ======================================================================
# The plot above only tested B0(varphi), which is the same on every
# field line -- so it cannot detect a mismatch in alpha.
# The first-order correction does depend on the field line, through chi.
# Comparing it is therefore a test of the field-line correspondence.

#iota_M = stel_config.iota - stel_config.N_helicity

# iota_M = stel_config.iota - stel_config.N_helicity
# chi    = alpha + iota_M * stel_config.varphi

# X1 = stel_config.X1c*np.cos(chi) + stel_config.X1s*np.sin(chi)

# # what the near-axis says the correction should be
# correction_nae  = r * stel_config.curvature * X1 * stel_config.B0

# # what VMEC has, after removing the on-axis field
# correction_vmec = vmec['B0'] - stel_config.B0

# plt.figure(figsize=(7, 4))
# plt.plot(stel_config.varphi, correction_nae,  label='near-axis  $r \\kappa X_1 B_0$')
# plt.plot(stel_config.varphi, correction_vmec, '--', label='VMEC  $|B| - B_0$')
# plt.xlabel(r'$\varphi$')
# plt.ylabel(r'$|B| - B_0$')
# plt.legend()
# plt.title('first-order part of |B| -- tests the field line')
# plt.tight_layout()
# plt.show()

# print("pyQIC:   sG =", stel_config.sG, "  spsi =", stel_config.spsi)
# print("pyQIC:   N_helicity =", stel_config.N_helicity)
# print("simsopt: toroidal_flux_sign =", fl.toroidal_flux_sign)

# plt.figure(figsize=(7,4))
# plt.plot(stel_config.varphi, stel_config.curvature, label='kappa')
# plt.plot(stel_config.varphi, stel_config.X1c, label='X1c')
# plt.axhline(0, color='k', lw=0.5)
# plt.legend()
# plt.xlabel('varphi')
# plt.show()

# print("\n--- grad_psi_squared ---")
# print("NAE :", nae['grad_psi_squared'][:4])
# print("VMEC:", vmec['grad_psi_squared'][:4])
# print("ratio:", vmec['grad_psi_squared'][:4] / nae['grad_psi_squared'][:4])

# print("\n--- grad_alpha_squared ---")
# print("NAE :", nae['grad_alpha_squared'][:4])
# print("VMEC:", vmec['grad_alpha_squared'][:4])
# print("ratio:", vmec['grad_alpha_squared'][:4] / nae['grad_alpha_squared'][:4])

# print("\n--- B_varphi_up ---")
# print("NAE :", nae['B_varphi_up'][:4])
# print("VMEC:", vmec['B_varphi_up'][:4])
# print("ratio:", vmec['B_varphi_up'][:4] / nae['B_varphi_up'][:4])

# print("pyQIC G0 :", stel_config.G0)
# print("VMEC bvco:", v.wout.bvco[1])
# print("VMEC buco:", v.wout.buco[1])

# psi_edge_vmec = v.wout.phi[-1] / (2*np.pi)
# psi_edge_nae  = 0.5 * stel_config.Bbar * r_boundary**2
# print("VMEC psi_edge:", psi_edge_vmec)
# print("NAE  psi_edge:", psi_edge_nae)
# print("ratio:", psi_edge_vmec / psi_edge_nae)

# print("B_sup_phi        :", get('B_sup_phi')[:3])
# print("B^2/(G+iota*I)   :", (modB**2 / (v.wout.bvco[1] + fl.iota*v.wout.buco[1]))[:3])
# print("L_reference      :", fl.L_reference)
# print("B_reference      :", fl.B_reference)
# ======================================================================
#  Plot: near-axis vs VMEC
# ======================================================================

aspects = np.array(aspects)

fig, ax = plt.subplots(figsize=(7, 5))

for k in all_errors:
    ax.loglog(aspects, all_errors[k], 'o-', label=k, markersize=4)

# first-order near-axis should converge like 1/A
ref = np.array([aspects.min(), aspects.max()])
ax.loglog(ref, all_errors['P_1'][-1] * ref[0] / ref, 'k--', lw=1, label='$1/A$')

ax.set_xlabel('aspect ratio $A$')
ax.set_ylabel('relative $L^2$ error')
ax.set_title(f'{config_name},  s={s},  alpha={alpha}')
ax.legend(fontsize=8)
ax.grid(alpha=0.2, which='both')
fig.tight_layout()
plt.show()

# varphi = stel_config.varphi

# # where the axis curvature vanishes -- the drive switches off there
# zeros = varphi[np.where(np.diff(np.sign(stel_config.curvature)))[0]]


# def compare_group(keys, title, filename):
#     fig, ax = plt.subplots(len(keys), 1, sharex=True,
#                            figsize=(6.5, 2.1*len(keys)))
#     ax = np.atleast_1d(ax)
#     for a, k in zip(ax, keys):
#         a.plot(varphi, PQW_nae[k],  label='near-axis')
#         a.plot(varphi, PQW_vmec[k], '--', label='VMEC')
#         a.set_ylabel(k)
#         a.axhline(0, color='k', lw=0.5, alpha=0.5)
#         for z in zeros:
#             a.axvline(z, color='0.6', ls=':', lw=1)
#     ax[0].legend(fontsize=9)
#     ax[-1].set_xlabel(r'$\varphi$')
#     fig.suptitle(f'{title}\n{config_name},  A_{run_index},  s={s},  alpha={alpha}',
#                  fontsize=10)
#     fig.tight_layout()
#     fig.savefig(filename)


# compare_group(['P_1', 'P_2', 'P_3'], 'Principal coefficient', 'cmp_P.pdf')
# compare_group(['Q_1', 'Q_2'],        'Potential',             'cmp_Q.pdf')
# compare_group(['W_1', 'W_2', 'W_3'], 'Weight function',       'cmp_W.pdf')

# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmin, argrelmax
from scipy.stats import gaussian_kde

#  Quasi-potential 
def quasi_potential(data_flat, x_grid):
    kde     = gaussian_kde(data_flat, bw_method='silverman')
    density = kde(x_grid)
    U       = -np.log(density + 1e-10)
    U      -= U.min()
    return U

#  Dominance angle analysis 
def compute_dominance_angle(x_grid, U, label_A='Pop A', label_B='Pop B',
                             order=15, verbose=True):
    """
    Computes the tilt angle of the potential landscape between two wells.

    Returns a dict with:
      - angle_deg     : tilt angle in degrees (signed)
      - dominant_pop  : which population dominates
      - U_A, U_B      : well depths
      - delta_U       : U_B - U_A  (positive = A dominates)
      - barrier_height: height of barrier above the lower well
    """
    minima_idx = argrelmin(U, order=order)[0]
    maxima_idx = argrelmax(U, order=order)[0]

    if len(minima_idx) < 2:
        if verbose:
            print(" Less than 2 minima found — landscape may be monostable.")
        return None

    # Take the two deepest minima as the two wells
    sorted_minima = minima_idx[np.argsort(U[minima_idx])]
    idx_A, idx_B  = sorted(sorted_minima[:2])  # left=A, right=B by position

    x_A, U_A = x_grid[idx_A], U[idx_A]
    x_B, U_B = x_grid[idx_B], U[idx_B]

    # Barrier: highest maximum between the two minima
    between_mask = (maxima_idx > idx_A) & (maxima_idx < idx_B)
    if between_mask.sum() == 0:
        if verbose:
            print(" No barrier found between the two minima.")
        return None

    idx_barrier    = maxima_idx[between_mask][np.argmax(U[maxima_idx[between_mask]])]
    x_bar, U_bar   = x_grid[idx_barrier], U[idx_barrier]

    #  Tilt angle 
    # Angle of the line connecting well A to well B in (x, U) space
    # Positive angle → B is higher than A → A dominates (lower = more stable)
    # Negative angle → A is higher than B → B dominates
    delta_x   = x_B - x_A          # always positive (B is to the right)
    delta_U   = U_B - U_A          # sign carries dominance direction
    angle_rad = np.arctan2(delta_U, delta_x)
    angle_deg = np.degrees(angle_rad)

    # Barrier heights above each well
    barrier_above_A = U_bar - U_A  # energy to escape from A → transition rate A→B
    barrier_above_B = U_bar - U_B  # energy to escape from B → transition rate B→A

    # Dominant population = deeper well (lower U)
    if U_A < U_B:
        dominant   = label_A
        subdominant = label_B
    else:
        dominant   = label_B
        subdominant = label_A

    results = dict(
        x_A=x_A, U_A=U_A,
        x_B=x_B, U_B=U_B,
        x_bar=x_bar, U_bar=U_bar,
        delta_U=delta_U,
        angle_deg=angle_deg,
        dominant=dominant,
        subdominant=subdominant,
        barrier_above_A=barrier_above_A,
        barrier_above_B=barrier_above_B,
    )

    if verbose:
        print(f"  Well {label_A}      : x={x_A:.3f},  U={U_A:.3f}")
        print(f"  Well {label_B}      : x={x_B:.3f},  U={U_B:.3f}")
        print(f"  Barrier            : x={x_bar:.3f}, U={U_bar:.3f}")
        print(f"  ΔU (B−A)           : {delta_U:+.3f}")
        print(f"  Tilt angle         : {angle_deg:+.2f}°")
        print(f"  Barrier height A→B : {barrier_above_A:.3f}  (escape from {label_A})")
        print(f"  Barrier height B→A : {barrier_above_B:.3f}  (escape from {label_B})")
        print(f"  ► Dominant state   : {dominant}")
        print()

    return results

#  Visualization 
def plot_dominance(x_grid, U, res, title='', label_A='Pop A', label_B='Pop B',
                   color='purple', ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x_grid, U, color=color, linewidth=2.5, zorder=3)
    ax.fill_between(x_grid, U, U.max(), alpha=0.08, color=color)

    if res is None:
        ax.set_title(f'{title}\n(monostable — no angle computed)')
        return ax

    # Well markers
    ax.scatter([res['x_A']], [res['U_A']], color='steelblue', s=100,
               zorder=5, edgecolors='k', linewidths=0.8, label=f'{label_A} well')
    ax.scatter([res['x_B']], [res['U_B']], color='tomato',    s=100,
               zorder=5, edgecolors='k', linewidths=0.8, label=f'{label_B} well')
    ax.scatter([res['x_bar']], [res['U_bar']], color='gold',  s=100,
               marker='^', zorder=5, edgecolors='k', linewidths=0.8, label='Barrier')

    # Tilt arrow connecting well A to well B
    ax.annotate('', xy=(res['x_B'], res['U_B']),
                 xytext=(res['x_A'], res['U_A']),
                 arrowprops=dict(arrowstyle='->', color='black',
                                 lw=2, mutation_scale=18))

    # Barrier height brackets
    for x_w, U_w, side_label, c in [
        (res['x_A'], res['U_A'], f"ΔU_A→B\n{res['barrier_above_A']:.2f}", 'steelblue'),
        (res['x_B'], res['U_B'], f"ΔU_B→A\n{res['barrier_above_B']:.2f}", 'tomato'),
    ]:
        ax.annotate('', xy=(x_w, res['U_bar']), xytext=(x_w, U_w),
                    arrowprops=dict(arrowstyle='<->', color=c, lw=1.5))
        ax.text(x_w + (res['x_bar'] - x_w) * 0.18, (U_w + res['U_bar']) / 2,
                side_label, color=c, fontsize=8, ha='center', va='center')

    # Angle annotation
    angle_txt = (f"θ = {res['angle_deg']:+.1f}°\n"
                 f"► {res['dominant']} dominates")
    props = dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                 edgecolor='gray', alpha=0.9)
    ax.text(0.97, 0.95, angle_txt, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='right', bbox=props)

    ax.set_xlabel('Signal value', fontsize=10)
    ax.set_ylabel('U = −log P',   fontsize=10)
    ax.set_title(title,           fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)

    return ax

#  Run it 
x_grid = np.linspace(X_all.min(), X_all.max(), 500)
U_all  = quasi_potential(X_all, x_grid)
U_g1   = quasi_potential(X_g1,  x_grid)
U_g2   = quasi_potential(X_g2,  x_grid)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, U, title, color in zip(
    axes,
    [U_all,          U_g1,         U_g2],
    ['All data',     'Group 1',    'Group 2'],
    ['purple',       'steelblue',  'tomato']
):
    print(f"── {title} ──")
    res = compute_dominance_angle(x_grid, U, label_A='Pop A', label_B='Pop B')
    plot_dominance(x_grid, U, res, title=title, color=color, ax=ax)

plt.suptitle('Quasi-potential landscape — dominance angle analysis',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#  Multi-condition summary: angle across all samples 
print("\n── Dominance angle summary across conditions ──")
print(f"{'Condition':<15} {'Sample':<8} {'Angle (°)':<12} {'Dominant':<12} {'ΔU':<10}")
print("-" * 57)

for condition, results_list in all_results.items():
    for s_idx, res_sample in enumerate(results_list):
        # Recompute on the stored potentials
        x_g   = res_sample['x_grid']
        for U_key, grp_label in [('U_all', 'all'), ('U_g1', 'g1'), ('U_g2', 'g2')]:
            U_s = res_sample[U_key]
            r   = compute_dominance_angle(x_g, U_s,
                                          label_A='Pop A', label_B='Pop B',
                                          verbose=False)
            if r:
                print(f"{condition:<15} {s_idx+1:<8} "
                      f"{r['angle_deg']:>+8.1f}°   "
                      f"{r['dominant']:<12} "
                      f"{r['delta_U']:>+.3f}")

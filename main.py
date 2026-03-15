import numpy as np
from scipy.linalg import sqrtm as scipy_sqrtm
import time
import os
import warnings
from components import *

warnings.filterwarnings('ignore')
np.random.seed(42)

I4  = np.eye(4,  dtype=complex)   
I16 = np.eye(16, dtype=complex)   



def build_spdc_state():
    ph_H = np.zeros(4, dtype=complex); ph_H[0] = 1.0   
    ph_V = np.zeros(4, dtype=complex); ph_V[1] = 1.0   
    phi_plus = (np.kron(ph_H, ph_H) + np.kron(ph_V, ph_V)) / np.sqrt(2)
    return np.outer(phi_plus, phi_plus.conj())

def expand_to_16(U_4x4, photon_idx):
    if photon_idx == 0: return np.kron(U_4x4, I4)
    else:               return np.kron(I4, U_4x4)

def expand_to_64(U_4x4, photon_idx):
    if photon_idx == 0:   return np.kron(np.kron(U_4x4, I4), I4)
    elif photon_idx == 1: return np.kron(np.kron(I4, U_4x4), I4)
    elif photon_idx == 2: return np.kron(np.kron(I4, I4), U_4x4)

def partial_trace_16(rho16, keep):
    T = rho16.reshape(4, 4, 4, 4)
    if keep == 0: return np.einsum('ikjk->ij', T)
    else:         return np.einsum('kikj->ij', T)

def trace_out_spatial(rho4):
    T = rho4.reshape(2, 2, 2, 2)
    return np.einsum('kikj->ij', T)

def add_ancilla_vacuum(rho_16):
    ph_H = np.zeros(4, dtype=complex); ph_H[0] = 1.0  
    rho_ancilla = np.outer(ph_H, ph_H.conj())
    return np.kron(rho_16, rho_ancilla)

def trace_out_ancilla(rho_64):
    T = rho_64.reshape(16, 4, 16, 4)
    return np.einsum('ikjk->ij', T)

def simulate(circuit, rho_in, use_ancilla=False):
    if use_ancilla:
        rho_current = add_ancilla_vacuum(rho_in)
        U_total = np.eye(64, dtype=complex)
        
        for (name, param, ph) in circuit:
            if name == 'CK':
                ph1 = ph % 3
                ph2 = (ph + 1) % 3
                U_total = get_CK_64(param, ph1, ph2) @ U_total
            else:
                ph = ph % 3 
                U4 = COMP_FUNCS[name](param) if NEEDS_PARAM[name] else COMP_FUNCS[name]()
                U_total = expand_to_64(U4, ph) @ U_total
                
        rho_out_64 = U_total @ rho_current @ U_total.conj().T
        return trace_out_ancilla(rho_out_64)
        
    else:
        rho_current = rho_in
        U_total = I16.copy()
        
        for (name, param, ph) in circuit:
            if name == 'CK':
                U_total = get_CK_16(param, 0, 1) @ U_total
            else:
                ph = ph % 2 
                U4 = COMP_FUNCS[name](param) if NEEDS_PARAM[name] else COMP_FUNCS[name]()
                U_total = expand_to_16(U4, ph) @ U_total
                
        return U_total @ rho_current @ U_total.conj().T


def is_pure(rho, tol=1e-6):
    return abs(np.real(np.trace(rho @ rho)) - 1.0) < tol

def compute_fidelity(rho_target, rho_out):
    if is_pure(rho_target):
        F = np.real(np.trace(rho_target @ rho_out))
        return float(np.clip(F, 0.0, 1.0))

    sqrt_target = scipy_sqrtm(rho_target)
    M           = sqrt_target @ rho_out @ sqrt_target
    F           = np.real(np.trace(scipy_sqrtm(M))) ** 2
    return float(np.clip(F, 0.0, 1.0))

def score_circuit(circuit, rho_target, mode, rho_in):
    target_is_mixed = not is_pure(rho_target)
    rho_out = simulate(circuit, rho_in, use_ancilla=target_is_mixed)

    if   mode == '16x16':   rho_cmp = rho_out
    elif mode == 'ptrace0': rho_cmp = partial_trace_16(rho_out, 0)
    elif mode == 'ptrace1': rho_cmp = partial_trace_16(rho_out, 1)
    elif mode == 'pol0':    rho_cmp = trace_out_spatial(partial_trace_16(rho_out, 0))
    elif mode == 'pol1':    rho_cmp = trace_out_spatial(partial_trace_16(rho_out, 1))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return compute_fidelity(rho_target, rho_cmp)



def random_gene():
    name  = np.random.choice(COMP_NAMES)
    param = np.random.uniform(0, 2 * np.pi) if NEEDS_PARAM[name] else 0.0
    ph    = np.random.randint(0, 3) 
    return (name, param, ph)

def init_population(pop_size, circuit_len):
    return [[random_gene() for _ in range(circuit_len)] for _ in range(pop_size)]

def tournament_select(population, fitnesses, k=3):
    idx  = np.random.choice(len(population), k, replace=False)
    best = idx[np.argmax([fitnesses[i] for i in idx])]
    return population[best]

def crossover(parent1, parent2):
    n  = min(len(parent1), len(parent2))
    pt = np.random.randint(1, n)
    return parent1[:pt] + parent2[pt:], parent2[:pt] + parent1[pt:]

def mutate(individual, mut_rate=0.25, param_noise=0.4):
    mutated = []
    for (name, param, ph) in individual:
        if np.random.random() < mut_rate:
            mutated.append(random_gene())
        else:
            if NEEDS_PARAM[name] and np.random.random() < 0.35:
                param = (param + np.random.normal(0, param_noise)) % (2 * np.pi)
            mutated.append((name, param, ph))
    return mutated

def run_GA(rho_target, mode='16x16', circuit_len=14, pop_size=120, n_generations=400, cx_rate=0.7, mut_rate=0.25, elite_frac=0.1, target_fid=0.99, verbose=True):
    rho_in  = build_spdc_state()
    n_elite = max(2, int(pop_size * elite_frac))
    pop     = init_population(pop_size, circuit_len)
    best    = None
    best_f  = 0.0
    hist    = []
    t0      = time.time()

    if verbose:
        print(f"\n{'='*62}")
        print(f"  GENETIC ALGORITHM SEARCH")
        print(f"{'='*62}")
        print(f"  Population : {pop_size}   |   Generations : {n_generations}")
        print(f"  Circuit L  : {circuit_len}    |   Mode         : {mode}")

    for gen in range(n_generations):
        fits = []
        raw_fidelities = []
        
        for ind in pop:
            fid = score_circuit(ind, rho_target, mode, rho_in)
            active_count = sum(1 for gene in ind if gene[0] != 'NONE')
            adjusted_score = fid - (active_count * 0.00001)
            fits.append(adjusted_score)
            raw_fidelities.append(fid)

        gi = int(np.argmax(fits))
        if raw_fidelities[gi] >= best_f:
            best_f = raw_fidelities[gi]
            best   = pop[gi][:]
        hist.append(best_f)

        if verbose and (gen + 1) % 20 == 0:
            print(f"  Gen {gen+1:4d}/{n_generations}  |  Best F = {best_f:.5f}  |  Avg F = {np.mean(fits):.5f}  |  {time.time()-t0:.1f}s")

        if best_f >= target_fid:
            if verbose: print(f"\n  Target fidelity reached at generation {gen+1}!")
            break

        ranked  = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
        new_pop = [pop[i][:] for i in ranked[:n_elite]]

        while len(new_pop) < pop_size:
            p1   = tournament_select(pop, fits)
            p2   = tournament_select(pop, fits)
            c1, c2 = crossover(p1, p2) if np.random.random() < cx_rate else (p1[:], p2[:])
            new_pop.append(mutate(c1, mut_rate))
            if len(new_pop) < pop_size:
                new_pop.append(mutate(c2, mut_rate))

        pop = new_pop

    if verbose:
        print(f"\n  Best fidelity : {best_f:.6f}")
        print(f"  Time elapsed  : {time.time()-t0:.2f}s")

    return best, best_f, hist



def print_circuit(circuit, fidelity_val):
    active_circuit = [(n, p, ph) for (n, p, ph) in circuit if n != 'NONE']
    
    print(f"\n{'='*62}")
    print(f"  OPTIMAL CIRCUIT   (Fidelity = {fidelity_val:.6f})")
    print(f"  Used {len(active_circuit)} out of max {len(circuit)} allowed slots")
    print(f"{'='*62}")
    print(f"  {'Step':<5}  {'Component':<8}  {'Parameter (rad)':<22}  {'Acts on'}")
    print(f"  {'-'*55}")
    
    step_num = 1
    for (name, param, ph) in active_circuit:
        p_str = f"{param:.5f} rad" if NEEDS_PARAM[name] else "  —  (fixed)"
        if name == 'CK': acts_on_str = f"photons {ph} & {(ph+1)%3}"
        else:            acts_on_str = f"photon {ph}"
            
        print(f"  {step_num:<5}  {name:<8}  {p_str:<22}  {acts_on_str}")
        step_num += 1

    print()
    if   fidelity_val >= 0.99: print("  ✓  SUCCESS  — fidelity >= 0.99")
    elif fidelity_val >= 0.90: print("  ~  CLOSE   — try more generations or larger circuit")
    else:                      print("  ✗  NOT YET — increase pop_size, n_generations, or max circuit length")

def print_comparison(rho_target, rho_achieved):
    print(f"\n  Final fidelity: {compute_fidelity(rho_target, rho_achieved):.6f}")

def plot_convergence(hist):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 4))
        plt.plot(hist, lw=1.8, color='royalblue', label='Best fidelity')
        plt.axhline(0.99, color='green', lw=1.2, ls='--', alpha=0.7, label='Target (0.99)')
        plt.xlabel('Generation')
        plt.ylabel('Fidelity')
        plt.title('GA Convergence — Fidelity vs Generation')
        plt.ylim(0, 1.08)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception:
        pass



def make_presets():
    ph_H = np.zeros(4, dtype=complex); ph_H[0] = 1.0  
    ph_V = np.zeros(4, dtype=complex); ph_V[1] = 1.0  

    HH = np.kron(ph_H, ph_H)
    VV = np.kron(ph_V, ph_V)
    HV = np.kron(ph_H, ph_V)
    VH = np.kron(ph_V, ph_H)

    def dm(psi):
        return np.outer(psi, psi.conj())

    I_16 = np.eye(16, dtype=complex)
    classical_mix = 0.5 * dm(HH) + 0.5 * dm(VV)
    phi_plus = (HH + VV) / np.sqrt(2)
    werner_state = 0.8 * dm(phi_plus) + 0.2 * (I_16 / 16.0)

    return {
        '1': (dm((HH + VV) / np.sqrt(2)),  "|Φ+⟩  Bell state (starting state)"),
        '2': (dm((HH - VV) / np.sqrt(2)),  "|Φ-⟩  Bell state"),
        '3': (dm((HV + VH) / np.sqrt(2)),  "|Ψ+⟩  Bell state"),
        '4': (dm((HV - VH) / np.sqrt(2)),  "|Ψ-⟩  Bell state"),
        '5': (classical_mix, "Classical Mixture (50% HH, 50% VV)"),
        '6': (werner_state, "Werner State (80% Phi+, 20% Noise)"),
    }

def normalize_density_matrix(rho):
    rho = (rho + rho.conj().T) / 2
    tr = np.trace(rho)
    if abs(tr) > 1e-12:
        rho = rho / tr
    return rho

def input_custom_matrix(dim, mode):
    print(f"\n  How would you like to input the {dim}x{dim} matrix?")
    print("    1. Load from a .npy file (Recommended)")
    print("    2. Enter manually row-by-row")
    
    choice = input("  Choice (1-2): ").strip()
    
    if choice == '1':
        filepath = input(f"  Enter filename (e.g., target_{dim}.npy): ").strip()
        if os.path.exists(filepath):
            try:
                rho = np.load(filepath)
                if rho.shape != (dim, dim):
                    print(f"  Error: Expected shape ({dim}, {dim}), but got {rho.shape}.")
                    return None
                rho = normalize_density_matrix(rho)
                print(f"  Successfully loaded {filepath}!")
                return rho, f"Loaded {dim}x{dim} from {filepath}", mode
            except Exception as e:
                print(f"  Error reading file: {e}")
        else:
            print(f"  File not found: {filepath}. Make sure it is uploaded!")
        return None
        
    elif choice == '2':
        print(f"\n  Enter {dim}x{dim} density matrix (one row at a time)")
        print("  Format per row: val1 val2 ... (use j for imaginary, e.g. 0.5+0j)")
        rows = []
        for i in range(dim):
            while True:
                try:
                    row_input = input(f"  Row {i+1}: ").strip().split()
                    if len(row_input) != dim:
                        print(f"  Please enter exactly {dim} values.")
                        continue
                    row = [complex(x) for x in row_input]
                    rows.append(row)
                    break
                except Exception as e:
                    print(f"  Invalid input: {e}. Please try again.")
        rho = np.array(rows, dtype=complex)
        rho = normalize_density_matrix(rho)
        return rho, f"Manual {dim}x{dim} Matrix", mode
        
    else:
        print("  Invalid choice.")
        return None

def get_target():
    presets = make_presets()

    print("\n" + "="*62)
    print("  QUANTUM OPTICS VIRTUAL LABORATORY")
    print("  AI-Powered Experiment Designer")
    print("="*62)
    print("\n  Select target state category:")
    print("    1. Built-in 16x16 Presets (Bell States, Werner, etc.)")
    print("    2. Custom 16x16 Matrix (Full 2-Photon State)")
    print("    3. Custom 4x4 Matrix (Single Photon - Spatial & Pol)")
    print("    4. Custom 2x2 Matrix (Single Photon - Polarization Only)")

    cat = input("\n  Your choice (1-4): ").strip()

    if cat == '1':
        print("\n  Select a preset:")
        for k, (_, label) in presets.items():
            print(f"    {k}. {label}")
        p_choice = input("\n  Choice: ").strip()
        if p_choice in presets:
            rho, label = presets[p_choice]
            return rho, label, '16x16'
            
    elif cat == '2':
        res = input_custom_matrix(16, '16x16')
        if res: return res
        
    elif cat == '3':
        res = input_custom_matrix(4, 'ptrace0')
        if res: return res
        
    elif cat == '4':
        res = input_custom_matrix(2, 'pol0')
        if res: return res

    print("\n  Falling back to default Bell state Φ- (16x16)")
    rho, label = presets['2']
    return rho, label, '16x16'


if __name__ == "__main__":
    rho_target, target_label, mode = get_target()

    print(f"\n  Target state  : {target_label}")
    print(f"  Matrix size   : {rho_target.shape[0]}x{rho_target.shape[1]}")
    print(f"  Mode          : {mode}")
    print(f"  Pure state?   : {is_pure(rho_target)}")

    print("\n  GA settings (press Enter for defaults):")
    inp = input("  Circuit length  [default 10] : ").strip()
    circuit_len = int(inp) if inp.isdigit() else 10

    inp = input("  Population size [default 80] : ").strip()
    pop_size = int(inp) if inp.isdigit() else 80

    inp = input("  Generations     [default 200]: ").strip()
    n_gen = int(inp) if inp.isdigit() else 200

    best_circuit, best_fidelity, history = run_GA(
        rho_target    = rho_target,
        mode          = mode,
        circuit_len   = circuit_len,
        pop_size      = pop_size,
        n_generations = n_gen,
        target_fid    = 0.99,
        verbose       = True,
    )

    rho_in = build_spdc_state()
    target_is_mixed = not is_pure(rho_target)
    rho_out_16 = simulate(best_circuit, rho_in, use_ancilla=target_is_mixed)

    if   mode == '16x16':   rho_achieved = rho_out_16
    elif mode == 'ptrace0': rho_achieved = partial_trace_16(rho_out_16, 0)
    elif mode == 'ptrace1': rho_achieved = partial_trace_16(rho_out_16, 1)
    elif mode == 'pol0':    rho_achieved = trace_out_spatial(partial_trace_16(rho_out_16, 0))
    elif mode == 'pol1':    rho_achieved = trace_out_spatial(partial_trace_16(rho_out_16, 1))

    print_circuit(best_circuit, best_fidelity)
    print_comparison(rho_target, rho_achieved)
    plot_convergence(history)
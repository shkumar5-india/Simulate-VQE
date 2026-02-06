import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from qiskit import transpile
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

# --- UI CONFIG ---
st.set_page_config(page_title="QUANTUM VQE LAB", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1e293b; text-align: center; margin-bottom: 30px; }
    .metric-card {
        background: white; padding: 20px; border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; text-align: center;
    }
    .spec-label { color: #64748b; font-size: 0.85rem; text-transform: uppercase; font-weight: 600; margin-bottom: 5px; }
    .spec-value { color: #0f172a; font-size: 1.5rem; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">QUANTUM MOLECULAR SOLVER [VQE ENGINE]</div>', unsafe_allow_html=True)

def execute_vqe_engine(mol_type):
    # System Configuration
    if mol_type == "H2":
        d_range = np.arange(0.5, 2.1, 0.3)
        mapper = ParityMapper(num_particles=(1, 1))
        mapper_name = "ParityMapper (2-Qubit Reduction)"
        atom_symbol = "H"
        use_active_space = False
    else:  # LiH
        d_range = [1.2, 1.4, 1.6, 1.8, 2.0]
        mapper = JordanWignerMapper()
        mapper_name = "JordanWignerMapper"
        atom_symbol = "Li"
        use_active_space = True

    all_dist, all_energ = [], []
    best_overall_energy = float("inf")
    best_conv_history = []
    best_ansatz = None
    
    status = st.status(f"Initializing VQE for {mol_type}...", expanded=True)

    for d in d_range:
        status.update(label=f"Solving Hamiltonian at: {d:.3f} Å", state="running")
        
        # 1. Driver Setup
        atom_str = f"{atom_symbol} 0 0 0; H 0 0 {d}"
        driver = PySCFDriver(atom=atom_str, basis="sto3g")
        problem = driver.run()
        
        # 2. Energy Offset Logic
        # Qiskit Nature 0.7+ stores constant shifts (Nuclear + Frozen Core) in hamiltonian.constants
        hamiltonian_ops = problem.hamiltonian.second_q_op()
        
        if use_active_space:
            transformer = ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=2)
            problem = transformer.transform(problem)
            hamiltonian_ops = problem.hamiltonian.second_q_op()

        # Extract total constant shift (Nuclear Repulsion + Core Energy)
        energy_shift = sum(problem.hamiltonian.constants.values())

        # 3. Circuit Construction
        ansatz = UCCSD(
            problem.num_spatial_orbitals, 
            problem.num_particles, 
            mapper,
            initial_state=HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
        )

        # 4. VQE Execution
        temp_history = []
        def callback(count, params, mean, std=None): 
            temp_history.append(mean + energy_shift)

        vqe = VQE(Estimator(), ansatz, SLSQP(maxiter=100), callback=callback)
        qubit_op = mapper.map(hamiltonian_ops)
        result = vqe.compute_minimum_eigenvalue(qubit_op)

        total_energy = float(result.eigenvalue.real) + energy_shift
        all_dist.append(d)
        all_energ.append(total_energy)

        # Keep track of the best result for the circuit drawing/convergence plot
        if total_energy < best_overall_energy:
            best_overall_energy = total_energy
            best_conv_history = temp_history
            best_ansatz = ansatz

    status.update(label="VQE Simulation Complete!", state="complete", expanded=False)
    
    # Transpilation for hardware metrics
    hw_circ = transpile(best_ansatz, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)
    
    return all_dist, all_energ, best_conv_history, all_dist[np.argmin(all_energ)], best_overall_energy, hw_circ, mapper_name

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Simulation Settings")
    mol_choice = st.selectbox("Select Molecule", ["H2", "LiH"])
    st.info("H2 uses Sto-3g with Parity Mapping. LiH uses Active Space (2,2) to reduce qubit requirements.")

run_btn = st.button("RUN QUANTUM ANALYSIS", use_container_width=True, type="primary")

if run_btn:
    d, e, conv, b_dist, m_e, circ, m_name = execute_vqe_engine(mol_choice)
    
    # --- METRIC CARDS ---
    st.markdown("### 📊 KEY PERFORMANCE INDICATORS")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-card"><p class="spec-label">Optimal Distance</p><p class="spec-value">{b_dist:.3f} Å</p></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><p class="spec-label">Ground State Energy</p><p class="spec-value">{m_e:.5f} Ha</p></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><p class="spec-label">Gate Depth</p><p class="spec-value">{circ.depth()}</p></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><p class="spec-label">Qubits</p><p class="spec-value">{circ.num_qubits}</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # --- CHARTS ---
    g1, g2 = st.columns(2)
    with g1:
        fig1, ax1 = plt.subplots()
        ax1.plot(d, e, 'o-', color='#2563eb', linewidth=2, markersize=8)
        ax1.axvline(b_dist, color='orange', linestyle='--', alpha=0.6)
        ax1.set_title("Potential Energy Surface (PES)", fontweight='bold')
        ax1.set_xlabel("Interatomic Distance (Å)")
        ax1.set_ylabel("Total Energy (Hartree)")
        ax1.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig1)
        
    with g2:
        fig2, ax2 = plt.subplots()
        ax2.plot(conv, color='#10b981', linewidth=2)
        ax2.set_title("Optimizer Convergence @ Equilibrium", fontweight='bold')
        ax2.set_xlabel("Iteration Step")
        ax2.set_ylabel("Energy (Hartree)")
        ax2.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig2)

    # --- TECHNICAL BREAKDOWN ---
    st.markdown("### 🛠 HARDWARE DIAGNOSTICS")
    t1, t2 = st.columns([1, 1])
    
    with t1:
        st.markdown("**Algorithm Specifications**")
        diag_df = pd.DataFrame({
            "Component": ["Mapping Strategy", "Ansatz Type", "Optimizer", "Basis Set"],
            "Configuration": [m_name, "UCCSD", "SLSQP", "STO-3G"]
        })
        st.table(diag_df)
        
    with t2:
        st.markdown("**Gate Operations (Transpiled)**")
        ops = circ.count_ops()
        gate_df = pd.DataFrame({"Gate": list(ops.keys()), "Count": list(ops.values())})
        st.table(gate_df)

    with st.expander("🔬 VIEW QUANTUM CIRCUIT ARCHITECTURE"):
        st.markdown("Visualizing the transpiled circuit optimized for hardware-native gates:")
        fig_circ = circ.draw('mpl', scale=0.7)
        st.pyplot(fig_circ)

else:
    st.info("Click the button above to begin the Quantum Variational simulation.")


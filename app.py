import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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
    .spec-label { color: #64748b; font-size: 0.85rem; text-transform: uppercase; font-weight: 600; }
    .spec-value { color: #0f172a; font-size: 1.5rem; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">QUANTUM MOLECULAR SOLVER [VQE ENGINE]</div>', unsafe_allow_html=True)

def execute_vqe_engine(mol_type):
    if mol_type == "H2":
        d_range = np.arange(0.5, 1.1, 0.1)
        mapper = ParityMapper(num_particles=(1, 1))
        mapper_name = "ParityMapper (2-Qubit Reduction)"
        atom_symbol = "H"
        use_active_space = False
    else:
        d_range = [1.2, 1.4, 1.595, 1.8, 2.0]
        mapper = JordanWignerMapper()
        mapper_name = "JordanWignerMapper"
        atom_symbol = "Li"
        use_active_space = True

    all_dist, all_energ = [], []
    best_energy = float("inf")
    best_conv = []
    best_ansatz = None

    status = st.status(f"Executing Real VQE for {mol_type}...", expanded=True)

    for d in d_range:
        status.update(label=f"Solving Hamiltonian at: {d:.3f} Å", state="running")
        driver = PySCFDriver(atom=f"{atom_symbol} 0 0 0; H 0 0 {d}", basis="sto3g")
        problem = driver.run()

        # Step 1: Baseline Offset (Nuclear Repulsion)
        total_offset = problem.nuclear_repulsion_energy
        
        if use_active_space:
            # Step 2: Extract Frozen Core Energy manually before transformation
            transformer = ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=2)
            
            # Catch core energy BEFORE transform (reliable in 0.7+)
            core_energy = 0.0
            if hasattr(transformer, 'occupied_core_energies'):
                core_val = transformer.occupied_core_energies
                core_energy = np.sum(core_val) if isinstance(core_val, (list, np.ndarray)) else core_val
            
            problem = transformer.transform(problem)
            
            # Step 3: Combine with Nuclear Repulsion
            total_offset = problem.nuclear_repulsion_energy + core_energy

        ansatz = UCCSD(
            problem.num_spatial_orbitals, 
            problem.num_particles, 
            mapper,
            initial_state=HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
        )

        temp_conv = []
        def callback(count, params, mean, std=None): 
            temp_conv.append(mean)

        vqe = VQE(Estimator(), ansatz, SLSQP(), callback=callback)
        hamiltonian = mapper.map(problem.second_q_ops()[0])
        result = vqe.compute_minimum_eigenvalue(hamiltonian)

        # TOTAL ENERGY = VQE Electronic + (Nuclear + Core Offset)
        total_energy = float(result.eigenvalue.real) + total_offset
        all_dist.append(d)
        all_energ.append(total_energy)

        if total_energy < best_energy:
            best_energy = total_energy
            best_conv = [e + total_offset for e in temp_conv]
            best_ansatz = ansatz

    status.update(label="VQE Simulation Complete", state="complete", expanded=False)
    hw_circ = transpile(best_ansatz, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)
    
    return all_dist, all_energ, best_conv, all_dist[np.argmin(all_energ)], best_energy, hw_circ, mapper_name

# --- UI CONTROL FLOW ---
mol_choice = st.selectbox("Select Target System", ["H2", "LiH"])
run_btn = st.button("RUN QUANTUM ANALYSIS", use_container_width=True)

if run_btn:
    d, e, conv, b_dist, m_e, circ, m_name = execute_vqe_engine(mol_choice)
    
    st.markdown("### 📊 SIMULATION METRICS")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-card"><p class="spec-label">Bond Length</p><p class="spec-value">{b_dist:.3f} Å</p></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><p class="spec-label">Ground State Energy</p><p class="spec-value">{m_e:.5f} Ha</p></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><p class="spec-label">Transpiled Depth</p><p class="spec-value">{circ.depth()}</p></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><p class="spec-label">Logical Qubits</p><p class="spec-value">{circ.num_qubits}</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    g1, g2 = st.columns(2)
    with g1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(d, e, 'o-', color='#2563eb', linewidth=2)
        ax1.set_title("Potential Energy Surface (PES)", fontweight='bold')
        ax1.set_xlabel("Distance (Å)"); ax1.set_ylabel("Total Energy (Ha)")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
    with g2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(conv, color='#10b981', linewidth=2)
        ax2.set_title("Optimization Path @ Equilibrium", fontweight='bold')
        ax2.set_xlabel("Iteration"); ax2.set_ylabel("Total Energy (Ha)")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    st.markdown("### 🛠 HARDWARE & ALGORITHM DIAGNOSTICS")
    t1, t2 = st.columns(2)
    with t1:
        st.table({
            "Parameter": ["Quantum Mapper", "Logical Qubits", "Circuit Depth", "Optimizer"], 
            "Value": [m_name, circ.num_qubits, circ.depth(), "SLSQP"]
        })
    with t2:
        st.table({"Gate Type": list(circ.count_ops().keys()), "Count": list(circ.count_ops().values())})

    with st.expander("🔬 DECOMPOSED QUANTUM CIRCUIT ARCHITECTURE"):
        st.pyplot(circ.draw('mpl', scale=0.8))

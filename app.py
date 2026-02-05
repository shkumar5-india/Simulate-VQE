import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- FIXED QISKIT 1.0+ IMPORTS ---
from qiskit import transpile
from qiskit.primitives import Estimator # Standard Estimator for algorithms
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

# --- UI CONFIG ---
st.set_page_config(page_title="QUANTUM VQE ENGINE", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp { background: #010409; font-family: 'monospace'; }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid #00d2ff55;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .neon-text { color: #00d2ff; text-shadow: 0 0 10px #00d2ff; font-weight: bold; }
    .spec-box { border-left: 3px solid #00ff88; padding-left: 15px; margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 style="text-align:center; color:white; letter-spacing:5px;">VQE NEURAL CORE</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#00d2ff;">REAL-TIME QUANTUM STATE OPTIMIZATION</p>', unsafe_allow_html=True)

# --- CORE SIMULATION ENGINE ---
def run_vqe_engine(mol_type):
    if mol_type == "H2":
        d_range = np.arange(0.5, 1.1, 0.1)
        mapper = ParityMapper(num_particles=(1, 1))
        target, core_e, atom = 0.70, 0, "H"
    else:
        d_range = [1.2, 1.595, 2.0]
        mapper = JordanWignerMapper()
        target, core_e, atom = 1.595, -7.783, "Li"

    all_distances = []
    all_energies = []
    final_ansatz = None
    
    status_spot = st.empty()
    plot_spot = st.empty()

    for d in d_range:
        status_spot.markdown(f'<p class="neon-text">COMPUTING STATE AT: {d} Å ...</p>', unsafe_allow_html=True)
        
        driver = PySCFDriver(atom=f"{atom} 0 0 0; H 0 0 {d}", basis="sto3g")
        prob = driver.run()
        if mol_type == "LiH":
            prob = ActiveSpaceTransformer(2, 2).transform(prob)
        
        ansatz = UCCSD(prob.num_spatial_orbitals, prob.num_particles, mapper,
                       initial_state=HartreeFock(prob.num_spatial_orbitals, prob.num_particles, mapper))
        
        # Real-time Animation Logic
        iter_energies = []
        def callback(count, params, mean, std):
            iter_energies.append(mean + prob.nuclear_repulsion_energy + core_e)
            with plot_spot.container():
                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots(figsize=(5,3))
                    plt.style.use('dark_background')
                    ax.plot(all_distances, all_energies, 'o-', color='#00d2ff', label="Scan")
                    ax.set_title("SURFACE SCAN")
                    st.pyplot(fig)
                    plt.close()
                with c2:
                    fig2, ax2 = plt.subplots(figsize=(5,3))
                    ax2.plot(iter_energies, color='#00ff88', label="Optimization")
                    ax2.set_title(f"LIVE VQE ({d} Å)")
                    st.pyplot(fig2)
                    plt.close()

        vqe = VQE(Estimator(), ansatz, SLSQP(), callback=callback if abs(d-target)<0.05 else None)
        res = vqe.compute_minimum_eigenvalue(mapper.map(prob.second_q_ops()[0]))
        
        current_energy = res.eigenvalue.real + prob.nuclear_repulsion_energy + core_e
        all_distances.append(d)
        all_energies.append(current_energy)
        
        if abs(d-target)<0.05:
            final_ansatz = ansatz

    status_spot.empty()
    hw_circ = transpile(final_ansatz, basis_gates=['cx','rz','sx','x'], optimization_level=1)
    return all_distances, all_energies, target, min(all_energies), hw_circ

# --- UI CONTROL ---
l, r = st.columns([1, 4])
with l:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    choice = st.selectbox("TARGET SYSTEM", ["H2", "LiH"])
    btn = st.button("INITIALIZE ENGINE")
    st.markdown('</div>', unsafe_allow_html=True)

if btn:
    d_list, e_list, best_d, min_e, circ = run_vqe_engine(choice)
    
    st.markdown("---")
    res1, res2 = st.columns([2, 1])
    
    with res1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🛠 HARDWARE DIAGNOSTICS")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="spec-box"><b>QUBITS</b><br><span class="neon-text">{circ.num_qubits}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="spec-box"><b>DEPTH</b><br><span class="neon-text">{circ.depth()}</span></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="spec-box"><b>CNOT GATES</b><br><span class="neon-text">{circ.count_ops().get("cx", 0)}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="spec-box"><b>TOTAL GATES</b><br><span class="neon-text">{sum(circ.count_ops().values())}</span></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="spec-box"><b>OPTIMIZER</b><br><span class="neon-text">SLSQP</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="spec-box"><b>BASIS</b><br><span class="neon-text">STO-3G</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with res2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("💎 ANALYSIS")
        st.metric("BOND LENGTH", f"{best_d} Å")
        st.metric("ENERGY", f"{min_e:.5f} Ha")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("🔬 VIEW QUANTUM ARCHITECTURE"):
        fig_circ = circ.draw('mpl', style='iqp-dark')
        st.pyplot(fig_circ)

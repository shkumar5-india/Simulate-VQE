import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# Corrected Qiskit 1.0+ Imports
from qiskit import transpile
from qiskit.primitives import StatevectorEstimator as Estimator # Updated for modern Qiskit
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

# Page Config
st.set_page_config(page_title="Quantum VQE Lab", layout="wide")
st.title("🧪 Quantum Chemistry: VQE Bond Length Predictor")
st.markdown("---")

def run_simulation(molecule_type):
    with st.spinner(f"Running Quantum Simulation for {molecule_type}... Please wait."):
        if molecule_type == "H2":
            distances = np.arange(0.5, 1.0, 0.1) # Fast scan
            mapper = ParityMapper(num_particles=2)
            target = 0.75
            core_e = 0
        else:
            distances = [1.2, 1.595, 2.0] # Fast scan for LiH
            mapper = JordanWignerMapper()
            target = 1.595
            core_e = -7.783

        energies, conv_history = [], []
        final_ansatz = None

        for d in distances:
            temp_history = []
            def callback(count, params, mean, std): temp_history.append(mean)
            
            atom_sym = "H" if molecule_type == "H2" else "Li"
            driver = PySCFDriver(atom=f"{atom_sym} 0 0 0; H 0 0 {d}", basis="sto3g")
            problem = driver.run()
            
            if molecule_type == "LiH":
                problem = ActiveSpaceTransformer(2, 2).transform(problem)

            ansatz = UCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper,
                           initial_state=HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper))

            vqe = VQE(Estimator(), ansatz, SLSQP(), callback=callback if abs(d-target)<0.01 else None)
            result = vqe.compute_minimum_eigenvalue(mapper.map(problem.second_q_ops()[0]))
            
            total_e = result.eigenvalue.real + problem.nuclear_repulsion_energy + core_e
            energies.append(total_e)
            
            if abs(d-target)<0.01:
                conv_history = [e + problem.nuclear_repulsion_energy + core_e for e in temp_history]
                final_ansatz = ansatz

        hw_circ = transpile(final_ansatz, basis_gates=['cx','rz','sx','x'], optimization_level=1)
        return distances, energies, conv_history, target, min(energies), hw_circ

# Sidebar for controls
st.sidebar.header("Simulation Settings")
mol = st.sidebar.selectbox("Select Molecule", ["H2", "LiH"])

if st.sidebar.button("Run Quantum Simulation"):
    dist, energ, conv, best_d, min_e, circ = run_simulation(mol)
    
    # Dashboard Layout
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Bond Length", f"{best_d} Å")
    col2.metric("Ground State Energy", f"{min_e:.5f} Ha")
    col3.metric("Quantum Gates (CX)", circ.count_ops().get('cx', 0))

    # Plots
    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots()
        ax1.plot(dist, energ, 'bo-')
        ax1.set_title(f"{mol} Potential Energy Surface")
        ax1.set_xlabel("Distance (Å)")
        ax1.set_ylabel("Energy (Ha)")
        st.pyplot(fig1)
        
    with c2:
        fig2, ax2 = plt.subplots()
        ax2.plot(conv, color='green')
        ax2.set_title(f"Convergence at {best_d} Å")
        ax2.set_xlabel("Iteration")
        st.pyplot(fig2)

    st.subheader("Final Hardware Circuit Statistics")
    st.write(f"**Qubits Used:** {circ.num_qubits} | **Circuit Depth:** {circ.depth()}")

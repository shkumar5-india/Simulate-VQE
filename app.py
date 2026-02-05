from flask import Flask, jsonify, render_template
import io, base64, time
import numpy as np
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

app = Flask(__name__)

def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', facecolor='#1e1e1e')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def simulate_molecule(molecule_type):
    if molecule_type == "H2":
        distances = np.arange(0.5, 1.0, 0.1) # Simplified for speed
        energies, conv_history = [], []
        target = 0.75
    else:
        distances = [1.0, 1.4, 1.595, 2.0]
        energies, conv_history = [], []
        target = 1.595

    final_ansatz = None
    for d in distances:
        temp_history = []
        callback = lambda count, params, mean, std: temp_history.append(mean)
        
        driver = PySCFDriver(atom=f"{'H' if molecule_type=='H2' else 'Li'} 0 0 0; H 0 0 {d}", basis="sto3g")
        raw_prob = driver.run()
        
        if molecule_type == "LiH":
            transformer = ActiveSpaceTransformer(2, 2)
            problem = transformer.transform(raw_prob)
            mapper = JordanWignerMapper()
            core_e = -7.783
        else:
            problem = raw_prob
            mapper = ParityMapper(num_particles=2)
            core_e = 0

        ansatz = UCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper,
                       initial_state=HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper))

        vqe = VQE(Estimator(), ansatz, SLSQP(), callback=callback if abs(d-target)<0.01 else None)
        result = vqe.compute_minimum_eigenvalue(mapper.map(problem.second_q_ops()[0]))
        
        total_e = result.eigenvalue.real + problem.nuclear_repulsion_energy + core_e
        energies.append(total_e)
        
        if abs(d-target)<0.01:
            conv_history = [e + problem.nuclear_repulsion_energy + core_e for e in temp_history]
            final_ansatz = ansatz

    # Generate PES Plot
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    plt.style.use('dark_background')
    ax1.plot(distances, energies, 'o-', color='#00d2ff')
    ax1.set_title(f"{molecule_type} Energy Profile")
    pes_plot = fig_to_base64(fig1)
    
    # Generate Convergence Plot
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(conv_history, color='#00ff88')
    ax2.set_title("VQE Convergence")
    conv_plot = fig_to_base64(fig2)

    hw_circ = transpile(final_ansatz, basis_gates=['cx','rz','sx','x'], optimization_level=1)
    
    return {
        "molecule": molecule_type,
        "bond_length": target,
        "energy": min(energies),
        "qubits": hw_circ.num_qubits,
        "depth": hw_circ.depth(),
        "gates": hw_circ.count_ops().get('cx', 0),
        "pes_plot": pes_plot,
        "conv_plot": conv_plot
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run/<molecule>')
def run_simulation(molecule):
    data = simulate_molecule(molecule)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
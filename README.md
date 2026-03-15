# physis-ML-hackathon

## Overview
This project is an AI-powered inverse-design simulator for photonic quantum circuits. Given a target density matrix $\rho_{output}$ defined on an $n$-qubit photonic Hilbert space, our Genetic Algorithm (GA) autonomously discovers the optimal experimental setup required to prepare that quantum state. 

Unlike standard quantum computing models that treat photons as simple 2D qubits, our physics engine correctly encodes qubits using both *photonic polarization and spatial modes*. The core components are derived as $4\times4$ Unitary matrices (e.g., $I_{spatial} \otimes U_{pol}$) using Kronecker products, creating an accurate model of a physical lab bench. Our solver natively generates entangled pure states using Spontaneous Parametric Down-Conversion (SPDC) sources. Furthermore, it employs a robust *ancilla-photon and post-selection pipeline* to probabilistically simulate and prepare highly complex mixed states (e.g., Werner States).

---

## Component Library (Physical Purposes)
As outlined in the problem statement requirements, our component library features the following optical tools:

### Phase Shifter (PS)
* **In Math:** This is a diagonal matrix. It applies a complex phase shift ($\phi$) strictly to the second element (the Vertical polarization component).
* **In Physics:** In a lab, this is often a piece of glass with a variable index of refraction. It artificially slows down a specific component of the light, altering its quantum phase without changing its actual path or rotating its probability amplitude.

### Half-Wave Plate (HWP)
* **In Math:** Defines as a $2\times2$ matrix containing angle $\theta$ because it only affects polarization and is a tensored with the spatial identity matrix to form a $4\times4$ matrix.
* **In Physics:** A wave plate is a special crystal through which light travels at different speed along its vertical and horizontal axes by physically rotating the angle by $\theta$.

### Quarter-Wave Plate (QWP)
* **In Math:** A $2\times2$ matrix containing squared sine cosine terms along with complex img numbers (1j). It is also tensored with the spatial identity matrix.
* **In Physics:** It’s a crystal which delays ones polarization axes by 90 degrees. Transforms linearly polarized light into elliptically or circularly polarized light.

### 50:50 Beam Splitter (BS)
* **In Math:** Unlike the wave plates, this matrix acts on the spatial paths. It is a $2\times2$ matrix with transmission ($1/\sqrt{2}$) and reflection ($i/\sqrt{2}$) elements. It is tensored with the polarization identity. The imaginary $i$ represents the necessary 90-degree phase shift that occurs when light reflects off a denser medium.
* **In Physics:** This is a partially silvered piece of glass. When a single photon hits it, the photon does not break in half. Instead, it enters a spatial superposition—it has a **50%** probability of passing straight through and a **50%** probability of bouncing off at an angle. It is the fundamental component for creating spatial interference (like in a Mach-Zehnder interferometer).

### Polarizing Beam Splitter (PBS)
* **In Math:** This is where spatial and polarization math collide. The code uses projection matrices and leaves horizontal light in its current path.
* **In Physics:** If a photon is horizontally polarized the PBS allows it to pass straight through as if the glass isn’t there, while if a photon is vertically polarized the PBS acts like mirror and reflects it to a complete new path. It is highly valuable for entangling a photon spatial location.

### Cross-Kerr Nonlinear Crystal (CK)
* **In Math:** It is a multi photon gate. The code dynamically builts a diagonal matrix which scans the system. If it detects two specific photons which are both in vertical polarization state it multiplies the amplitude by $e^{i\theta}$.
* **In Physics:** Cross-Kerr crystal, the electromagnetic field of one photon alters the refractive index of the crystal, which slightly slows down the second photon. This conditional phase shift allows the two independent photons to become deeply entangled.

### SPDC source
* **In Math:** Mathematically, the SPDC source acts as the starting generator for your quantum simulation, rather than a component that modifies an existing state. By default, it generates a perfect 2-photon Bell state as a $16\times16$ density matrix. If the target requires 4 qubits, it dynamically combines multiple sources to build a massive $256\times256$ starting matrix.
* **In Physics:** It is a crystal takes one of those high-energy laser photons and splits it into two lower-energy "twin" photons. Because they were born from the exact same photon at the exact same time, these twins come out perfectly entangled. It's basically the gold-standard trick physicists use to create quantum entanglement on demand!

---

## The Scoring Engine
The AI evaluates generated circuits using the exact Quantum Fidelity calculation to compare the ideal target state and the generated output state:

$$F(\rho_{target}, \rho_{out}) = \left(\text{Tr}\sqrt{\sqrt{\rho_{target}} \rho_{out} \sqrt{\rho_{target}}} \right) ^2$$

## How to Run
1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Execute:** Run `python main.py`
3. **Judge's Console:** The interactive console allows you to select built-in presets (e.g., Bell States, Werner States) or dynamically load custom `.npy` files representing target density matrices.

### Requirements
```text
numpy==1.26.4
scipy==1.13.0
matplotlib==3.8.4

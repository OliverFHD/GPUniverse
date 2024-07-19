# GPUniverse

This python package bundles utilities from the following two publications:
- Friedrich, Singh, Doré (2022)
- Friedrich, Cao, Carroll, Cheng, Singh (2024)


notebook GPO_Universe.ipynb:
---------------------------
The notebook GPO_Universe.ipynb is implementing the calculations presented in Friedrich, Singh, Doré (2022) that model quantum fields in finite dimensional Hilbert spaces with Generalised Pauli Operators (GPOs).


simulate_overlapping_qubits.py:
------------------------------
The file simulate_overlapping_qubits.py contains code for simulating sets of qubits that are only quasi independent (i.e. the Pauli algebras of different qubits have small, but non-zero anti-commutator). Friedrich, Cao, Carroll, Cheng, Singh (2024) have used these simulations to validate some of the analytical results for their holographic version of the Weyl field.

--> This part of the package has GPU capability.

holographic_Weyl_field_plots.ipynb:
----------------------------------
Notebook that reproduces key figures from Friedrich, Cao, Carroll, Cheng, Singh (2024). Some of these figures use data that is provided in the sub-directory "./overlapping_qubits_data/".


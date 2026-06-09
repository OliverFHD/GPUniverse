# GPUniverse

This python package bundles utilities from the following two publications:
- Friedrich, Singh, Doré (2022)
- Friedrich, Cao, Carroll, Cheng, Singh (2024)
- Cao, Friedrich, Girard, Loizeau, Singh (2026)


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

Hint_vs_Hq.ipynb:
----------------------------------
Notebook that reproduces computations from section 3 of Cao, Friedrich, Girard, Loizeau, Singh (2026).

--> For the code with which we find minimum-interaction qubits starting from an input spectrum, see the following repo: https://github.com/nicolasloizeau/quantum_mereology
--> And example notebook for our complete iterative algorithm should appear in that repo within a few days of the paper's arxiv release

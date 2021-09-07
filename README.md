# auto_eqf
General code for the Equivariant Filter (EqF) in Python3.
The automatic EqF uses numerical differentiation to minimise the work involved in implementing an equivariant filter.
Of course, for analytic derivatives are preferred for applications where speed is important, but the automatic EqF framework facilitates fast and easy prototyping.

## Dependencies

In order to run the code, the following dependencies are required.
- NumPy: `pip install numpy`
- pylie: `https://github.com/pvangoor/pylie`
- Matplotlib: (for the example) `pip install matplotlib`

## Citation

This code was produced as part of academic research.
If you are using this code in a research setting, please cite the following paper:

- van Goor, Pieter, Tarek Hamel, and Robert Mahony. "Equivariant filter (eqf)." *arXiv preprint arXiv:2010.14666* (2020).
# MIDL Python Example

This is a minimal Python project implementing the core workflow of
Mutual-Information-based Dimensional Learning (MIDL) from:

- Lei Zhang, Guowei He (2025), *Computer Methods in Applied Mechanics and Engineering*.

## Files

- `midl.py`: core MIDL class and result dataclass.
- `example_math.ipynb`: runnable synthetic demo.
- `Example`: This folder contains several examples of dimensionless analysis using MI-DL.
- `requirements.txt`: Python dependencies.

## Quick start

```bash
# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run the demo
python example_math.ipynb
```

## Notes

- Inputs to `MIDL.fit(...)`:
  - `Pi_independent`: independent dimensionless groups, shape `(N, n)`, values must be strictly positive.
  - `pi_dependent`: dependent dimensionless quantity, shape `(N,)`.
- The algorithm returns orthonormal directions `W`, MI scores per direction, and an inferred dominant count.
- The implementation is kept close to the paper's flow chart:
  1) log transform, 2) solve first MI maximization, 3) loop in orthogonal complements, 4) detect dominant quantities by MI gap.

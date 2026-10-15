# DRL_based_MPPT_SCC_CubeSat_EPS
A DRL-based (DQN) MPPT algorithm integrated with a Fibonacci sequence switched capacitor converter(FSSCC) for efficient energy harvesting in CubeSat EPS applications

## Interactive MPPT simulator

The [browser simulator](simulator/README.md) compares P&O with a trained demonstration DQN under representative irradiance, panel, controller, and averaged converter settings. It includes response plots, energy and tracking metrics, a time inspector, CSV export, reproducible training code, and numerical validation.

The simulator uses an empirical PV curve, an averaged FSSCC model, and a demonstration DQN trained in the reduced environment. It complements the MATLAB/Simulink model and original training data.

See [simulator/README.md](simulator/README.md) for setup, model assumptions, and validation commands.

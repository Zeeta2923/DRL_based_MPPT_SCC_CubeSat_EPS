# CubeSat MPPT Lab

Browser simulator for comparing P&O and DQN control of an averaged Fibonacci Sequence Switched Capacitor Converter (FSSCC). The default parameters are representative; the panel and converter settings can be adjusted for sensitivity studies.

## Run locally

From this directory, use Python 3 to serve the static app:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory dist
```

Open <http://127.0.0.1:8000>. No JavaScript dependencies or build step are needed. Opening `index.html` directly as a file may prevent loading the modules or policy.

Change panel irradiance, temperature, reference Voc/Isc, curve shape, MPPT step, sampling period, response time, noise, exploration, and seed. Expand the averaged FSSCC controls to adjust the effective ratio, bus voltage, capacitance, loss coefficient, and frequency limits. Compare panel power, voltage, frequency, bus power, tracking efficiency, energy, settling, and ripple. The P–V plot and time inspector show both operating points; Export CSV saves the trajectories and settings.

## Model boundary

The PV curve is empirical and has one maximum. The converter uses an effective ratio and charge-transfer coefficient selected by the user, rather than values derived from the actual 10-switch circuit. A shared ideal, model-based inner PFM loop maps voltage references to switching frequencies. First-order voltage dynamics and quasi-static bus energy are estimates. Individual switches, capacitor ripple, PWM duty timing, battery SOC, partial shading, and orbital dynamics are not simulated.

The model assumptions and DQN training range are documented in the app. The included `dist/policy.json` contains demonstration weights trained in this reduced environment. Comparisons apply to the selected parameters and do not validate the full MATLAB/Simulink circuit.

## Validate

With Node.js 18 or newer, from this directory:

```bash
npm test
```

The suite exercises 66 parameter combinations, including eclipse, no sunlight, and converter constraints. It checks the analytic MPP against an independent sweep, current balance, power bounds, energy integration, seeded repeatability, input validation, interface events, and CSV structure. Interface checks use a DOM fixture; browser visual/layout verification was unavailable.

## Reproduce the demonstration DQN

Training requires Python 3 and NumPy. From this directory:

```bash
python3 -m venv .venv
# Activate the environment for your operating system, then:
python -m pip install numpy
python scripts/train_dqn.py
```

Training overwrites `dist/policy.json`. The seed is 42. The 5 → 48 → 48 → 5 Q-network uses replay memory, a target network, and Adam over 208,000 transitions. Architecture, reward, hyperparameters, and operating ranges are recorded in the policy metadata and app. Floating-point differences across NumPy/platform versions can affect reproduced weights. Inference runs locally in the browser, with no external inference service or browser training.

## Access and data

This is a static application. Authentication for a hosted installation must be provided by the hosting platform. No deployment workflow is included.

Calculations run in the browser. Experiment settings are not uploaded. CSV export downloads the settings and response data locally.

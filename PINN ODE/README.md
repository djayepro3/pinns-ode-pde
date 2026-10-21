# Physics-Informed Neural Networks for Ordinary Differential Equations

This folder holds a small series of hands-on projects that teach a neural network the physics behind an ordinary differential equation (ODE). Each project solves a real problem from physics or engineering, and each one is built twice: once **from scratch with NumPy**, so you can see exactly what happens inside, and once with **PyTorch**, using the tools people use in practice.

The projects are ordered by the **order of the equation**, meaning the highest derivative it contains. Each one adds something new, so they are best read in order.

## The projects at a glance

| Project | Equation | Order | Type of problem | Data used? | What is new |
|---|---|---|---|---|---|
| [1D Ball](1D%20Ball/) | $\frac{dh}{dt} = v_{0} - g t$ | 1st | Initial-value | Yes, noisy readings | The basic idea: add the equation to the loss so the network stops following noise |
| [2D Damped Harmonic](2D%20Damped%20Harmonic/) | $m x'' + c x' + k x = 0$ | 2nd | Initial-value | Yes, noisy readings from a short window | Second derivatives, two initial conditions, predicting beyond the data, an energy check, a training video |
| [3D Blasius Flow](3D%20Blasius%20Flow/) | $f''' + 0.5 f f'' = 0$ | 3rd | Boundary-value | No, the equation is solved on its own | Third derivatives, conditions in two places, a failure mode and its fix, an inverse problem from real-style measurements |

The number in each folder name is the order of the equation (1st, 2nd, 3rd). It has nothing to do with the number of space dimensions. Those come in the PDE folder.

## The learning path

**1. Ball (start here).** A ball is thrown upwards and we note its height a few times. A network trained on the readings alone follows every bit of noise. A network that is also told `dh/dt = v0 - g t` and where the ball started gives a smoother, more truthful curve. This project introduces the whole recipe: the data loss, the equation loss, the initial-condition loss, and how automatic differentiation gets the derivative of a network.

**2. Damped harmonic oscillator.** A weight on a spring bounces with shrinking swings. We only measure the first part of the motion and ask the network to predict the rest. The data-only network has nothing to go on and drifts away. The network that knows the equation continues the decaying swing. The project derives the exact solution for all three damping regimes, checks the networks against the energy balance, finds the friction from the data, and shows the training as a side-by-side video.

**3. Blasius boundary layer.** Air flowing over a flat plate slows down in a thin layer next to the surface, and that layer decides the friction drag. The whole layer reduces to one third-order equation with no closed-form solution. Here the network is an equation solver: there are no measurements. The project derives the equation from the flow equations, shows that a network started from random weights can get stuck on a wrong answer, fixes it with a short warm-up, and then recovers the wall friction from noisy velocity readings that stop short of the wall.

## What every project contains

Each project folder has the same layout:

| Item | What it is |
|---|---|
| `README.md` | The summary of that project, with its results |
| the notebook (`.ipynb`) | The full project, in four parts (below) |
| `media/` | The training video and GIF (in the oscillator and Blasius projects) |

Inside each notebook:

1. **Background.** The physics, the full mathematical derivations, and how the loss is built.
2. **Setup.** The constants, the exact or reference solution, the data, and helper functions.
3. **Part A: from scratch.** NumPy only. The network, its derivatives, the gradients worked out by hand, a test of those gradients, the optimiser, training and results.
4. **Part B: advanced.** PyTorch with automatic differentiation, then extra tools such as L-BFGS, `torch.func`, inverse problems and loss-weight studies.

## Ideas that build up across the series

| Idea | Where it first appears |
|---|---|
| Data loss + equation loss + initial-condition loss | Ball |
| Automatic differentiation, and the same derivatives written by hand | Ball (Part B), Oscillator (Part A) |
| Where the equation is checked matters (collocation points) | Ball |
| Several initial conditions for a higher-order equation | Oscillator |
| Predicting beyond the measured region | Oscillator |
| Checking a model with a physical law instead of the answer (energy) | Oscillator |
| Learning an unknown constant from data (inverse problem) | Oscillator, Blasius |
| A training video and GIF, and an epoch slider | Oscillator, Blasius |
| Boundary-value problems, with no data at all | Blasius |
| Training that gets stuck, and how to spot and fix it | Blasius |
| A similarity solution that covers a whole family of problems | Blasius |

## A few results

These come from the NumPy runs in Part A of each notebook. Each project's own README has the full tables.

- **Ball.** The error against the exact curve falls from about 0.69 m for the data-only network to about 0.31 m when the equation is checked at the data times, and to about 0.10 m when it is checked at 50 points.
- **Oscillator.** In the part of the motion with no measurements, the data-only network is off by about 2.2 m (it drifts away entirely), while the network that knows the equation is off by about 0.011 m.
- **Blasius.** The network reproduces the wall slope that sets the drag, 0.3321, matching the reference solution. From random starting weights, 4 of 6 runs solved the problem and 2 got stuck. A short warm-up fixed the stuck run we tried.

These are single runs with fixed random seeds, so read them as illustrations and not as benchmarks.

## How to run the notebooks

You need Python 3.9 or newer.

```bash
pip install numpy scipy matplotlib pillow jupyter
pip install ipywidgets       # optional, for the sliders
pip install torch            # needed for Part B of every notebook
pip install deepxde          # optional, only for the DeepXDE cells where a notebook has them
```

Open a notebook with `jupyter notebook`, or upload it to Google Colab, which already has PyTorch. Run the sections in order. Part A of each notebook needs only NumPy, Matplotlib and (for Blasius) SciPy, so you can work through all the from-scratch parts without PyTorch.

To make the MP4 files for the videos you also need `ffmpeg`. If it is missing, the code skips the MP4 step and still writes the GIF. You can run `pip install imageio-ffmpeg` to supply it.

Each notebook takes a few minutes to run in Part A and longer in Part B, because differentiating through a network several times is slower than a plain forward pass. The project READMEs give rough times.

## What comes next

These projects deal with equations that depend on a single variable. The next step is **partial differential equations**, where the unknown depends on space and time together. Those projects live in the [PINN PDE](../PINN%20PDE/) folder.

## Main references

1. M. Raissi, P. Perdikaris, G. E. Karniadakis. *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686-707 (2019).
2. S. Cuomo, V. S. Di Cola, F. Giampaolo, G. Rozza, M. Raissi, F. Piccialli. *Scientific machine learning through physics-informed neural networks: Where we are and what's next.* Journal of Scientific Computing, 92, 88 (2022).
3. D. P. Kingma and J. Ba. *Adam: A method for stochastic optimization.* ICLR (2015).
4. L. Lu, X. Meng, Z. Mao, G. E. Karniadakis. *DeepXDE: A deep learning library for solving differential equations.* SIAM Review, 63(1), 208-228 (2021).

Each project's README lists the references specific to that project.

<!--
To add a new project later:
1. Create a new folder and put its notebook, README and media folder inside.
2. Add a row to the table "The projects at a glance".
3. Add a short paragraph under "The learning path".
4. Add any new ideas to the table "Ideas that build up across the series".
5. Add a one-line result under "A few results".
-->
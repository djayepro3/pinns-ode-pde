# Solving the Blasius Boundary Layer with a Physics-Informed Neural Network

**Author:** Dishanand Jayeprokash

Air flowing over a wing does not slide past the surface freely. The air touching the metal is held still by friction, and a thin layer above it goes gradually from "stopped" to "full speed". That thin region is the **boundary layer**, and it decides how much drag a wing has. For a flat plate, the whole layer reduces to one third-order differential equation, written down by Heinrich Blasius in 1908. It has no closed-form solution.

In this project a neural network learns the solution **using only the equation and three boundary conditions, with no measurements at all**. The notebook then shows what can go wrong, how to fix it, and how the same machinery recovers wall friction from noisy velocity readings.

<img src="https://github.com/djayepro3/pinns-ode-pde/blob/main/PINN%20ODE/3D%20ODE%20Blasius%20Boundary%20Layer/media/boundary_layer.gif" width="100%">

*Both networks solve the same equation from the same random weights. Left: a plain random start, which gets stuck on a wrong answer. Right: a start with a short warm-up, which finds the Blasius profile. Top: the velocity profile. Middle: the boundary layer over the plate, as the network sees it. Bottom: the network's wall slope against the exact value.*

This is the third project in a small series on physics-informed neural networks for ordinary differential equations. The first was a first-order equation (a thrown ball), the second a second-order one (a damped oscillator). Here the equation is **third-order**, and three things are new:

- There are **no data**. The network is an equation solver, and we judge it against a classical reference solution.
- The conditions sit in **two places**: two at the wall and one far away. This is a boundary-value problem, not an initial-value problem.
- Training can **get stuck**. The equation has a useless "do nothing" solution, and a network started from random weights can sink into it.

## The problem in short

After simplifying the flow equations for a thin layer and using the fact that the layer grows like the square root of the distance from the front edge, the problem becomes

```
f''' + 0.5 * f * f'' = 0
f(0) = 0,   f'(0) = 0,   f'(infinity) = 1
```

where `f'(eta)` is the air speed as a fraction of the free-stream speed, and `eta` is the height above the plate measured in boundary-layer units. The notebook derives this equation step by step from the flow equations.

One solution covers every plate, every speed, and every fluid. The number that matters most is the slope at the wall, `f''(0) = 0.3321`, because it sets the friction drag: the skin-friction coefficient is `2 f''(0) / sqrt(Re_x)`, about `0.664 / sqrt(Re_x)`.

For air at 10 m/s, half a metre from the front edge of a plate, the layer is about **4.25 mm** thick and the Reynolds number is about 3.3 x 10^5. Try other numbers with the sliders in the notebook.

## What is in the notebook

The file is `pinn_blasius_boundary_layer.ipynb`.

| Section | Contents |
|---|---|
| Background | Why boundary layers matter, a full derivation of the Blasius equation, the boundary conditions, a reference-solution trick based on rescaling, the quantities engineers read off (skin friction, layer thickness, drag), a built-in physics check, the PINN loss for a third-order equation, how to differentiate the network three times, the "do nothing" trap, and the inverse problem |
| Setup | Constants, the reference solver, an interactive explorer (speed, distance, air or water), a map of the boundary layer over a plate, the noisy measurements for the inverse problem, and the helper functions for the video |
| Part A: NumPy only | A network that returns its own first, second and third derivatives, gradients derived by hand, a gradient check against finite differences, Adam, a survey of random starts, the warm-start fix, the training video and GIF, and an inverse problem with a repeat test |
| Part B: PyTorch | The same experiment with automatic differentiation, then L-BFGS, `torch.func`, an inverse problem with a trainable constant, and wedge flows with a pressure gradient (Falkner-Skan) |
| Summary and references | What the experiments showed, the limits of the example, our own design choices, ideas to try, and the reading list |

## What was found

These numbers are from the NumPy run in Part A.

**1. It solves the equation.** A network trained on the equation and the three boundary conditions alone reproduces the reference solution.

| Quantity | PINN | Reference | Textbook |
|---|---|---|---|
| Wall slope `f''(0)` | 0.3321 | 0.3321 | 0.3321 |
| Layer thickness `eta_99` | 4.916 | 4.910 | 4.91 |
| Displacement integral | 1.7214 | 1.7208 | 1.721 |
| Momentum integral | 0.6645 | 0.6641 | 0.664 |

The error in the velocity profile is about 9 x 10^-5. The momentum integral also matches `2 f''(0)`, which is a physics check that needs no reference solution.

**2. Random starts can get stuck.** We trained six networks with different random weights (4000 epochs each).

| Seeds | Wall slope | Profile error | Result |
|---|---|---|---|
| 0, 1, 3, 5 | about 0.332 | 2 x 10^-4 to 6 x 10^-4 | solved |
| 2, 4 | about -0.12 to -0.13 | 0.84 to 0.88 | stuck |

The function `f = 0` satisfies the equation and the wall conditions, and breaks only the far-field condition. A stuck network settles into a compromise near that useless solution. Which seeds get stuck depends on the settings, so read "2 out of 6" as an illustration and not as a failure rate. A stuck run also fails the momentum check, so you can spot it without knowing the right answer.

**3. A warm start fixes it.** Before switching the physics on, we train the network for 500 epochs to follow a rough sketch of the velocity profile, `f' = 1 - exp(-eta/2)`. For seed 2, which got stuck from a cold start, the warm start reaches a wall slope of 0.33215 with a profile error of about 9 x 10^-5.

**4. It recovers wall friction from noisy readings.** The inverse problem imitates a real measurement: 14 noisy velocity readings between 0.6 mm and 6 mm above the plate, with nothing closer to the wall (the hardest place to measure). The length scale of the layer is treated as unknown and learned along with the profile. Quantities at the wall are what set the drag.

| Method | Wall velocity gradient error |
|---|---|
| PINN | -1.8% |
| Cubic polynomial extended to the wall | +26.2% |

A repeat over 300 noise draws, reusing the profile learned in the main run and only refitting the scale, gave the PINN approach a mean error of about 0% with a spread of about 2%. The polynomial was typically more than 50% too high, with a spread of about 11%.

These are results from one set of random weights and one noise sample (plus the repeat test above), so read them as an illustration and not a benchmark. Part B repeats the experiments in PyTorch. Its numbers will differ a little, because the starting weights differ and PyTorch works in 32-bit numbers where the NumPy code uses 64-bit ones.

## The training video and GIF

Section A10 turns snapshots taken during training into an animation that compares a cold start with a warm start. For each network it shows the velocity profile against the exact one, a map of the boundary layer built from the network's current profile with the exact edge of the layer as a dashed line, and a strip tracking the network's wall slope as training goes on. Snapshots are spaced more densely in the early epochs, where most of the change happens.

The code is in the Setup section:

- `make_training_video` writes a **GIF** (using Pillow, no extra software) and an **MP4** (using `ffmpeg`).
- `convert_mp4_to_gif` turns an existing MP4 into a GIF with a clean colour palette.
- `snapshot_grid` draws a still picture of a few moments in training. It shows up in a GitHub preview, where animations in notebook output may not.
- `scrub_epochs` gives an epoch slider you can drag, in Jupyter or Colab.

If `ffmpeg` is not installed, the MP4 step prints a message and is skipped. You can install `ffmpeg`, or run `pip install imageio-ffmpeg`, which the code will find automatically. The files are saved to a `media` folder next to the notebook.

## How to run it

You need Python 3.9 or newer.

```bash
pip install numpy scipy matplotlib pillow jupyter
pip install ipywidgets       # optional, for the sliders
pip install torch            # needed for Part B only
jupyter notebook pinn_blasius_boundary_layer.ipynb
```

Run the Background and Setup sections first, then Part A or Part B. Part A needs only NumPy, SciPy and Matplotlib, so you can run everything up to the end of Part A without PyTorch. You can also open the notebook in Google Colab, which already has PyTorch.

Run times on an ordinary laptop CPU, with no GPU needed:

- Setup and Part A together took about 7 minutes in our test, on a single processor core.
- Part B takes longer, because differentiating the network three times costs more. Expect roughly 15 to 25 minutes in total.

## A few things worth knowing

- **Gradient check.** Part A includes a test that compares the hand-written gradients, including the path through the third derivative, against numerical estimates. They agree to a few parts in a million.
- **A solver, not a curve fit.** There are no data in the main experiment. The reference solution is computed with a classical method (one run of a standard integrator plus a rescaling), and only used to judge the network.
- **Our own choices.** The far edge of the domain stands in for infinity at `eta = 8`, the equation is checked at 200 points, and the learning rate falls from 0.003 to 0.0001. The summary section lists these and the reason for each.
- **Things that did not work in our quick tests.** Giving the far-field condition ten or a hundred times more weight did not rescue a stuck run. Building the wall conditions into the network by writing `f = eta^2 * N(eta)` also got stuck in our one test. We did not investigate further.
- **Limits of the model.** The Blasius solution holds for steady, laminar flow with no pressure gradient, up to a Reynolds number of about 5 x 10^5. Part B shows the case with a pressure gradient (Falkner-Skan flows).
- **A classical solver is faster.** For this equation, a standard method is quicker and more accurate. A PINN is worth it when data and equations must be combined, when constants are unknown, or when the problem is a PDE.
- **Random seeds.** Both parts fix their seeds. PyTorch and NumPy draw different random numbers, so which seeds get stuck can differ between Part A and Part B.

## Things to try

- Change `ETA_MAX` to 6 or 12 and see how the answer moves.
- Use fewer collocation points and watch how far the network can wander between them.
- Make the warm-up guess worse and see whether it still helps.
- Move the measurement window away from the wall, or add more noise, and watch the inverse problem degrade.
- Try flow that slows down along the surface (`m < 0` in Part B) and look for the point where the wall slope reaches zero.

## References

1. L. Prandtl. *Uber Flussigkeitsbewegung bei sehr kleiner Reibung.* Verhandlungen des III. Internationalen Mathematiker-Kongresses, Heidelberg, 484-491 (1904).
2. H. Blasius. *Grenzschichten in Flussigkeiten mit kleiner Reibung.* Zeitschrift fur Mathematik und Physik, 56, 1-37 (1908).
3. V. M. Falkner and S. W. Skan. *Some approximate solutions of the boundary layer equations.* Philosophical Magazine, 12, 865-896 (1931).
4. H. Schlichting and K. Gersten. *Boundary-Layer Theory.* 9th edition, Springer (2017).
5. F. M. White. *Viscous Fluid Flow.* 3rd edition, McGraw-Hill (2006).
6. J. D. Anderson. *Fundamentals of Aerodynamics.* 6th edition, McGraw-Hill (2017).
7. M. Raissi, P. Perdikaris, G. E. Karniadakis. *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686-707 (2019).
8. S. Cuomo, V. S. Di Cola, F. Giampaolo, G. Rozza, M. Raissi, F. Piccialli. *Scientific machine learning through physics-informed neural networks: Where we are and what's next.* Journal of Scientific Computing, 92, 88 (2022).
9. D. P. Kingma and J. Ba. *Adam: A method for stochastic optimization.* ICLR (2015).
10. D. C. Liu and J. Nocedal. *On the limited memory BFGS method for large scale optimization.* Mathematical Programming, 45, 503-528 (1989).
11. A. Krishnapriyan, A. Gholami, S. Zhe, R. Kirby, M. W. Mahoney. *Characterizing possible failure modes in physics-informed neural networks.* NeurIPS 34 (2021).
12. S. Wang, X. Yu, P. Perdikaris. *When and why PINNs fail to train: A neural tangent kernel perspective.* Journal of Computational Physics, 449, 110768 (2022).
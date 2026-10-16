# Solving a Second-Order ODE with a Physics-Informed Neural Network: The Damped Oscillator

A weight hangs from a spring and bounces, and each bounce is a little smaller than the last because friction keeps taking energy away. We measure its position for the first 1.8 seconds and then stop. What does it do next?

A neural network trained on those measurements alone cannot tell us. A network that is also trained to obey the equation of motion can. This project shows that difference, step by step, and lets you watch it happen as the networks train.

![Training animation: data-only network vs PINN](PINN ODE/2D ODE Damped Harmonic Osc/media/damped_oscillator_training_from_mp4.gif)

*Left: a network that sees only the data. Right: a network that sees the data and the equation. Both start from the same random weights.*

This is the second project in a small series. The first one, the ball throw, was a first-order equation with one initial condition. Here the equation is **second-order**, which brings three new things: the network has to be differentiated twice, the problem needs two initial conditions (position and velocity), and the answer oscillates.

## The problem in short

A mass on a spring with friction obeys

```
m x'' + c x' + k x = 0,      x(0) = x0,   x'(0) = v0
```

We use `m = 1`, `k = 16`, `c = 0.8`, `x0 = 1` and `v0 = 0`. That gives a natural frequency of 4 rad/s and a damping ratio of 0.1, so the weight swings about two and a half times in 4 seconds while the amplitude shrinks to roughly a fifth. The exact solution is

```
x(t) = exp(-a t) * [ x0 cos(wd t) + (v0 + a x0)/wd * sin(wd t) ],    a = 0.4,  wd = 3.98
```

We only measure the first 1.8 seconds (18 noisy readings), but we ask the network to predict all 4 seconds. The physics is checked at 120 times across the full interval, which costs no extra measurements.

The training loss has four parts: the data, the differential equation, the starting position and the starting velocity.

```
total loss = w_data * data loss + w_ode * ODE loss + w_ic * (position + velocity at t = 0)
```

## What is in the notebook

The file is `pinn_damped_harmonic_oscillator.ipynb`.

| Section | Contents |
|---|---|
| Background | Where the equation appears, how to derive it, a full derivation of the exact solution for all three damping regimes, why two initial conditions are needed, why energy always drains, how the loss is built for a second-order equation, and how the network is differentiated twice |
| Setup | Constants, the exact solution with a check that it satisfies the equation, the noisy data, and the helper functions for plots and for making the video and GIF |
| Part A: NumPy only | A network that returns its own first and second time derivatives, gradients derived by hand, a gradient check against finite differences, Adam written from its equations, training, an energy check, the video and GIF, and an inverse problem that finds the friction from the data |
| Part B: PyTorch | The same experiment with automatic differentiation, then an inverse problem with a trainable constant, L-BFGS, `torch.func`, a loss-weight study, an energy check, and an optional DeepXDE version |
| Summary and references | What the experiments showed, the limits of the example, the design choices that are ours, ideas to try, and the reading list |

## What was found

These numbers are from the NumPy run in Part A. They are the error of each fitted curve against the exact solution (root-mean-square error, in metres), split into the measured window and the part with no data.

| Model | Inside the window (0 to 1.8 s) | Outside the window (1.8 to 4 s) | Overall |
|---|---|---|---|
| The noisy data itself | 0.053 | not measured | not measured |
| Network trained on data only | 0.055 | 2.202 | 1.633 |
| PINN | 0.021 | 0.011 | 0.016 |

- **Inside the window** both networks follow the readings. The data-only network pushes its data loss far below the noise level, so it is reproducing the noise, while the PINN smooths part of it away.
- **Outside the window** the data-only network has nothing to go on and its curve falls away to about −2.7 m. The PINN carries on with the decaying swing and stays very close to the exact curve.
- **Energy check.** A real damped oscillator can only lose energy. The PINN's energy never rises. The data-only network's energy rises over about 74% of the time axis and ends up many times larger than where it started.
- **Inverse problem.** With the damping coefficient treated as unknown and started far from the truth (at 0.1 and at 2.0), both runs settle near 0.77. The true value is 0.8, so the estimate is a little under 4% low. With only 18 noisy readings over about one swing, some error is expected.

This is one noise sample and one random start, so read the table as an illustration and not a benchmark. Part B repeats the comparison in PyTorch. Its numbers will differ somewhat, because the starting weights differ and PyTorch works in 32-bit numbers where the NumPy code uses 64-bit ones.

## The training video and GIF

Part A (section A10) turns snapshots taken during training into an animation. The two networks are shown side by side with the epoch counter above them, and a strip at the bottom plots each network's error on a log scale as training goes on. The snapshots are spaced more densely in the early epochs, where most of the visible change happens.

The code that makes it is in the Setup section:

- `make_training_video` writes a **GIF** (using Pillow, no extra software) and an **MP4** (using `ffmpeg`).
- `convert_mp4_to_gif` turns an existing MP4 into a GIF with a clean colour palette.
- `snapshot_grid` draws a still picture of a few moments in training. It shows up in a GitHub preview, where animations in notebook output may not.
- `scrub_epochs` gives an epoch slider you can drag, in Jupyter or Colab.

If `ffmpeg` is not installed, the MP4 step prints a message and is skipped. You can install `ffmpeg`, or run `pip install imageio-ffmpeg`, which the code will find automatically. The files are saved to a `media` folder next to the notebook.

## How to run it

You need Python 3.9 or newer.

```bash
pip install numpy matplotlib pillow jupyter
pip install ipywidgets       # optional, for the sliders
pip install torch            # needed for Part B only
pip install deepxde          # optional, only for the last cell of Part B
jupyter notebook pinn_damped_harmonic_oscillator.ipynb
```

Run the Background and Setup sections first, then Part A or Part B. Part A needs only NumPy and Matplotlib, so you can run everything up to the end of Part A without PyTorch. You can also open the notebook in Google Colab, which already has PyTorch.

Run times on an ordinary laptop CPU, with no GPU needed:

- Setup and Part A together take a few minutes. Each 20,000-epoch training run in NumPy takes about 20 to 25 seconds.
- Part B takes longer, because differentiating the network twice costs more. Expect something like 15 to 25 minutes in total, mostly from the 20,000-epoch runs and the loss-weight study.

## A few things worth knowing

- **Gradient check.** Part A includes a test that compares the hand-written gradients, including the second-derivative path, against numerical estimates. They agree to better than one part in a hundred million.
- **Our own choices.** The equation residual is divided by `k`, and the velocity error by the natural frequency, so every loss term is on the scale of a position and the weights can stay near 1. The learning rate shrinks from 0.003 to 0.0001 over training. The summary section lists these choices and the reason for each.
- **Oscillations are harder than the ball.** Two and a half swings are noticeably harder to learn than a parabola. That is why this example needs 20,000 epochs, and why PINNs struggle more as the number of swings grows. The notebook says so openly and points to the research on it.
- **A classical solver would be faster.** For an equation this simple, a standard integrator is quicker and more accurate. A PINN is worth it when data and equations must be combined, when constants are unknown, or when the problem is a PDE.
- **Random seeds.** Both parts fix their seeds, so reruns give the same numbers on the same machine.

## Things to try

- Change `c` to make the oscillator critically damped or overdamped. The exact-solution function already handles all three cases.
- Make the measured window longer or shorter and watch how the two networks respond.
- Increase the noise, or use fewer readings.
- Give the ODE loss a slightly wrong damping coefficient and see how far the PINN drifts from the truth.
- Lower the damping so the weight swings five or more times, and see where training starts to struggle.

## References

1. M. Raissi, P. Perdikaris, G. E. Karniadakis. *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686–707 (2019).
2. L. Lu, X. Meng, Z. Mao, G. E. Karniadakis. *DeepXDE: A deep learning library for solving differential equations.* SIAM Review, 63(1), 208–228 (2021).
3. D. P. Kingma and J. Ba. *Adam: A method for stochastic optimization.* ICLR (2015).
4. X. Glorot and Y. Bengio. *Understanding the difficulty of training deep feedforward neural networks.* AISTATS, 249–256 (2010).
5. D. C. Liu and J. Nocedal. *On the limited memory BFGS method for large scale optimization.* Mathematical Programming, 45, 503–528 (1989).
6. A. G. Baydin, B. A. Pearlmutter, A. A. Radul, J. M. Siskind. *Automatic differentiation in machine learning: a survey.* Journal of Machine Learning Research, 18(153), 1–43 (2018).
7. S. Wang, Y. Teng, P. Perdikaris. *Understanding and mitigating gradient flow pathologies in physics-informed neural networks.* SIAM Journal on Scientific Computing, 43(5), A3055–A3081 (2021).
8. A. Krishnapriyan, A. Gholami, S. Zhe, R. Kirby, M. W. Mahoney. *Characterizing possible failure modes in physics-informed neural networks.* NeurIPS 34 (2021).
9. S. Wang, X. Yu, P. Perdikaris. *When and why PINNs fail to train: A neural tangent kernel perspective.* Journal of Computational Physics, 449, 110768 (2022).
10. W. E. Boyce and R. C. DiPrima. *Elementary Differential Equations and Boundary Value Problems.* Wiley, 10th edition (2012).
11. J. R. Taylor. *Classical Mechanics.* University Science Books (2005), Chapter 5.
# Solving a 1D ODE with a Physics-Informed Neural Network

A ball is thrown straight up. We have 12 noisy height readings and we also know the physics behind the motion. This project shows how to use both together by training a small neural network whose loss function includes the differential equation.
ls
The work is a single Jupyter notebook, built in two parts:

- **Part A** writes everything by hand with NumPy. There is no machine-learning library, so you can see exactly what happens inside a PINN.
- **Part B** does the same problem in PyTorch, then goes further with tools that people use on bigger problems.

## The problem in short

With only gravity acting, the height of the ball follows

```
dh/dt = v0 - g*t,      h(0) = h0
```

which has the exact solution `h(t) = h0 + v0*t - 0.5*g*t^2`. We use `g = 9.81`, `h0 = 3` and `v0 = 9`, and generate ten "measurements" between 0 and 2.1 seconds by adding noise to the exact curve.

A network trained on the data alone ends up following the noise. A PINN adds two more terms to the loss, one for the equation and one for the starting height, so the fitted curve stays close to what the physics allows.

```
total loss = w_data * data loss  +  w_ode * ODE loss  +  w_ic * initial-condition loss
```

## What is in the notebook

| Section | Contents |
|---|---|
| Background | The problem, why plain fitting overfits, the idea behind PINNs, the maths for each loss term, and how the derivative of the network is obtained |
| Part A: NumPy only | Network and forward pass, the time derivative worked out by the chain rule, gradients derived by hand, a finite-difference gradient check, Adam written from its equations, training, plots |
| Part B: PyTorch | The code with explanations, then: a comparison of data-only vs PINN, a sweep over the physics weight, Adam followed by L-BFGS, derivatives with `torch.func`, an ODE residual plot, and an optional DeepXDE version |
| Summary and change log | What the experiments showed and limits of the example |

## What was found

These numbers are from the NumPy run in Part A. They show the error of each fitted curve against the exact solution (root-mean-square error, in metres).

| Model | Error |
|---|---|
| The noisy data itself | 0.677 |
| Network trained on data only | 0.686 |
| PINN, physics checked at the 10 data times | 0.309 |
| PINN, physics checked at 50 points | 0.096 |

The main lesson is that *where* you enforce the equation matters. Checking it only at the measurement times leaves the network free to wiggle in between. Checking it on a denser set of times, which costs no extra measurements, gives a much better fit.

This is one noise sample and one random start, so read the table as an illustration and not a benchmark. Part B runs the same comparison in PyTorch; its numbers will differ slightly because of different starting weights and float32 arithmetic.

## How to run it

You need Python 3.9 or newer.

```bash
pip install numpy matplotlib scipy jupyter
pip install torch            # needed for Part B only
pip install deepxde          # optional, only for the last cell
jupyter notebook pinn_1d_ode_ball_throw.ipynb
```

Part A needs only NumPy and Matplotlib. If you do not have PyTorch, you can still run everything up to the end of Part A. You can also open the notebook in Google Colab, which already has PyTorch.

The notebook runs in a few minutes on an ordinary laptop CPU, and no GPU is needed.

## A few things worth knowing

- **Gradient check.** Part A includes a test that compares the hand-written gradients against numerical estimates. They agree to about one part in a million.
- **Loss spikes.** With Adam at learning rate 0.01, the loss occasionally jumps for a short while and then recovers. This is normal for this setting.
- **Random seeds.** Both parts fix their seeds, so reruns give the same numbers on the same machine.

## References

1. M. Raissi, P. Perdikaris, G. E. Karniadakis. *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686–707 (2019).
2. Vizuara AI Labs and S. Panat. *Teach your neural network to "respect" Physics.* Vizuara AI Newsletter, 2025.
3. D. P. Kingma and J. Ba. *Adam: A method for stochastic optimization.* ICLR (2015).
4. L. Lu, X. Meng, Z. Mao, G. E. Karniadakis. *DeepXDE: A deep learning library for solving differential equations.* SIAM Review, 63(1), 208–228 (2021).



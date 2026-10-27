# The Heat Equation with Physics-Informed Neural Networks: Soil Sensors and a Spacecraft Heat Shield

**Author:** Dishanand Jayeprokash

Heat moving through a material follows one of the best-known equations in physics. This project uses it, together with a few temperature readings, to answer two real questions:

1. **Soil.** Four thermometers are buried at different depths. How fast does heat spread through this soil, and what is the temperature at depths where nobody put a sensor?
2. **A spacecraft heat shield.** A returning spacecraft is wrapped in plasma for many minutes. Its outer surface reaches about 1200 °C, while the aluminium structure behind an insulating tile must stay under about 175 °C. How hot does the back of the tile get, and when, if nobody can put a sensor right there?

The two cases are one problem in different clothes: *use a few temperature readings and the physics of heat flow to find an unknown material property and to predict temperatures where nothing was measured.*

The work is a single Jupyter notebook, built twice: once **from scratch with NumPy** so you can see what happens inside, and once with **PyTorch** using the tools people use in practice. It is the first project of the PDE (partial differential equation) series, after the three ODE projects.

<img src="https://github.com/djayepro3/pinns-ode-pde/blob/main/PINN%20PDE/1D%20Heat%20Equation/media/reentry_heat_shield.gif" width="100%">


*Re-entry, minute by minute. Left: the tile, with plasma above it and thermocouples inside. Middle: temperature through the tile. Top right: the hot face and the thermocouples. Bottom right: the back face, where nothing is measured, with the temperature limit. The back face peaks tens of minutes after the surface has cooled ("heat soak").*

<img src="https://github.com/djayepro3/pinns-ode-pde/blob/main/PINN%20PDE/1D%20Heat%20Equation/media/soil_heat_training_pytorch.gif" width="100%">

*Training a network on the soil data, epoch by epoch. Left: a network that sees only the sensor readings. Right: a network that also knows the heat equation and has to find the soil's diffusivity (bottom right) starting from a wrong guess.*

## What is new compared with the ODE projects

- **The unknown depends on two variables**, depth and time, so the network has two inputs.
- **There are many more places to check the equation**: a whole depth–time map, not just a line.
- **The equation has a second derivative in depth and a first derivative in time**, so the network has to be differentiated in two different directions.
- **The surface temperature is a boundary condition**, and in the tile case the back face and the starting state are conditions as well.

## The problems in short

The temperature `T(z, t)` at depth `z` and time `t` obeys

```
dT/dt = alpha * d2T/dz2,        alpha = k / (rho * c)
```

where `alpha` is the **thermal diffusivity**: how quickly heat spreads. The notebook derives this equation from an energy balance and Fourier's law.

**Case 1: soil.** With a clean daily surface cycle, the equation has an exact solution, a damped and delayed wave:

```
T(z, t) = T_mean + A * exp(-z/d) * cos(w t - z/d),        d = sqrt(2 alpha / w)
```

The swing shrinks and arrives later with depth. This is the same "decaying cosine" shape as the damped oscillator, but spread over depth. For our soil (`alpha = 5e-7 m2/s`), the damping depth `d` is 11.7 cm, the swing is only 8% of the surface swing at 30 cm, and its delay there is almost 10 hours.

**Case 2: a heat-shield tile.** A tile 7 cm thick, with a known hot-face temperature history (rising for 4 minutes, holding for 5, falling for 10), an insulated back, and a start at room temperature. The notebook derives the exact solution for a tile of finite thickness (a series of decaying sines, by separation of variables) and shows why the structure behind the tile peaks **after** the heating ends.

The tile numbers are **illustrative**, in the right range for a ceramic insulation tile at room temperature, not data for a real vehicle.

## What is in the notebook

The file is `pinn_heat_equation_soil_and_heat_shield.ipynb`.

| Section | Contents |
|---|---|
| Background | Where the heat equation comes from (fully derived), the exact daily-cycle solution, the classical amplitude and phase methods for reading sensors, thermal inertia (the remote-sensing link), the finite-thickness tile solution, heat soak, a heat-budget check that needs no exact answer, the Crank-Nicolson solver, and how the loss is built for a PDE |
| Setup | Constants, exact solutions, a classical finite-difference solver, interactive explorers for both cases (including "is this tile thick enough?"), synthetic sensor readings, classical estimators, and the video helpers |
| Part A: NumPy only | A network that returns its own time and depth derivatives, gradients derived by hand and tested, Adam. For soil: sensors-only vs physics training, the diffusivity estimate, a heat-budget check, a training video, and a cloudy-afternoon test. **For the spacecraft: the tile's diffusivity from thermocouples, the back-face prediction, an energy check, and the re-entry video** |
| Part B: PyTorch | Automatic differentiation, L-BFGS, `torch.func`, a study of how many sensors you need, and the heat-shield tile again |
| Summary and references | What the experiments showed, the limits of the examples, our own design choices, ideas to try, and the reading list |

## What was found

These numbers are from the NumPy run in Part A. Errors are against the exact (soil) or classical-solver (tile) answer.

### Soil: four sensors at 5, 10, 20 and 30 cm, hourly for 3 days, 0.2 °C noise

| Question | Result |
|---|---|
| Classical methods on 300 repeats with new noise | amplitude method and phase method each wander by about 2.8%; fitting the exact formula by about 0.9% |
| Diffusivity from this run's readings | network 2.5% low; exact-formula fit 2.2% low; amplitude method 4.2% low; phase method 0.4% high |
| Temperature map error, whole depth-time map | sensors-only network 0.76 °C; network with physics 0.046 °C |
| Where the sensors-only network goes wrong | about 0.8 °C above the first sensor and 1.0 °C below the last, and several degrees in the deep corner of the map |
| Heat-budget mismatch (smaller is better) | exact 0.006; network with physics 0.014; sensors-only network 0.245 |
| Cloudy afternoon (messy surface, no exact formula) | network 2.3% low; fit that reruns the classical solver 1.4% low |

**The honest reading.** On clean daily-cosine data the network **matches** the best classical estimate but does not beat it. Its advantage is that it needs no exact formula. With a surface temperature that has a cloud dip, the same code still worked. The simple harmonic methods also still work there, because a two-hour cloud barely disturbs the daily wave.

### Spacecraft tile: thermocouples at 1, 2, 3 and 4 cm, every 30 seconds for an hour, 3 °C noise, nothing at the back face

| Question | Result |
|---|---|
| True back-face peak (classical solver) | 147 °C at minute 43, which is 24 minutes **after** the surface heating ends (a 28 °C margin under the illustrative 175 °C limit) |
| Other thicknesses | 5 cm: 262 °C (too hot). 6 cm: 192 °C (too hot). 8 cm: 118 °C |
| Diffusivity from a classical fit with the solver | 0.24% error, with a spread of about 0.14% over 40 repeats; peak within a fraction of a degree |
| Diffusivity from the network with physics | 0.1% low |
| Temperature map error, whole tile | thermocouples-only network 25 °C; network with physics 3.5 °C |
| Back-face peak | thermocouples-only network 153 °C at minute 36 (wrong shape and about 7 minutes early); network with physics 138 °C at minute 43 |

**The honest reading.** The network with physics gets the diffusivity and the timing of the peak right, and the shape of the whole back-face curve. But it **under-predicts the peak by about 9 °C**, and under-predicting is the unsafe direction when the question is a safety limit. The classical solver fit does better on the peak, though it needs a solver run many times. So in a real design you would add a margin and cross-check the network. What the network offers is that one training run gives both the diffusivity and the whole temperature history, with no solver loop.

These are single runs with fixed seeds (plus the repeat tests for the classical methods), so read them as illustrations and not as benchmarks. Part B repeats the main experiments in PyTorch. Its numbers will differ somewhat, because the starting weights differ and PyTorch works in 32-bit numbers where the NumPy code uses 64-bit ones.

## The videos and GIFs

- **Soil training video** (A11): two networks side by side, with the temperature map, the profile at day 2, and the error and diffusivity estimates as training goes on.
- **Re-entry video** (A13.7): the whole hour of the re-entry. It is built from the networks' own predictions: the tile cross-section, the temperature through the tile, the thermocouple histories, and the back face as it develops, with the limit line and a "heat soak" label.
- Both come with an MP4, a GIF, a still picture for GitHub previews, and a slider (in Jupyter or Colab).

The code that makes them is in the notebook. It writes a GIF with Pillow (always works) and an MP4 with `ffmpeg`. If `ffmpeg` is missing, the MP4 step prints a message and is skipped. You can install `ffmpeg` or run `pip install imageio-ffmpeg`, which the code finds automatically. The files go to a `media` folder next to the notebook.

## How to run it

You need Python 3.9 or newer.

```bash
pip install numpy scipy matplotlib pillow jupyter
pip install ipywidgets       # optional, for the sliders
pip install torch            # needed for Part B only
jupyter notebook pinn_heat_equation_soil_and_heat_shield.ipynb
```

Run the Background and Setup sections first, then Part A or Part B. Part A needs only NumPy, SciPy and Matplotlib, so you can work through all the from-scratch parts without PyTorch. You can also open the notebook in Google Colab, which already has PyTorch.

Rough run times on an ordinary laptop CPU, with no GPU needed:

- Setup and Part A together take about 10 to 15 minutes on a single processor core. The tile network takes the longest, around five minutes.
- Part B takes longer, because differentiating the network twice costs more. Expect something like half an hour or more (this is an estimate and has not been timed).

## A few things worth knowing

- **Gradient checks.** Part A tests the hand-written gradients against numerical estimates, once for the soil loss and once for the tile loss. They agree to about one part in a billion.
- **The tile properties are illustrative.** Real tile properties change a lot with temperature and pressure, and heat also moves by radiation inside a porous tile. The notebook keeps the diffusivity constant, which is what makes the equation linear.
- **The boundary conditions are simplified on purpose.** The tile has a known hot-face temperature, an insulated back (cautious, because the real structure absorbs some heat), and a surface that cools to ambient in ten minutes (optimistic, because a real surface cools more slowly). The soil model holds the temperature at 0.6 m constant, which costs a few hundredths of a degree.
- **The surface temperature is known** in both cases. If it is not, more sensors are needed and the problem gets much harder.
- **Real soil is not uniform.** Water and layers make a single diffusivity only an apparent value, and then the amplitude and phase methods often disagree. The network inherits the same assumption.
- **A classical solver is quicker** for the heat equation when everything is known. The network is worth it when data and physics must be combined, when constants are unknown, or when conditions are messy.
- **Random seeds.** Both parts fix their seeds. PyTorch and NumPy draw different random numbers, so results are close but not identical between the parts.

## Things to try

- Raise the sensor noise, or use fewer readings, and watch the estimates spread.
- Move the soil sensors deeper (20, 30, 40 cm) and see how the amplitude method suffers first.
- Make the soil cycle yearly instead of daily. The damping depth grows by about 19 times, to roughly 2 m.
- Use two layers of soil with different diffusivities and see what the classical methods report.
- Move the tile thermocouples deeper or use fewer of them, and see how the back-face prediction changes.
- Change the tile thickness and see where the design stops being safe (the sliders in Setup S13 make this quick).
- Make the tile's surface cool more slowly after the heating, and see how much stronger the heat soak becomes.

## References

1. H. S. Carslaw and J. C. Jaeger. *Conduction of Heat in Solids.* 2nd edition, Oxford University Press (1959).
2. F. P. Incropera, D. P. DeWitt, T. L. Bergman, A. S. Lavine. *Fundamentals of Heat and Mass Transfer.* 7th edition, Wiley (2011).
3. D. Hillel. *Environmental Soil Physics.* Academic Press (1998).
4. R. Horton, P. J. Wierenga, D. R. Nielsen. *Evaluation of methods for determining the apparent thermal diffusivity of soil near the surface.* Soil Science Society of America Journal, 47(1), 25-32 (1983).
5. J. C. Price. *Thermal inertia mapping: A new view of the Earth.* Journal of Geophysical Research, 82(18), 2582-2590 (1977).
6. J. Crank and P. Nicolson. *A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type.* Proceedings of the Cambridge Philosophical Society, 43(1), 50-67 (1947).
7. J. V. Beck, B. Blackwell, C. R. St. Clair Jr. *Inverse Heat Conduction: Ill-Posed Problems.* Wiley (1985).
8. O. M. Alifanov. *Inverse Heat Transfer Problems.* Springer (1994).
9. S. D. Williams and D. M. Curry. *Thermal Protection Materials: Thermophysical Property Data.* NASA Reference Publication 1289 (1992).
10. M. Raissi, P. Perdikaris, G. E. Karniadakis. *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686-707 (2019).
11. D. P. Kingma and J. Ba. *Adam: A method for stochastic optimization.* ICLR (2015).

The notebook's own reference list has the full set.
# Method and file guide

## Dimensionless inputs

- `Nw`: initial plume standard deviation divided by grid spacing.
- `Pe`: grid Péclet number.
- `CFL`: time-step factor applied to the combined explicit stability limit.
- `tau`: dimensionless transport time.

## Calibration outputs

The Gaussian benchmark relates peak attenuation to an equivalent numerical
diffusion coefficient. Only cases that satisfy the prescribed finiteness,
mass-balance, boundary-clearance, phase, and profile-error checks are used to
fit a positive-diffusion envelope. Compression, anti-diffusion, and invalid
cases remain part of the reliability assessment but are not treated as positive
numerical diffusion.

For scheme `s`, the fitted upper envelope defines the risk index

```text
RND_s = kappa_s,95(Nw) * Pe.
```

`RND_s = 1` means that the envelope-estimated numerical diffusion is equal in
magnitude to the physical diffusion.

## External validation

The top-hat and cosine-bell cases use a convolution solution of the prescribed
initial profile and the advection–diffusion Green function as the reference.
The Gaussian envelope is queried before the numerical result is evaluated.
Peak, phase, and profile errors are subsequently used to rank the candidate
schemes with the same score definition used by the atlas.

## Profile files

- `reach_C_profile.csv`: numerical concentration history and final profile.
- `reach_analytical_profile.csv`: analytical or convolution reference profile.
- `mass_conservation.csv`: mass-balance history.
- `reach_waterlevel_profile.csv`: hydraulic state used by the solver.

The example folders preserve the original output format so that readers can
inspect the same data structure used by the analysis code.


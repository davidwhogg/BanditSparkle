# BanditSparkle
in bad data, even incoherent modes look coherent

## Authors
Feeney & Hogg

## License
Copyright 2017 the authors. **All rights reserved**. Everything will be released under a sensible license, and soon.

## Abstract
Stellar oscillations are stochastically driven high-Q modes. If a star is observed through an observing campaign that is shorter than the coherence time of a mode, the mode will effectively appear coherent within that time interval. Here we exploit this to create an inexpensive probabilistic model for asteroseismology data. The huge advantage of this approach to traditional asteroseismology approaches is that it does not require the performance of a fourier transform or anything like it; this permits the measurement of asteroseismic parameters in data sets that are irregularly or sparsely sampled, or in data sets where the critical modes are near the edges of the available frequency space (near the inverse exposure time at the high-frequency end or near the inverse of the duration of the campaign at the low-freqency end). It also permits the construction of a tractable and justifiable likelihood function, even when the noise processes are non-trivial. We produce a parameterized description of an asteroseismic frequency spectrum and use it along with this likelihood function to produce posterior estimates of asteroseismic stars in <i>Kepler</i> data. We sub-sample the data to (short duration) <i>TESS</i>-like and (sparsely sampled) <i>Gaia</i>-like data sets and show that even in these much smaller data sets, we ought to be able to estimate asteroseismic parameters for many stars, and (because we are just turning the Bayesian crank) without human intervention.

## Notes
- We have actually done *nothing* at this point; this is all vapor-ware.
- There is another paper to write about variable exposure time, and why <i>TESS</i> is making a huge mistake by using a uniform time sampling.

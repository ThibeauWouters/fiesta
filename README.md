# fiesta 🎉

`fiesta`: **F**ast **I**nference of **E**lectromagnetic **S**ignals and **T**ransients with j**A**x

![fiesta logo](docs/fiesta_logo.jpeg)

**NOTE:** `fiesta` is currently under development -- stay tuned!

## Installation

pip installation is currently work in progress. Install from source by cloning this Github repository and running
```
pip install -e .
```

NOTE: This is using an older and custom version of `flowMC`. Install by cloning the `flowMC` version at [this fork](https://github.com/ThibeauWouters/flowMC/tree/fiesta) (branch `fiesta`).

## Training surrogate models

To train your own surrogate models, have a look at some of the example scripts in the repository for inspiration, under `trained_models`

- `train_Bu2019lm.py`: Example script showing how to train a surrogate model for the POSSIS `Bu2019lm` kilonova model. 
- `train_afterglowpy_tophat.py`: Example script showing how to train a surrogate model for `afterglowpy`, using a tophat jet structure.  

## Examples

- `run_AT2017gfo_Bu2019lm.py`: Example where we infer the parameters of the AT2017gfo kilonova with the `Bu2019lm` model.
- `run_GRB170817_tophat.py`: Example where we infer the parameters of the GRB170817 GRB with a surrogate model for `afterglowpy`'s tophat jet. **NOTE** This currently only uses one specific filter. The complete inference will be updated soon.
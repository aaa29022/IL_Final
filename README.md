# IL_Final

## Contributors

Angran Li, Yin-Hung Chen

## Improving the Dynamics of MGAIL with Ensemble and Data Reweighting

Extension of ICML 2017 paper "End-to-End Differentiable Adversarial Imitation Learning", by Nir Baram, Oron Anschel, Itai Caspi, Shie Mannor.

## Dependencies
* Gym >= 0.8.1
* Mujoco-py >= 0.5.7
* Tensorflow >= 1.0.1

## Running
Run the following command to train the Mujoco Hopper environment by imitating an expert trained with TRPO

```python
# Run the original MGAIL
python3 main.py

# Run with data reweighting
python3 main.py -r

# Run with 5 ensemble models
python3 main.py -e 5

# Run with data reweighting and 10 ensemble models
python3 main.py -r -e 10
```

# SLED
Self-Labelled Encoder-Decoder (SLED) for myelin water imaging data analysis

## How SLED works:
- Data: multi-echo gradient echo-based (mGRE) myelin water imaging (MWI) data
- Encoder: a series of neural networks to estimate latent parameters such as T<sub>2</sub> or T<sub>2</sub><sup>*</sup> times and amplitudes
- Decoder: a typical 3-pool model (myelin, axonal, and free water pools)
- Training: exclusively trained for each dataset which is self-labelled

<img width="862" alt="sled_schematics" src="sled_schematics.png">

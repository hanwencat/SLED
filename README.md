# SLED
Self-Labelled Encoder-Decoder (SLED) for myelin water imaging data analysis

## How SLED works:
- Data: multi-echo gradient echo-based (mGRE) myelin water imaging (MWI) data
- Encoder: a series of neural networks to estimate latent parameters such as T<sub>2</sub> or T<sub>2</sub><sup>*</sup> times and amplitudes
- Decoder: a typical 3-pool model (myelin, axonal, and free water pools)
- Training: exclusively trained for each dataset which is self-labelled

<img width="750" alt="sled_schematics" src="/sled_schematics.png">

## File structure
- *configs*: configuration files to initialize all fitting parameters for SLED fitting
- *data*: example data is stored here
- *models*: trained model (the best epoch) is saved here
- *results*: parameter maps are saved here
- *src*: all source code is placed here
    - *models*: building blocks for the SLED model and the code for model training
    - *utility*: utility functions for image processing and customized tensorflow losses
    - *main.py*: entry point to the SLED fitting

## Run SLED fitting
- clone this repo to your local computer
    ```
    git clone git@github.com:hanwencat/SLED_mese.git
    ```
- nevigate to the root of this repo
- use conda to create a virtual environment for the SLED fitting
    ```
    conda env create -n sled -f environment.yml
    conda activate sled
    ```
- change the config file (e.g. the data_path in the *configs/defaults.yml*) as needed
- run the *src/main.py* to fit
    ```
    python src/main.py
    ```
- fitting is logged in *logs/training.log*
- fitted maps are saved in the *result/* folder and all fitting info are stored as nifty header extension
    - *mwf.nii.gz*: the fitted MWF map
    - *t2s.nii.gz*: the fitted t2 times for the 3 pools (4D data)
    - *amps.nii.gz*: the fitted amplitudes for the 3 pools (4D data)

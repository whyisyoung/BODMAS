# BODMAS Malware Dataset

## Introduction
The BODMAS Malware Dataset is created and maintained by [Blue Hexagon](https://bluehexagon.ai/) and [UIUC](https://illinois.edu/).

It contains 57,293 malware and 77,142 benign Windows PE files, including binaries (disarmed malware only), feature vectors, and metadata.

Further details can be found in our paper “BODMAS: An Open Dataset for Learning based Temporal Analysis of PE Malware” [[PDF](https://liminyang.web.illinois.edu/data/DLS21_BODMAS.pdf)], Deep Learing and Security Workshop 2021 (co-located with IEEE Security and Privacy 2021).

If you end up building on this dataset as part of a project or publication, please include a reference to our paper:

```
@inproceedings{bodmas,
  title = {BODMAS: An Open Dataset for Learning based Temporal Analysis of PE Malware},
  author = {Yang, Limin and Ciptadi, Arridhana and Laziuk, Ihar and Ahmadzadeh, Ali and Wang, Gang},
  booktitle = {4th Deep Learning and Security Workshop},
  year = {2021}
}
```

## Download
Please visit [this link](https://whyisyoung.github.io/BODMAS/) for more details.

## Installation
1. Before we get started, please check your server storage and memory. I ran most of the experiments on our lab clusters containing 9 servers (see specification [here](https://gangw.cs.illinois.edu/cluster.html)). I use Fabric to distribute code to different servers to simplify repetitive experiments. You can use 1 server, but you need to change some shell scripts, see the Examples section.

2. Clone this repo to your home directory (you can save to other directories but you need to change some scripts if you did, see the warning in the Examples section:
    ```bash
    cd ~
    git clone git@github.com:whyisyoung/BODMAS.git
    ```

3. We recommend setting up a Python 3.6.8 virtual environment (other Python 3.6 or above versions might also work but didn't test).

    ```bash
    cd BODMAS/code/
    pip install requirements.txt
    python setup.py install
    ```

## Configuration
1. For BODMAS, follow the guidelines of the Download section. Put `bluehex_metadata.csv` and `bluehex.npz` under `BODMAS/code/multiple_data/`.
2. For Ember and UCSB-packerware, you can download pre-processed feature vectors and metadata here (about 3.4 GB in total): [Google Drive link](https://drive.google.com/drive/folders/12DMPeh8DA2ukPATnHX4K__shWFJIiBN5?usp=sharing). Note for Ember, we combine Ember 2017 and 2018 as a whole. Put the 4 downloaded files under `BODMAS/code/multiple_data/`.

3. For SOREL-20M, you can download pre-trained LightGBM and DNN models here: [https://github.com/sophos-ai/SOREL-20M](https://github.com/sophos-ai/SOREL-20M)
If you want to use pretrained SOREL-20M models, you need to specify your locations for some folders in `code/bodmas/config.py`:

```python
    'sophos_model_folder': '/home/datashare/sophos/baselines/checkpoints/lightGBM/',
    'sophos_features_folder': '/home/datashare/sophos/lightGBM-features/'
```

## Examples

### Testing pre-trained models on our BODMAS dataset (Table II in our paper):
1. Using Ember and random seed 0 as the training set (**PLEASE change the hostname of "angel" to yours**):
    ```bash
    cd BODMAS/code/
    ./main_pretrain.sh
    ```

    For other random seeds (1-4), uncomment the rest of the first code block of `main_pretrain.sh`, also change the hostname of ("beast" "bishop" "colossus" "cyclops") to yours. It would be highly recommended to run only 1 random seed each time if you don't have enough memory.

    Call graph:
    ```bash
    main_pretrain.sh -> fabric_pretrain.py -> run_pretrain.sh -> pretrain_model_test_on_bluehex.py
    ```

    WARNING: If you didn't put this repo under your home directory (i.e., this repo would appear as `~/BODMAS`), you might need to change the line 18 of `fabric_pretrain.py`. This also applies to `fabric_multiclass.py` (line 17)

2. Using Sophos pre-trained models, uncomment the second code block of `main_pretrain.sh` and change the hostname to yours. Using UCSB as the training set, uncomment the third code block of `main_pretrain.sh` and change the hostname accordingly. Code for Sophos-DNN is very similar thus omitted here.

### Incremental Retraining (Fig.1 in our paper)
1. Before running the script, if you want to test Transcend, you need to ask for access to the Transcend code from Feargus Pendlebury and Lorenzo Cavallaro (https://s2lab.cs.ucl.ac.uk/) . **Please CC me as well.** Otherwise you can uncomment the corresponding import and related code.

2. Use corresponding code blocks and change the hostname to yours accordingly:
    ```bash
    ./run_ember_drift.sh
    ```

3. Call graph:
    ```bash
    run_ember_drift.sh -> concept_drift_ember.py
    ```

### Training with New Data (Fig. 2 in our paper)
1. Change the hostname to yours accordingly and run the following script. It's highly recommend to run each random seed sequentially to avoid memory error unless you can run them on multiple servers.
    ```bash
    ./main_bluehex_binary.sh
    ```

2. Call graph:
    ```bash
    main_bluehex_binary.sh -> bluehex_main.py
    ```


### Multi-class classification (Fig. 3, 4 in our paper)
1. Use corresponding code blocks and change the hostname to yours accordingly:
    ```bash
    ./main_bluehex_multiclass.sh
    ```

    Call graph:
    ```bash
    main_bluehex_multiclass.sh -> fabric_multiclass.py -> run_multiclass.sh -> bluehex_main.py
    ```

## Contact
If you have any questions, please contact Limin (liminy2@illinois.edu).

## Licensing
BSD 2-Clause "Simplified" License.

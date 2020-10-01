# CADE: Contrastive Autoencoder for Drifting detection and Explanation

The repository contains the code for detecting and explaining a specific type of concept drift (i.e., previously unseen families) in security applications like malware attribution and network intrusion classification.

Further details can be found in the paper "*CADE: Detecting and Explaining Concept Drift Samples for Security Applications*" by Limin Yang, Wenbo Guo, Qingying Hao, Arridhana Ciptadi, Ali Ahmadzadeh, Xinyu Xing, Gang Wang (USENIX Security 2021). We also include supplemental materials in the repo (`USENIX_21_drifting_Supplementary_Materials.pdf`) due to page limit.   Check out http://liminyang.web.illinois.edu for up-to-date information on the project.

If you end up building on this research or code as part of a project or publication, please include a reference to the USENIX Security paper:
```
@inproceedings{yang2021cade,
    title = {CADE: Detecting and Explaining Concept Drift Samples for Security Applications},
    author = {Limin Yang, Wenbo Guo, Qingying Hao, Arridhana Ciptadi, Ali Ahmadzadeh, Xinyu Xing, Gang Wang},
    booktitle = {Proc. of USENIX Security},
    year = {2021}
}
```

## 1. Installation

Before getting started we recommend setting up a Python 3.6.5 or 3.6.8 virtual environment (other Python 3.6 or above versions might also work but didn't test).

* If you are using CPU-based tensorflow, install all required packages:

  ```bash
  pip install -r requirements-tensorflow-cpu.txt
  python setup.py install
  ```

* If you are using GPU-based tensorflow, please try the following steps to setup:

  ```bash
  module load cuda-toolkit/9.0  # other versions might also work but didn't test
  # you may also try pyenv and virtualenv to create the virtual environment, here we use Anaconda
  conda create -n cade-gpu python=3.6.8
  conda activate cade-gpu
  pip install scipy==1.3.3
  pip install numpy==1.16.1
  pip install --ignore-installed tensorflow-gpu==1.12.0
  pip install keras==2.2.5
  pip install sklearn==0.23.2
  pip install matplotlib==3.1.2
  pip install seaborn==0.11.0
  pip install tqdm==4.49.0
  python setup.py install
  ```





## 2. Configuration

The preprocessed Drebin and IDS2018 dataset can be found under the `data` folder. If you prefer to modify the preprocessing step, you may download the original dataset here: https://www.sec.cs.tu-bs.de/~danarp/drebin/index.html and https://www.unb.ca/cic/datasets/ids-2018.html and fill out the configuration in `cade/config.py`.





## 3. Usage

There are a number of command line arguments to run our program:

```bash
$ python main.py -h
usage: main.py [-h] [--data DATA] [-c {mlp,rf}] [--stage {detect,explanation}]
               [--pure-ae {0,1}] [--quiet {0,1}] [--cae-hidden CAE_HIDDEN]
               [--cae-batch-size CAE_BATCH_SIZE] [--cae-lr CAE_LR]
               [--cae-epochs CAE_EPOCHS] [--cae-lambda-1 CAE_LAMBDA_1]
               [--similar-ratio SIMILAR_RATIO] [--margin MARGIN]
               [--display-interval DISPLAY_INTERVAL]
               [--mad-threshold MAD_THRESHOLD]
               [--exp-method {distance_mm1,approximation_loose}]
               [--exp-lambda-1 EXP_LAMBDA_1] [--mlp-retrain {0,1}]
               [--mlp-hidden MLP_HIDDEN] [--mlp-batch-size MLP_BATCH_SIZE]
               [--mlp-lr MLP_LR] [--mlp-epochs MLP_EPOCHS]
               [--mlp-dropout MLP_DROPOUT] [--newfamily-label NEWFAMILY_LABEL]
               [--tree TREE] [--rf-retrain {0,1}]
```

See `cade/utils.py` or run `python main.py -h` for detailed help. You may also check `run_drebin_cade.sh` for a bunch of examples.





## 4. Examples

### 4.1 Drift detection

1. To get the detection performance of CADE on the Drebin dataset (iteratively choose one family from 8 families as the unseen family):

   ```bash
   ./run_drebin_cade.sh

   # After the shell script finished running
   python -u average_all_detection_results.py drebin 0
   # 0 means using CADE, while 1 means using Vanilla AE
   ```

2. To get the detection performance of CADE on the IDS2018 dataset (iteratively choose one family from 3 families as the unseen family):

   ```bash
   ./run_ids_cade.sh

   # After the shell script finished running
   python -u average_all_detection_results.py IDS 0
   ```

3. To get the detection performance of Vanilla Autoencoder on the Drebin dataset:

   ```bash
   ./run_drebin_pure_ae.sh

   # After the shell script finished running
   python -u average_all_detection_results.py drebin 1
   ```

4. To get the detection performance of Vanilla Autoencoder on the IDS2018 dataset:

   ```bash
   ./run_ids_pure_ae.sh

   # After the shell script finished running
   python -u average_all_detection_results.py IDS 1
   ```



### 4.2 Drift explanation

1. CADE explaining drift samples on the Drebin-Fakedoc setting (i.e., drebin_new_7):

   ```bash
   ./run_cade_exp_drebin_fakedoc.sh
   # It will generate reports/drebin_new_7/mask_distance_mm1_0.001.npz,
   # which is already provided.
   # This step is time-consuming and non-deterministic,
   # so we include the explanation output for saving reproduction time and easier comparison.
   ```

2. CADE explaining drift samples on the IDS2018-Infiltration setting:

   ```bash
   ./run_cade_exp_ids_infiltration.sh
   # It will generate reports/IDS_new_Infilteration/mask_distance_mm1_0.001.npz,
   # which is already provided.
   ```

3. Boundary-based explanation on the Drebin-Fakedoc setting:

   ```bash
   ./run_boundary_exp_drebin_fakedoc.sh
   # It will generate reports/drebin_new_7/mask_approximation_loose_0.001.npz,
   # which is already provided.
   ```

4. Boundary-based explanation on the IDS2018-Infiltration setting:

   ```bash
   ./run_boundary_exp_ids_infiltration.sh
   # It will generate reports/IDS_new_Infilteration/mask_approximation_loose_0.001.npz,
   # which is already provided.
   ```

5. Compare CADE with boundary-based explanation and random explanation (using distance as the evaluation metric)

   1. Drebin-FakeDoc

   ```bash
   # 1. To get original distance and CADE distance
   python -u evaluate_explanation_by_distance.py drebin_new_7 distance_mm1 0.001 1 0.1

   # 2. To get random explanation distance
   python -u evaluate_explanation_by_distance.py drebin_new_7 random 0.001 0 0.1
   # since we randomly run 100 times, there might be minor difference on the output.

   # 3. To get boundary-based explanation distance
   python -u evaluate_explanation_by_distance.py drebin_new_7 approximation_loose 0.001 0 0.1

   # 4. To get gradient-based explanation distance
   nohup python -u evaluate_explanation_by_distance.py drebin_new_7 gradient 0.001 0 0.1 \
   > logs/nohup-drebin_new_7-gradient-exp.log &
   ```

   2. IDS2018-Infiltration

   ```bash
   # 1. To get original distance and CADE distance
   nohup python -u evaluate_explanation_by_distance.py IDS_new_Infilteration distance_mm1 \
   0.001 1 0.1 > logs/nohup-IDS-distance-mm1-exp.log &

   # 2. To get random explanation distance
   nohup python -u evaluate_explanation_by_distance.py IDS_new_Infilteration random \
   0.001 0 0.1 > logs/nohup-IDS-random-exp.log &
   # since we randomly run 100 times, there might be minor difference on the output.

   # 3. To get boundary-based explanation distance
   nohup python -u evaluate_explanation_by_distance.py IDS_new_Infilteration \
   approximation_loose 0.001 0 0.1 > logs/nohup-IDS-boundary-exp.log &

   # 4. To get gradient-based explanation distance
   nohup python -u evaluate_explanation_by_distance.py IDS_new_Infilteration gradient \
   0.001 0 0.1 > logs/nohup-IDS-gradient-exp.log &
   ```





## 5. Contact

If you have any questions, please contact Limin (liminy2@illinois.edu).





## 6. Licensing

For ethical considerations, code and data is covered by a modified BSD 3-Clause License which restricts the use of the code to academic purposes and which specifically prohibits commercial applications.

> Any redistribution or use of this software must be limited to the purposes of non-commercial scientific research or non-commercial education. Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.


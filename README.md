# Seizure Detection

This repository contains the winning submission for UPenn and Mayo Clinic's Seizure Detection Challenge on Kaggle.

It has been updated as of 18 Feb 2018 to include a requirements.txt for installing dependencies via pip. Small changes
were necessary to get the repo working again with updated libraries. These changes were tested using `sample_clip.mat`
from the Kaggle competition data rather than the original data which I did not have on hand. For the original submission
without these changes, you may checkout the tag `original-submission`, e.g. `git checkout original-submission`.

http://www.kaggle.com/c/seizure-detection

This README and repository modelled on https://www.kaggle.com/wiki/ModelSubmissionBestPractices

## Hardware / OS platform used

 * 15" Retina MacBook Pro (Late 2013) 2.7GHz Core i7, 16GB RAM
 * OS X Mavericks

## Dependencies

### Required

 * python 2.7 with virtualenv
 * hickle==2.1.0
 * numpy==1.14.0
 * scikit-learn==0.19.1
 * scipy==1.0.0

### Optional (to try out various data transforms)

 * pywt (for Daubechies wavelet)
 * scikits talkbox (for MFCC)

### Installing dependencies

Setup the virtualenv and install the dependencies

```
virtualenv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model and make predictions

Activate the virtualenv

```
. venv/bin/activate
```

Obtain the competition data and place it in the root directory of the project.
```
seizure-data/
  Dog_1/
    Dog_1_ictal_segment_1.mat
    Dog_1_ictal_segment_2.mat
    ...
    Dog_1_interictal_segment_1.mat
    Dog_1_interictal_segment_2.mat
    ...
    Dog_1_test_segment_1.mat
    Dog_1_test_segment_2.mat
    ...

  Dog_2/
  ...
```

The directory name of the data should match the value in SETTINGS.json under the key `competition-data-dir`.

Then simply run:
```
python -m train
```

One classifier is trained for each patient, and dumped to the data-cache directory.

```
data-cache/classifier_Dog_1_fft-with-time-freq-corr-1-48-r400-usf-gen1_rf3000mss1Bfrs0.pickle
data-cache/classifier_Dog_2_fft-with-time-freq-corr-1-48-r400-usf-gen1_rf3000mss1Bfrs0.pickle
...
data-cache/classifier_Patient_8_fft-with-time-freq-corr-1-48-r400-usf-gen1_rf3000mss1Bfrs0.pickle
```

Although using these classifiers outside the scope of this project is not very straightforward.

More convenient is to run the predict script.

```
python -m predict
```

This will take at least 2 hours. Feel free to update the classifier's `n_jobs` parameter
in `seizure_detection.py`.

A submission file will be created under the directory specified by the `submission-dir` key
in `SETTINGS.json` (default `submissions/`).

Predictions are made using the test segments found in the competition data directory. They
are iterated over starting from 1 counting upwards until no file is found.

i.e.
```
seizure-data/
  Dog_1/
    Dog_1_test_segment_1.mat
    Dog_1_test_segment_2.mat
    ...
    Dog_1_test_segment_3181.mat
```

To make predictions on a new dataset, simply replace these test segments with new ones.
The files must numbered sequentially starting from 1 otherwise it will not find all of
the files.

This project uses a custom task system which caches task results to disk using hickle format and
falling back to pickle. First a task's output will be checked if it is in the data cache on disk,
and if not the task will be executed and the data cached.

See `seizure/tasks.py` for the custom tasks defined for this project. More specifically the
`MakePredictionsTask` depends on `TrainClassifierTask`, which means `predict.py` will train
and dump the classifiers as well as make predictions.

## Run cross-validation

```
python -m cross_validation
```

Cross-validation set is obtained by splitting on entire seizures. For example if there are 4 seizures,
3 seizures are used for training and 1 is used for cross-validation.


## SETTINGS.json

```
{
  "competition-data-dir": "seizure-data",
  "data-cache-dir": "data-cache",
  "submission-dir": "./submissions"
}
```

* `competition-data-dir`: directory containing the downloaded competition data
* `data-cache-dir`: directory the task framework will store cached data
* `submission-dir`: directory submissions are written to


## Model documentation

Available at https://github.com/MichaelHills/seizure-detection/raw/master/seizure-detection.pdf

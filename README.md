# Seizure Detection

This repository contains the winning submission for UPenn and Mayo Clinic's Seizure Detection Challenge on Kaggle.

http://www.kaggle.com/c/seizure-detection

This README and repository modelled on https://www.kaggle.com/wiki/ModelSubmissionBestPractices

##Hardware / OS platform used

 * 15" Retina MacBook Pro (Late 2013) 2.7GHz Core i7, 16GB RAM
 * OS X Mavericks

##Dependencies

###Required

 * Python 2.7
 * scikit_learn-0.14.1
 * numpy-1.8.1
 * pandas-0.14.0
 * scipy
 * hickle (plus h5py and hdf5, see https://github.com/telegraphic/hickle for installation details)

###Optional (to try out various data transforms)

 * pywt (for Daubechies wavelet)
 * scikits talkbox (for MFCC)

##Train the model and make predictions

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
./train.py
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
./predict.py
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
./cross_validation.py
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

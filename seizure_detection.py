import json
import os.path
import numpy as np
from common import time
from common.data import CachedDataLoader, makedirs
from common.pipeline import Pipeline
from seizure.transforms import FFT, Slice, Magnitude, Log10, FFTWithTimeFreqCorrelation, MFCC, Resample, Stats, \
    DaubWaveletStats, TimeCorrelation, FreqCorrelation, TimeFreqCorrelation
from seizure.tasks import TaskCore, CrossValidationScoreTask, MakePredictionsTask, TrainClassifierTask
from seizure.scores import get_score_summary, print_results

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def run_seizure_detection(build_target):
    """
    The main entry point for running seizure-detection cross-validation and predictions.
    Directories from settings file are configured, classifiers are chosen, pipelines are
    chosen, and the chosen build_target ('cv', 'predict', 'train_model') is run across
    all combinations of (targets, pipelines, classifiers)
    """

    with open('SETTINGS.json') as f:
        settings = json.load(f)

    data_dir = str(settings['competition-data-dir'])
    cache_dir = str(settings['data-cache-dir'])
    submission_dir = str(settings['submission-dir'])

    makedirs(submission_dir)

    cached_data_loader = CachedDataLoader(cache_dir)

    ts = time.get_millis()

    targets = [
        'Dog_1',
        'Dog_2',
        'Dog_3',
        'Dog_4',
        'Patient_1',
        'Patient_2',
        'Patient_3',
        'Patient_4',
        'Patient_5',
        'Patient_6',
        'Patient_7',
        'Patient_8'
    ]
    pipelines = [
        # NOTE(mike): you can enable multiple pipelines to run them all and compare results
        # Pipeline(gen_ictal=False, pipeline=[FFT(), Slice(1, 48), Magnitude(), Log10()]),
        # Pipeline(gen_ictal=False, pipeline=[FFT(), Slice(1, 64), Magnitude(), Log10()]),
        # Pipeline(gen_ictal=False, pipeline=[FFT(), Slice(1, 96), Magnitude(), Log10()]),
        # Pipeline(gen_ictal=False, pipeline=[FFT(), Slice(1, 128), Magnitude(), Log10()]),
        # Pipeline(gen_ictal=False, pipeline=[FFT(), Slice(1, 160), Magnitude(), Log10()]),
        # Pipeline(gen_ictal=False, pipeline=[FFT(), Magnitude(), Log10()]),
        # Pipeline(gen_ictal=False, pipeline=[Stats()]),
        # Pipeline(gen_ictal=False, pipeline=[DaubWaveletStats(4)]),
        # Pipeline(gen_ictal=False, pipeline=[Resample(400), DaubWaveletStats(4)]),
        # Pipeline(gen_ictal=False, pipeline=[Resample(400), MFCC()]),
        # Pipeline(gen_ictal=False, pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'us')]),
        # Pipeline(gen_ictal=True,  pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'us')]),
        Pipeline(gen_ictal=False, pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'usf')]), # winning submission
        # Pipeline(gen_ictal=True,  pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'usf')]), # higher score than winning submission
        # Pipeline(gen_ictal=False, pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'none')]),
        # Pipeline(gen_ictal=True,  pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'none')]),
        # Pipeline(gen_ictal=False, pipeline=[TimeCorrelation(400, 'usf', with_corr=True, with_eigen=True)]),
        # Pipeline(gen_ictal=False, pipeline=[TimeCorrelation(400, 'us', with_corr=True, with_eigen=True)]),
        # Pipeline(gen_ictal=False, pipeline=[TimeCorrelation(400, 'us', with_corr=True, with_eigen=False)]),
        # Pipeline(gen_ictal=False, pipeline=[TimeCorrelation(400, 'us', with_corr=False, with_eigen=True)]),
        # Pipeline(gen_ictal=False, pipeline=[TimeCorrelation(400, 'none', with_corr=True, with_eigen=True)]),
        # Pipeline(gen_ictal=False, pipeline=[FreqCorrelation(1, 48, 'usf', with_corr=True, with_eigen=True)]),
        # Pipeline(gen_ictal=False, pipeline=[FreqCorrelation(1, 48, 'us', with_corr=True, with_eigen=True)]),
        # Pipeline(gen_ictal=False, pipeline=[FreqCorrelation(1, 48, 'us', with_corr=True, with_eigen=False)]),
        # Pipeline(gen_ictal=False, pipeline=[FreqCorrelation(1, 48, 'us', with_corr=False, with_eigen=True)]),
        # Pipeline(gen_ictal=False, pipeline=[FreqCorrelation(1, 48, 'none', with_corr=True, with_eigen=True)]),
        # Pipeline(gen_ictal=False, pipeline=[TimeFreqCorrelation(1, 48, 400, 'us')]),
        # Pipeline(gen_ictal=False, pipeline=[TimeFreqCorrelation(1, 48, 400, 'usf')]),
        # Pipeline(gen_ictal=False, pipeline=[TimeFreqCorrelation(1, 48, 400, 'none')]),
    ]
    classifiers = [
        # NOTE(mike): you can enable multiple classifiers to run them all and compare results
        # (RandomForestClassifier(n_estimators=50, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf50mss1Bfrs0'),
        # (RandomForestClassifier(n_estimators=150, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf150mss1Bfrs0'),
        # (RandomForestClassifier(n_estimators=300, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf300mss1Bfrs0'),

        # NOTE(mike): The original submission classifier was min_samples_split=1, but I had to change it to 2 after upgrading scikit.
        # I'm not even sure min_samples_split=1 makes sense in hindsight, how can you split on 1 sample? Anyway to get the repo functional
        # again with newer libraries it's now 2.
        # (RandomForestClassifier(n_estimators=3000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf3000mss1Bfrs0'),
        (RandomForestClassifier(n_estimators=3000, min_samples_split=2, bootstrap=False, n_jobs=4, random_state=0), 'rf3000mss2Bfrs0'),
    ]
    cv_ratio = 0.5

    def should_normalize(classifier):
        clazzes = [LogisticRegression]
        return np.any(np.array([isinstance(classifier, clazz) for clazz in clazzes]) == True)

    def train_full_model(make_predictions):
        for pipeline in pipelines:
            for (classifier, classifier_name) in classifiers:
                print 'Using pipeline %s with classifier %s' % (pipeline.get_name(), classifier_name)
                guesses = ['clip,seizure,early']
                classifier_filenames = []
                for target in targets:
                    task_core = TaskCore(cached_data_loader=cached_data_loader, data_dir=data_dir,
                                         target=target, pipeline=pipeline,
                                         classifier_name=classifier_name, classifier=classifier,
                                         normalize=should_normalize(classifier), gen_ictal=pipeline.gen_ictal,
                                         cv_ratio=cv_ratio)

                    if make_predictions:
                        predictions = MakePredictionsTask(task_core).run()
                        guesses.append(predictions.data)
                    else:
                        task = TrainClassifierTask(task_core)
                        task.run()
                        classifier_filenames.append(task.filename())

                if make_predictions:
                    filename = 'submission%d-%s_%s.csv' % (ts, classifier_name, pipeline.get_name())
                    filename = os.path.join(submission_dir, filename)
                    with open(filename, 'w') as f:
                        print >> f, '\n'.join(guesses)
                    print 'wrote', filename
                else:
                    print 'Trained classifiers ready in %s' % cache_dir
                    for filename in classifier_filenames:
                        print os.path.join(cache_dir, filename + '.pickle')

    def do_cross_validation():
        summaries = []
        for pipeline in pipelines:
            for (classifier, classifier_name) in classifiers:
                print 'Using pipeline %s with classifier %s' % (pipeline.get_name(), classifier_name)
                scores = []
                S_scores = []
                E_scores = []
                for target in targets:
                    print 'Processing %s (classifier %s)' % (target, classifier_name)

                    task_core = TaskCore(cached_data_loader=cached_data_loader, data_dir=data_dir,
                                         target=target, pipeline=pipeline,
                                         classifier_name=classifier_name, classifier=classifier,
                                         normalize=should_normalize(classifier), gen_ictal=pipeline.gen_ictal,
                                         cv_ratio=cv_ratio)

                    data = CrossValidationScoreTask(task_core).run()
                    score = data.score

                    scores.append(score)

                    print '%.3f' % score, 'S=%.4f' % data.S_auc, 'E=%.4f' % data.E_auc
                    S_scores.append(data.S_auc)
                    E_scores.append(data.E_auc)

                if len(scores) > 0:
                    name = pipeline.get_name() + '_' + classifier_name
                    summary = get_score_summary(name, scores)
                    summaries.append((summary, np.mean(scores)))
                    print summary
                if len(S_scores) > 0:
                    name = pipeline.get_name() + '_' + classifier_name
                    summary = get_score_summary(name, S_scores)
                    print 'S', summary
                if len(E_scores) > 0:
                    name = pipeline.get_name() + '_' + classifier_name
                    summary = get_score_summary(name, E_scores)
                    print 'E', summary

            print_results(summaries)

    if build_target == 'cv':
        do_cross_validation()
    elif build_target == 'train_model':
        train_full_model(make_predictions=False)
    elif build_target == 'make_predictions':
        train_full_model(make_predictions=True)
    else:
        raise Exception("unknown build target %s" % build_target)

import numpy as np

# helper methods for printing scores

def get_score_summary(name, scores):
    summary = '%.3f-%.3f (mean=%.5f std=%.5f)' % (min(scores), max(scores), np.mean(scores), np.std(scores))
    score_list = ['%.3f' % score for score in scores]
    return '%s %s [%s]' % (name, summary, ','.join(score_list))


def print_results(summaries):
    summaries.sort(cmp=lambda x,y: cmp(x[1], y[1]))
    if len(summaries) > 1:
        print 'summaries'
        for s, mean in summaries:
            print s

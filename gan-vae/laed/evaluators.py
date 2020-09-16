# @Time    : 9/25/17 3:54 PM
# @Author  : Tiancheng Zhao
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from laed.utils import get_dekenize, get_tokenize
from scipy.stats import gmean
import logging
from laed.dataset.corpora import EOS, BOS, SYS, USR
from collections import defaultdict
import sqlite3
from nltk.util import ngrams
import math
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


class EvaluatorBase(object):
    def initialize(self):
        raise NotImplementedError

    def add_example(self, ref, hyp, domain='default'):
        raise NotImplementedError

    def get_report(self, include_error=False):
        raise NotImplementedError

    @staticmethod
    def _get_prec_recall(tp, fp, fn):
        precision = tp / (tp + fp + 10e-20)
        recall = tp / (tp + fn + 10e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        return precision, recall, f1

    @staticmethod
    def _get_tp_fp_fn(label_list, pred_list):
        tp = len([t for t in pred_list if t in label_list])
        fp = max(0, len(pred_list) - tp)
        fn = max(0, len(label_list) - tp)
        return tp, fp, fn
    
    def _parse_entities(self, tokens):
        entities = []
        for t in tokens:
            if '[' in t and ']' in t:
                entities.append(t)
        return entities


class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def score(self, hypothesis, corpus, n=1):
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in zip(hypothesis, corpus):
            # if type(hyps[0]) is list:
            #    hyps = [hyp.split() for hyp in hyps[0]]
            # else:
            #    hyps = [hyp.split() for hyp in hyps]

            # refs = [ref.split() for ref in refs]
            hyps = [hyps]
            # Shawn's evaluation
            # refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
            # hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']

            for idx, hyp in enumerate(hyps):
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)
                if n == 1:
                    break
        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu


class BleuEvaluator(EvaluatorBase):
    """
    Use string matching to find the F-1 score of slots
    Use logistic regression to find F-1 score of acts
    Use string matching to find F-1 score of KB_SEARCH
    """
    logger = logging.getLogger(__name__)

    def __init__(self, data_name):
        self.data_name = data_name
        self.domain_labels = defaultdict(list)
        self.domain_hyps = defaultdict(list)

    def initialize(self):
        self.domain_labels = defaultdict(list)
        self.domain_hyps = defaultdict(list)

    def add_example(self, ref, hyp, domain='movie'):
        self.domain_labels[domain].append(ref)
        self.domain_hyps[domain].append(hyp)

    def get_report(self, include_error=False):
        reports = []
        tokenize = get_tokenize()
        domain = 'movie'

        self.logger.info("Generate report for {} samples".format(len(self.domain_hyps[domain])))
        refs, hyps = [], []
        tp, fp, fn = 0, 0, 0
        for label, hyp in zip(self.domain_labels[domain], self.domain_hyps[domain]):
            # label = label.replace(EOS, '').replace(BOS, '')
            # hyp = hyp.replace(EOS, '').replace(BOS, '')
            ref_tokens = [BOS] + tokenize(label.replace(SYS, '').replace(USR, '').strip()) + [EOS]
            hyp_tokens = [BOS] + tokenize(hyp.replace(SYS, '').replace(USR, '').strip()) + [EOS]
            refs.append([ref_tokens])
            hyps.append(hyp_tokens)
            ref_entities = self._parse_entities(ref_tokens)
            hyp_entities = self._parse_entities(hyp_tokens)
            tpp, fpp, fnn = self._get_tp_fp_fn(ref_entities, hyp_entities)
            tp += tpp
            fp += fpp
            fn += fnn

        # compute corpus level scores
        bleu = BLEUScorer().score(hyps, refs)
        prec, rec, f1 = self._get_prec_recall(tp, fp, fn)
        report = "\nDomain: {} BLEU score {:.4f}\nEntity precision {:.4f} recall {:.4f} and f1 {:.4f}\n".format(domain, bleu, prec, rec, f1)
        reports.append(report)

        return "\n==== REPORT===={report}".format(report="========".join(reports))





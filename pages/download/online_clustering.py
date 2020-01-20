# Simple, ad-hoc implementation of a low-latency diarization system.
#
#            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
#                    Version 2, December 2004
#
# Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>
#
# Everyone is permitted to copy and distribute verbatim or modified
# copies of this license document, and changing it is allowed as long
# as the name is changed.
#
#            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
#   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION
#
#  0. You just DO WHAT THE FUCK YOU WANT TO.


import glob
import sys
import pickle

from copy import deepcopy
from collections import defaultdict, Counter
from itertools import zip_longest, product, takewhile, tee, dropwhile
from operator import itemgetter
from random import shuffle

import numpy as np

from running_stats import RunningStats
#from pegasos import MultiClassPegasos
#from new_pegasos import MultiClassPegasos
from new_pegasos import MultiClassPerceptron as MultiClassPegasos
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import GreedyDiarizationErrorRate, DiarizationPurity, DiarizationCoverage

try:
    from frogress import bar
except ImportError:
    def bar(x): return x

ADD = "add"
NEW = "new"
IGNORE = "ignore"
SILENCE = "silence"


def spatial_distance(x, y):
    cosine = scipy.spatial.distance.cosine(x, y)
    return np.arccos(np.clip(1.0 - cosine, -1.0, 1.0))


def binary_loss_with_matching(predicted_labels, gold_labels, return_rational=False):

    mapping = {}
    while True:
        confusion_matrix = Counter((a, b) for a, b in zip(
            predicted_labels, gold_labels) if a not in mapping and b not in mapping.values())

        if not confusion_matrix:
            break

        best_match, _ = max(confusion_matrix.items(), key=itemgetter(1))
        mapping[best_match[0]] = best_match[1]

    if return_rational:
        return sum(1 for a, b in zip(predicted_labels, gold_labels) if mapping.get(a, None) != b), len(gold_labels)
    else:
        return sum(1 for a, b in zip(predicted_labels, gold_labels) if mapping.get(a, None) != b) / len(gold_labels)


def rand_index(gold_clusters, predicted_clusters):

    if len(gold_clusters) == 1:
        return 0 if gold_clusters[0] == predicted_clusters[0] else 1

    a = 0
    b = 0
    c = 0
    for i, j in ((i, j) for i, j in product(range(len(gold_clusters)), repeat=2) if i != j):
        if gold_clusters[i] == gold_clusters[j] and predicted_clusters[i] == predicted_clusters[j]:
            a += 1
        elif gold_clusters[i] != gold_clusters[j] and predicted_clusters[i] != predicted_clusters[j]:
            b += 1
        else:
            c += 1

    return 1 - (a + b) / (a + b + c)


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def to_one_hot(value, bins):
    return
    yield
    if value > max(bins):
        value = max(bins)
    if value < min(bins) or value > max(bins):
        print(value, min(bins))
        raise Exception
    if value == min(bins):
        return (1 if i == 0 else 0 for i, _ in enumerate(bins[:-1]))
    else:
        return (1 if value <= b and value > a else 0 for a, b in pairwise(bins))


def arrayify(gen):
    "Convert a generator into a function which returns a list"
    def patched(*args, **kwargs):
        a = list(gen(*args, **kwargs))
        return np.array(a)
    return patched


def listify(gen):
    "Convert a generator into a function which returns a list"
    def patched(*args, **kwargs):
        return list(gen(*args, **kwargs))
    return patched


def dictify(gen):
    def patched(*args, **kwargs):
        return dict(gen(*args, **kwargs))
    return patched


class State:

    def __init__(self, distance=None, oracle="explore"):
        """
        Parameters
        ----------
        - distance, a function taking two arguments
        """
        self.first = True

        #
        self.distance = distance if distance is not None else lambda x, y: np.linalg.norm(
            x - y)

        # all the observations that have been seen so far for a given cluster
        self.points = defaultdict(list)
        # XXX we should get ride of this attribute
        self.n_points = 0

        # center of the different clusters = "model" of each identified speaker
        self.cluster_centers = {}
        self._centers_accum = {}

        # distribution of the distance between points of a cluster & center of
        # this cluster
        self.distance_to_centers = {}

        self.intra_distances = {}
        self.intra_stdev = {}
        self.radius = {}
        self.stats = {}

        # predicted & oracle sequence of labels
        self.predicted_labels = []
        self.gold_labels = []
        self.actions = []

        # sequence of observations
        self.observations = []

        self.label_mapping = {}
        self.next_label = 0

        self.oracle = oracle

    @property
    def clusters(self):
        """
        Return the set of all labels used to identify the clusters.
        """
        return self.points.keys()

    def add_prediction(self, obser, prediction, oracle, gold_label):
        obser = obser.astype(np.float64)
        pred_action, pred_label = prediction

        if oracle is not None:
            oracle_action, oracle_label = oracle
            if pred_action == NEW and oracle_action == NEW:
                # We are creating a new class at the right time, we
                # just have to enforce that the label is the same...
                self.label_mapping[gold_label] = pred_label
                self.next_label = pred_label + 1
            if pred_action == NEW and oracle_action == ADD:
                # We are creating a "extra" new class, we just
                # increase `next_label` to be sure that the given
                # label will not be used an extra time.
                self.next_label += 1
            if pred_action == ADD and oracle_action == NEW:
                # we miss the creation of a new class, we still want
                # to make a new action latter
                return
        else:
            oracle_label = pred_label
            oracle_action = pred_action
            if pred_action == NEW:
                self.next_label += 1

        self.predicted_labels.append(pred_label)
        self.gold_labels.append(oracle_label)
        self.observations.append(obser)
        self.actions.append(oracle_action)

        # IGNORE: the action is still added to the history so that all
        # list describing the history will always have the same size
        # as the number of observations
        #
        # but we are not updating the clusters
        if pred_action == IGNORE:
            return

        if self.oracle == "explore":
            update_label = pred_label
        else:
            update_label = oracle_label

        self.n_points += 1
        self.points[update_label].append(obser)

        # recompute position of the center
        self._centers_accum[update_label] = self._centers_accum[update_label] + \
            obser if update_label in self._centers_accum else obser
        self.cluster_centers[update_label] = self._centers_accum[update_label] / \
            len(self.points[update_label])

        # recompute distributions of distances within this cluster
        self.distance_to_centers[update_label] = [self.distance(
            self.cluster_centers[update_label], p) for p in self.points[update_label]]

        self.stats[update_label] = RunningStats(
            self.distance_to_centers[update_label])
#        self.intra_distances[update_label] = self.stats[update_label].mean
#        self.intra_stdev[update_label] = self.stats[update_label].stdev
#        self.radius[update_label] = self.stats[update_label].max

    @dictify
    def features(self, obser):

        # distance between the observation and all cluster centers
        dist_to_centers = {l: self.distance(obser, self.cluster_centers[l])
                           for l in self.clusters}
        # how many cluster centers are closer to the points that the `current'
        # cluster
        rank = {l: i for i, l in enumerate(
            sorted(dist_to_centers, key=lambda l: dist_to_centers[l]))}
        # stdev of the distance between each cluster center and the
        # points of this cluster after the current observation has
        # been added to the cluster
        stats = deepcopy(self.stats)
        for k in stats:
            stats[k].push(dist_to_centers[k])

        new_stdev = {label: stats[label].stdev for label in self.clusters}
        rank_stdev = {l: i for i, l in enumerate(
            sorted(new_stdev, key=lambda l: new_stdev[l]))}

        @arrayify
        def add_features(l):
            """
            Features describing adding the current observation to the class `l`
            """
            # bias
            # ====
            yield 1

            # Cluster geometry
            # ================
            # distance to the centroid
            yield dist_to_centers[l]
            # distance to the centroid compared to std dev
            yield dist_to_centers[l] / self.stats[l].stdev if self.stats[l].stdev != 0 else 0
            # number of centers that are closer
            yield rank[l]
            # closest center?
            yield from (0, 1) if rank[l] == 0 else (1, 0)
            # % of points in this cluster
            yield len(self.points[l]) / self.n_points
            # smallest increase in intra-cluster distance variance
            yield from (0, 1) if rank_stdev[l] != 0 else (1, 0)
            # further to the mean that the furthest point of the cluster
            yield from (0, 1) if dist_to_centers[l] < self.stats[l].max else (1, 0)
            # closest to center than average radius
            yield from (0, 1) if dist_to_centers[l] < .6 else (1, 0)
            # closest to center that max radius
            yield from (0, 1) if dist_to_centers[l] < 1 else (1, 0)

            # Cluster geometry (binned)
            # =========================
            # distance to the centroid (binned)
            yield from to_one_hot(dist_to_centers[l], np.arange(0, 3, .1))
            # distance to the centroid compared to std dev (binned)
            yield from to_one_hot(dist_to_centers[l] / self.stats[l].stdev if self.stats[l].stdev != 0 else 0,
                                  np.hstack((np.arange(0, 1, .1), np.arange(1, 10))))
            # number of centers that are closer (binned)
            yield from to_one_hot(rank[l], range(0, 10))
            # % of points in this cluster (binned)
            yield from to_one_hot(len(self.points[l]) / self.n_points, np.arange(0, 1, .1))

            # Structure of the sequence
            # =========================
            # distance from the previous point
            yield self.distance(self.observations[-1], obser)
            # distance from the previous point (binned)
            yield from to_one_hot(self.distance(self.observations[-1], obser), np.arange(0, 2, .05))
            # distance from the previous previous point
            yield from to_one_hot(self.distance(self.observations[-2] if len(self.observations) >= 2 else self.observations[-1], obser), np.arange(0, 2, .05))
            # same label as the label predicted for the previous point
            yield from (0, 1) if l == self.predicted_labels[-1] else (1, 0)

            # number of (consecutive) points with this label
            n = sum(1 for point in takewhile(
                lambda x: x == l, reversed(self.predicted_labels)))
            yield n
            yield from to_one_hot(n, [0, 1, 2, 3, 4, 5, 10, 15, 20, 30])

            # number of (consecutive) points with the same label
            n = sum(1 for point in takewhile(
                lambda x: x == self.predicted_labels[-1], reversed(self.predicted_labels)))
            yield n
            yield from to_one_hot(n, [0, 1, 2, 3, 4, 5, 10, 15, 20, 30])
            
            # number of points without this label
            n = sum(1 for point in takewhile(
                lambda x: x != l, reversed(self.predicted_labels)))
            yield n
            yield from to_one_hot(n, [0, 1, 2, 3, 4, 5, 10, 15, 20, 30])

        @arrayify
        def new_features():
            """
            Feature used to decide whether a new speaker has been identified
            """

            # bias
            yield 1

            # "Geometry" of current observations
            # ==================================
            # number of clusters created so far
            yield len(self.clusters)
            # number of clusters created so far (binned)
            yield from to_one_hot(len(self.clusters), range(20))
            # distance to nearest center
            yield min(dist_to_centers.values())
            # distance to nearest center (binned)
            yield from to_one_hot(min(dist_to_centers.values()), np.hstack((np.arange(0, 1, .1), np.arange(1, 10))))
            # further away from all clusters center from all other points?
            yield from (1, 0) if all(self.stats[l].max < dist_to_centers[l] for l in self.clusters) else (0, 1)
            # further than the max radius observed on the training set
            yield from (1, 0) if all(dist_to_centers[l] > 1 for l in self.clusters) else (0, 1)

            # # max increase of the radius
            # argmax_dist = max(self.clusters, key=lambda l: abs(self.stats[l].max - dist_to_centers[l]))
            # # i) difference
            # yield abs(self.radius[argmax_dist] - dist_to_centers[argmax_dist])
            # # ii) ratio
            # yield self.radius[argmax_dist] / dist_to_centers[argmax_dist]
            # # max increase of the radius (binned)
            # yield from to_one_hot(self.radius[argmax_dist] /
            # dist_to_centers[argmax_dist], np.arange(0, 4, .1))

            # Features from previous observations
            # ===================================
            # distance from the previous point
            yield self.distance(self.observations[-1], obser)
            # distance from the previous point (binned)
            yield from to_one_hot(self.distance(self.observations[-1], obser), np.arange(0, 2, .05))
            # previous action is a new ?
            yield from (0, 1) if self.actions[-1] == NEW else (1, 0)
            # Number of observations since last new locuteur
            last_creation = sum(1 for point in takewhile(
                lambda x: x == ADD, reversed(self.actions)))
            yield last_creation
            # Number of observations since last new locuteur (binned)
            yield from to_one_hot(last_creation, [0, 10, 20, 30, 40, 50, 100, 150])
            # Number of silences
            last_silences = sum(1 for point in takewhile(
                lambda x: x == SILENCE, reversed(self.predicted_labels)))
            yield last_silences
            yield from to_one_hot(last_silences, range(10))

            yield 1 if self.predicted_labels[-1] == SILENCE else 0

            # number of (consecutive) points with the same label
            n = sum(1 for point in takewhile(
                lambda x: x == self.predicted_labels[-1], reversed(self.predicted_labels)))
            yield n
            yield from to_one_hot(n, [0, 1, 2, 3, 4, 5, 10, 15, 20, 30])

            
        for label in self.clusters:
            yield (ADD, label), add_features(label)

        yield (NEW, self.next_label), new_features()

    def oracle_action(self, label):
        if label in self.label_mapping:
            return ADD, self.label_mapping[label]
        else:
            return NEW, self.next_label


class SequenceLabeler:

    def __init__(self, distance=None, oracle="explore", mu=1):
        self.model = MultiClassPegasos(mu)
        self.learner = self.model.fit(itemgetter(0))
        self.distance = distance
        self.oracle = oracle

    def _infer(self, observations, labels=[], partial_labels=[], debug=False):
        """
        Online prediction of the labels of a sequence of observations
        """
        state = State(self.distance, oracle=self.oracle)
        # first action is always the creation of the class 0
        state.add_prediction(observations[0], (NEW, 0), (NEW, 0), 0)
        yield 0, state

        for obser, label, partial_label in zip_longest(observations[1:], labels[1:], partial_labels[1:]):
            features = state.features(obser)

            if label is None:
                # testing
                gold_label = None
                if partial_label is None:
                    predicted_label = self.model.predict(
                        features, itemgetter(0))
                else:
                    # XXXX will only work for silence
                    predicted_label = IGNORE, partial_label
            else:
                # training
                gold_label = state.oracle_action(label)

                if partial_label is not None:
                    if partial_label != SILENCE:
                        predicted_label = gold_label
                    else:
                        predicted_label = IGNORE, SILENCE
                else:
                    predicted_label = self.learner.send((features, gold_label))

            state.add_prediction(obser, predicted_label, gold_label, label)
            yield predicted_label[1], state

    def predict(self, observations, partial_labels=[]):
        return list(l[0] for l in self._infer(observations, partial_labels=partial_labels, debug=True))

    def test(self, test_data, dump=False):
        n_obser = 0
        n_error = 0

        metric = GreedyDiarizationErrorRate()

        if dump:
            ofile = open(dump, "wb")

        for example in test_data:

            if len(example) == 2:
                observations, gold_labels = example
                partial_labels = [None] * len(gold_labels)
            else:
                observations, gold_labels, partial_labels = example

            predicted = self.predict(observations, partial_labels)

            if dump:
                pickle.dump((predicted, gold_labels), ofile)
            err, length = binary_loss_with_matching(
                predicted, gold_labels, return_rational=True)

            metric(*convert_labels(gold_labels, predicted))
            n_obser += length
            n_error += err

        return n_error / n_obser, abs(metric)

    def fit(self, train_data, test_data, n_iter=1, prefix=None):
        learning_curve = []

        for i in range(n_iter):
            n_error = 0
            n_predicted = 0

            train_der = GreedyDiarizationErrorRate()

            shuffle(train_data)
            
            for example in bar(train_data):

                if len(example) == 2:
                    observations, labels = example
                    partial_labels = [None] * len(labels)
                else:
                    observations, labels, partial_labels = example

                predicted = [pl for pl, state in self._infer(
                    observations, labels, partial_labels)]
                assert len(predicted) == len(observations)

                train_der(*convert_labels(labels, predicted))
                err, length = binary_loss_with_matching(
                    predicted, labels, return_rational=True)

                n_predicted += length
                n_error += err

            print("iteration {}".format(i))
            print("error: {:.2%}/{:.2%}".format(n_error /
                                                n_predicted, abs(train_der)))

            w = deepcopy(self.model.weight)
            self.model.avg()
            test_loss, test_der = self.test(
                test_data, dump=prefix.format(i) if prefix is not None else None)
            print("test: {:.2%}/{:.2%}".format(test_loss, test_der))

            self.model.weight = w
            
            learning_curve.append({"train_loss": n_error / n_predicted,
                                   "train_der": abs(train_der),
                                   "test_loss": test_loss,
                                   "test_der": test_der})
        return learning_curve


def convert_labels(y_true, y_pred):
    reference = Annotation()
    hypothesis = Annotation()

    for i, (r, h) in enumerate(zip(y_true, y_pred)):
        segment = Segment(i, i + 1)

        if h != SILENCE:
            hypothesis[segment] = h
        if r != SILENCE:
            reference[segment] = r

    return hypothesis, reference


def load_data(pattern):

    dataset = []
    for f in glob.glob(pattern):
        obser = np.load(f.replace(".y.", ".X."))
        labels = np.load(f)
        partial_labels = [SILENCE if x is None else None for x in
                          np.load(f.replace("delexicalized", "lexicalized"), encoding="latin1")]

        data = zip(obser, labels, partial_labels)

        obser, labels, partial_labels = zip(
            *dropwhile(lambda x: x[-1] == "silence", data))
        mapper = defaultdict(lambda: len(mapper))

        labels = [mapper[l] if pl != SILENCE else SILENCE for l,
                  pl in zip(labels, partial_labels)]

        assert partial_labels[0] != SILENCE
        assert labels[0] == 0

        dataset.append((obser, labels, partial_labels))

    return dataset


if __name__ == "__main__":

    import argparse

    from os.path import expanduser

    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", default=1, type=float)
    parser.add_argument("--length", default=None, type=int)
    parser.add_argument("--n_iter", default=5, type=int)
    parser.add_argument("--learning_curve", default=None,
                        type=argparse.FileType("wb"))
    parser.add_argument(
        "--oracle", choices=["explore", "exploit"], default="explore")
    parser.add_argument("--known_silence", action="store_true", default=False)
    parser.add_argument("--known_init", default=None, type=int)
    args = parser.parse_args()

    assert args.known_init is None or not args.known_silence

    if args.known_silence:
        train_data = load_data(expanduser(
            "~/workspace/onlinediarization/data/delexicalized/trn.*y.npy"))
        test_data = load_data(expanduser(
            "~/workspace/onlinediarization/data/delexicalized/dev.*y.npy"))
        
    elif args.known_init is not None:
        raise Exception("Must update _infer method first")
    # train_data = [(np.load(f),
    #                    np.load(f.replace(".X.", ".y.")),
    #                    np.load(f.replace(".X.", ".y."))[:args.known_init])
    #                   for f in glob.glob(expanduser("~/workspace/onlinediarization/data/delexicalized/trn.*X*"))]
    #     test_data = [(np.load(f),
    #                   np.load(f.replace(".X.", ".y.")),
    #                   np.load(f.replace(".X.", ".y."))[:args.known_init])
    # for f in
    # glob.glob(expanduser("~/workspace/onlinediarization/data/delexicalized/tst.*X*"))]
    else:
        train_data = [(np.load(f), np.load(f.replace(".X.", ".y."))) for f in glob.glob(
            expanduser("~/workspace/onlinediarization/data/delexicalized/trn.*X*"))]
        test_data = [(np.load(f), np.load(f.replace(".X.", ".y."))) for f in glob.glob(
            expanduser("~/workspace/onlinediarization/data/delexicalized/tst.*X*"))]

    assert test_data

    if args.length is not None:
        train_data = [tuple(xx[:args.length] for xx in x) for x in train_data]
        test_data = [tuple(xx[:args.length] for xx in x) for x in test_data]

    s = SequenceLabeler(mu=args.mu, oracle=args.oracle)
    lc = s.fit(train_data, test_data, args.n_iter,
               prefix="dev_pred_mu={}_oracle={}_iter={{}}.pkl".format(args.mu, args.oracle))

    print(s.test(test_data, dump=False))

    if args.learning_curve is not None:
        pickle.dump(lc, args.learning_curve)

# -*- coding: utf-8 -*-

from __future__ import print_function

from collections import Counter
import json

from femlAlgorithms import *
from helpers import normalize_str, getRelativePathtoWorking
from external.datasetcreator import filter_dataset, build_dataset_from_geonames
from helpers import StaticValues


class Evaluator:
    evaluatorType_action = {
        'SotAMetrics': calcSotAMetrics,
        'SotAML': calcSotAML,
        'customFEML': calcCustomFEML,
        'DLearning': calcDLearning,
        'TestMetrics': testMetrics,
        'customFEMLExtended': calcCustomFEMLExtended,
        'lSimilarityMetrics': calcLSimilarities,
    }

    def __init__(self, ml_algs, sorting=False, stemming=False, canonical=False, permuted=False, only_latin=False, encoding=None):
        self.ml_algs = [x for x in ml_algs.split(',')]
        self.permuted = permuted
        self.stemming = stemming
        self.canonical = canonical
        self.sorting = sorting
        self.latin = only_latin
        self.encoding = encoding

        self.termfrequencies = {
            'gram': Counter(),
            '2gram_1': Counter(), '3gram_1': Counter(),
            '2gram_2': Counter(), '3gram_2': Counter(), '3gram_3': Counter(),
        }
        self.termsperalphabet = {}
        self.stop_words = []
        self.abbr = {'A': [], 'B': []}
        self.evalClass = None

    def initialize(self, dataset, evalType='SotAMetrics', njobs=2, accuracyresults=False):
        try:
            self.evalClass = self.evaluatorType_action[evalType](njobs, accuracyresults)
        except KeyError:
            print("Unkown method")
            return 1

        with open(dataset) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                    delimiter='\t')
            for row in reader:
                self.evalClass.preprocessing(row)

    def evaluate_metrics(self, dataset='dataset-string-similarity.txt', feature_selection=None, features=None):
        if self.evalClass is not None:
            self.evalClass.freq_terms_list()

            lFeatures = [(True if x == 'True' else False) for x in features.split(',')] if feature_selection is None and features is not None else features
            print("Reading dataset...")
            with open(dataset) as csvfile:
                reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                        delimiter='\t')

                thres_type = 'orig'
                if self.sorting:
                    thres_type = 'sorted'
                if self.latin:
                    # thres_type += '_onlylatin'
                    thres_type += '_latin_EU/NA'
                if self.encoding:
                    thres_type += '_all'

                for row in reader:
                    self.evalClass.evaluate(
                        row, self.sorting, self.stemming, self.canonical, self.permuted, self.termfrequencies, thres_type, lFeatures
                    )
            if hasattr(self.evalClass, "train_classifiers"):
                self.evalClass.train_classifiers(self.ml_algs, polynomial=False, standardize=True, fs_method=feature_selection, features=lFeatures)
            self.evalClass.print_stats()

    def evaluate_metrics_with_various_thres(self, dataset='dataset-string-similarity.txt'):
        if self.evalClass is not None:
            self.evalClass.freq_terms_list()

            start_time = time.time()
            print("Reading dataset...")
            with open(dataset) as csvfile:
                reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                        delimiter='\t')

                print('Computing stats for threshold', end='')

                all_res = {}
                for m in StaticValues.methods: all_res[m[0]] = []
                separator = ''
                for i in range(30, 91, 5):
                    print('{0} {1}'.format(separator, float(i / 100.0)), end='')
                    sys.stdout.flush()
                    separator = ','

                    csvfile.seek(0)
                    for row in reader:
                        self.evalClass.evaluate(
                            row, self.sorting, self.stemming, self.canonical, self.permuted, self.termfrequencies,
                            float(i / 100.0)
                        )
                    if hasattr(self.evalClass, "train_classifiers"): self.evalClass.train_classifiers(self.ml_algs)
                    tmp_res = self.evalClass.get_stats()

                    for key, val in tmp_res.items():
                        all_res[key].append([float(i / 100.0), val])

                    self.evalClass.reset_vars()

                print('\nThe process took {0:.2f} sec'.format(time.time() - start_time))
                for k, val in all_res.items():
                    if len(val) == 0:
                        print('{0} is empty'.format(k))
                        continue

                    print(k, max(val, key=lambda x: x[1][0]))
                sys.stdout.flush()

    def evaluate_sorting_with_various_thres(self, dataset='dataset-string-similarity.txt'):
        if self.evalClass is not None:
            start_time = time.time()
            print("Reading dataset...")
            with open(dataset) as csvfile:
                reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                        delimiter='\t')

                separator = ''
                print('Computing stats for threshold', end='')

                thres_type = 'orig'
                if self.sorting:
                    thres_type = 'sorted'
                if self.latin:
                    # thres_type += '_onlylatin'
                    thres_type += '_latin_EU/NA'
                if self.encoding:
                    thres_type += '_all'

                all_res = {}
                for m in StaticValues.methods: all_res[m[0]] = []
                for i in range(55, 86, 5):
                    print('{0} {1}'.format(separator, float(i / 100.0)), end='')
                    sys.stdout.flush()
                    separator = ','

                    csvfile.seek(0)
                    for row in reader:
                        # if self.latin and (row['a1'] != 'LATIN' or row['a2'] != 'LATIN'): continue

                        self.evalClass.evaluate_sorting(row, float(i / 100.0), thres_type, self.stemming, self.permuted)
                    if hasattr(self.evalClass, "train_classifiers"): self.evalClass.train_classifiers(self.ml_algs)
                    tmp_res = self.evalClass.get_stats()

                    for key, val in tmp_res.items():
                        all_res[key].append([float(i / 100.0), val])

                    self.evalClass.reset_vars()

            print('\nThe process took {0:.2f} sec'.format(time.time() - start_time))
            for k, val in all_res.items():
                if len(val) == 0:
                    print('{0} is empty'.format(k))
                    continue

                print(k, max(val, key=lambda x: x[1][0]))

    def test_cases(self, dataset, test_case):
        if test_case - 1 == 0:
            print("Not implemented yet!!!")
        elif test_case - 1 == 1:
            if not os.path.exists("output"):
                os.makedirs("output")

            ngram_stats = {
                '2gram': Counter(), '3gram': Counter(), '4gram': Counter(),
                'gram_token': Counter(), '2gram_token': Counter(), '3gram_token': Counter()
            }
            abbr_stats = Counter()
            orig_strs = {}
            no_dashed_strs = 0

            with open(dataset) as csvfile:
                reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                        delimiter='\t')

                feml = FEMLFeatures()
                for row in reader:
                    row['s1'], row['s2'] = transform(row['s1'], row['s2'], canonical=True)

                    for sstr in ['s1', 's2']:
                        # calc the number of abbr that exist
                        abbr_str = feml.containsAbbr(row[sstr])
                        if abbr_str != '-':
                            if abbr_str not in orig_strs.keys(): orig_strs[abbr_str] = []
                            abbr_stats[abbr_str] += 1
                            orig_strs[abbr_str].append(row[sstr])

                        # search for dashes in strings
                        no_dashed_strs += feml.containsDashConnected_words(row[sstr])

                        row[sstr] = transform_str(row[sstr], canonical=True)
                        ngram_tokens, _ = normalize_str(row[sstr], self.stop_words)

                        for term in ngram_tokens:
                            ngram_stats['gram_token'][term] += 1
                        for gram in list(itertools.chain.from_iterable(
                                [[ngram_tokens[i:i + n] for i in range(len(ngram_tokens) - (n - 1))]
                                 for n in [2, 3]])
                        ):
                            if len(gram) == 2:
                                ngram_stats['2gram_token'][' '.join(gram)] += 1
                            else:
                                ngram_stats['3gram_token'][' '.join(gram)] += 1

                        # ngrams chars
                        # ngrams = zip(*[''.join(strA_ngrams_tokens)[i:] for i in range(n) for n in [2, 3, 4]])
                        for gram in list(itertools.chain.from_iterable(
                                [[''.join(ngram_tokens)[i:i + n] for i in range(len(''.join(ngram_tokens)) - (n - 1))]
                                 for n in [2, 3, 4]])
                        ):
                            if len(gram) == 2:
                                ngram_stats['2gram'][gram] += 1
                            elif len(gram) == 3:
                                ngram_stats['3gram'][gram] += 1
                            elif len(gram) == 4:
                                ngram_stats['4gram'][gram] += 1

            print("Found {} dashed words in the dataset.".format(no_dashed_strs))

            with open("./output/abbr.csv", "w+") as f:
                f.write('abbr\tcount\tstr\n')
                for value, count in abbr_stats.most_common():
                    f.write("{0}\t{1}\t{2}\n".format(value, count, ','.join(orig_strs[value][:10])))

            for nm in ngram_stats.keys():
                with open("./output/{0}s.csv".format(nm), "w+") as f:
                    f.write('gram\tcount\n')
                    for value, count in ngram_stats[nm].most_common():
                        f.write("{}\t{}\n".format(value.encode('utf8'), count))
        elif test_case - 1 == 2:
            if self.evalClass is not None:
                self.evalClass.freq_terms_list()

                print("Reading dataset...")
                with open(dataset) as csvfile:
                    reader = csv.DictReader(csvfile,
                                            fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                            delimiter='\t')

                    all_res = {}
                    for m in StaticValues.methods: all_res[m[0]] = []
                    feml = FEMLFeatures()
                    print('====================================================================================')
                    print("The averaged lSimilarity is being tested by default. To test the normal one, update ")
                    print("parameter 'averaged'=False under 'class testMetrics' in file 'femlAlgorithms.py'")
                    print('====================================================================================')
                    for n in [3.34] + list(range(4, 8)):
                        weight_combs = [
                            tuple(float(x/10.0) for x in seq)
                            for seq in itertools.product([1, 2, 3, 4, 5, 2.5, 3.33], repeat=2)
                            if sum(seq) == (10 - n)
                        ]

                        for w in weight_combs:
                            w = (float(n/10.0), ) + w
                            feml.update_weights(w)
                            print('Computing stats for weights ({})'.format(','.join(map(str, w))))
                            print('Computing stats for threshold', end='')

                            start_time = time.time()

                            tmp_res = {}
                            for m in StaticValues.methods: tmp_res[m[0]] = []
                            separator = ''
                            for i in range(35, 81, 5):
                                print('{0} {1}'.format(separator, float(i / 100.0)), end='')
                                separator = ','
                                #  required for python before 3.3
                                sys.stdout.flush()

                                internal_separator = ' ['
                                for k in range(60, 81, 5):
                                    print('{0}{1}'.format(internal_separator, float(k / 100.0)), end='')
                                    internal_separator = ', '
                                    #  required for python before 3.3
                                    sys.stdout.flush()

                                    csvfile.seek(0)
                                    for row in reader:
                                        self.evalClass.evaluate(
                                            row, self.sorting, self.stemming, self.canonical, self.permuted, self.termfrequencies,
                                            float(i / 100.0), float(k / 100.0)
                                        )
                                    if hasattr(self.evalClass, "train_classifiers"): self.evalClass.train_classifiers(self.ml_algs)
                                    res = self.evalClass.get_stats()

                                    for key, val in res.items():
                                        tmp_res[key].append([float(i / 100.0), val, list(w), float(k / 100.0)])

                                    self.evalClass.reset_vars()

                                print(']', end='')

                            print('\nThe process for weight ({0}) took {1:.2f} sec'.format(','.join(map(str, w)), time.time() - start_time))
                            for k, val in tmp_res.items():
                                if len(val) == 0:
                                    continue

                                all_res[k].extend(val)
                                print(k, max(val, key=lambda x: x[1][0]))

                    print("\nFinal Results")
                    for k, val in all_res.items():
                        if len(val) == 0:
                            continue

                        print(k, max(val, key=lambda x: x[1][0]))
        elif test_case - 1 == 3:
            if self.evalClass is not None:
                self.evalClass.freq_terms_list()

                print("Reading dataset...")
                with open(dataset) as csvfile:
                    reader = csv.DictReader(csvfile,
                                            fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                            delimiter='\t')
                    for row in reader:
                        self.evalClass.evaluate(
                            row, self.sorting, self.stemming, self.canonical, self.permuted, self.termfrequencies, 'sorted'
                        )
                    self.evalClass.debug_stats()
        elif test_case - 1 == 4:
            self.evalClass.freq_terms_list()

            output_f = open("./output/lsimilarity_terms.csv", "w+")
            output_f.write("res\tstr1\tbase_s1\tmismatch_s1\tspecial_s1\tstr2\tbase_s2\tmismatch_s2\tspecial_s2\n")
            print("Reading dataset...")
            with open(dataset) as csvfile:
                reader = csv.DictReader(csvfile,
                                        fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                        delimiter='\t')

                for row in reader:
                    a, b = transform(row['s1'], row['s2'], sorting=True, canonical=True)
                    baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(a, b, 0.55)
                    output_f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                        row['res'].upper(),
                        row['s1'], ','.join(baseTerms['a']).encode('utf8'),
                        ','.join(mismatchTerms['a']).encode('utf8'),
                        ','.join(specialTerms['a']).encode('utf8'),
                        row['s2'], ','.join(baseTerms['b']).encode('utf8'),
                        ','.join(mismatchTerms['b']).encode('utf8'),
                        ','.join(specialTerms['b']).encode('utf8')
                    ))

            output_f.close()
        else:
            print("Test #{} does not exist!!! Please choose a valid test to execute.".format(test_case))

    def print_false_posneg(self, datasets):
        if not os.path.exists("output"):
            os.makedirs("output")

        if len(datasets) == 2:
            reader = pd.read_csv((datasets[0]), sep='\t', names=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"])
            results = pd.read_csv(datasets[1], sep='\t', names=["res1", "res2"])

            print("No of rows for dataset: {0}".format(results.shape[0]/2))
            resultDf = pd.concat([results.iloc[results.shape[0]/2:], results.iloc[:results.shape[0]/2]], ignore_index=True)
            mismatches = pd.concat([reader, resultDf], axis=1)

            negDf = mismatches[
                (not self.latin or mismatches.a1 == 'LATIN') & (not self.latin or mismatches.a2 == 'LATIN') &
                (mismatches.res1 is True) & (mismatches.res1 != mismatches.res2)
            ]
            negDf.to_csv('./output/false_negatives.csv', sep='\t', encoding='utf-8', columns=['s1', 's2'])
            posDf = mismatches[
                (not self.latin or mismatches.a1 == 'LATIN') & (not self.latin or mismatches.a2 == 'LATIN') &
                (mismatches.res1 is False) & (mismatches.res1 != mismatches.res2)
            ]
            posDf.to_csv('./output/false_positives.csv', sep='\t', encoding='utf-8', columns=['s1', 's2'])
        elif len(datasets) == 3:
            reader = pd.read_csv(getRelativePathtoWorking(datasets[0]), sep='\t',
                                 names=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"])

            res1 = pd.read_csv(
                datasets[1], sep='\t',
                names=["res1"] + list(map(lambda x: "res1_{0}".format(x), StaticValues.methods_as_saved)) + \
                      ["res1_transformed_s1", "res1_transformed_s2"]
            )
            res2 = pd.read_csv(
                datasets[2], sep='\t',
                names=["res2"] + list(map(lambda x: "res2_{0}".format(x), StaticValues.methods_as_saved)) + \
                      ["res2_transformed_s1", "res2_transformed_s2"]
            )

            mismatches = pd.concat([reader, res1, res2], axis=1)
            mismatches = mismatches.sort_values(by=['res'], ascending=False)

            for metric_name in StaticValues.methods_as_saved:
                negDf = mismatches[
                    (not self.latin or mismatches.a1 == 'LATIN') & (not self.latin or mismatches.a2 == 'LATIN') &
                    (mismatches.res1 == mismatches['res1_{0}'.format(metric_name)]) &
                    (mismatches.res2 != mismatches['res2_{0}'.format(metric_name)])
                    ]
                negDf.to_csv('./output/false_enhancedmetric_{0}.csv'.format(metric_name), sep='\t',
                             encoding='utf-8', columns=['s1', 's2', 'res', "res2_transformed_s1", "res2_transformed_s2"])

                tmpDf = mismatches[ mismatches.res1 != mismatches.res2 ]
                if not tmpDf.empty: print(tmpDf)
        else: print("Wrong number {0} of input datasets to cmp".format(len(datasets)))

    def build_dataset(self):
        build_dataset_from_geonames(only_latin=self.latin)
        filter_dataset()

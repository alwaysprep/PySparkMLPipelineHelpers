"""
Inspired from komiya-atsushi's Scala code.
"""

from pyspark.ml.evaluation import Evaluator

import math

class BinaryLogLossEvaluator(Evaluator):

    def _evaluate(self, dataset):
        def f(data):
            probabilities, label = data
            epsilon = 1e-15
            probability = max(epsilon, min(1 - epsilon, probabilities[1]))
            return label * math.log(probability) + (1 - label) * math.log(1 - probability)

        minusLogLoss = dataset.select("probability", "label")\
                    .rdd\
                    .map(f)\
                    .mean()

        return -minusLogLoss

    def isLargerBetter(self):
        return False

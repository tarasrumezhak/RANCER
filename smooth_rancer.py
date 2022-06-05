from math import ceil

import numpy as np
import torch
from scipy.stats import binom_test
from statsmodels.stats.proportion import proportion_confint

from ddsmoothing.certificate import Certificate


class Smooth(object):
    """A smoothed classifier g

    Adapted from: https://github.com/locuslab/smoothing/blob/master/code/core.py
    """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: torch.Tensor, R: torch.Tensor,
                 certificate: Certificate):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        :param R: noise distribution "rotation" (transformation)
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.R = R
        self.certificate = certificate

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int, device: torch.device = torch.device('cuda:0')) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2/L1 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2/L1 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        counts_selection = self._sample_noise(x, n0, batch_size, device=device)
        cAHat = counts_selection.argmax().item()
        counts_estimation = self._sample_noise(x, n, batch_size, device=device)
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            return cAHat, self.certificate.compute_gap(pABar)

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int, device: torch.device = torch.device('cuda:0')) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size, device=device)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size, device: torch.device = torch.device('cuda:0')) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():

            counts = torch.zeros(self.num_classes, dtype=float, device=device)

            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((1, this_batch_size, 1, 1)).reshape((this_batch_size, 3, 32, 32))
                noise = self.certificate.sample_noise(batch, self.sigma)

                noise = noise.reshape((this_batch_size, 1, 3072))
                noise_t = torch.transpose(noise, 2, 1)

                rotated_noise_t = torch.matmul(self.R, noise_t)
                rotated_noise = torch.transpose(rotated_noise_t, 2, 1)
                rotated_noise = rotated_noise.reshape((this_batch_size, 3, 32, 32))

                predictions = self.base_classifier(batch + rotated_noise).argmax(1)  # rotated_noise

                counts += self._count_arr(predictions,
                                          device, self.num_classes)

            return counts.cpu().numpy()

    def _count_arr(self, arr: torch.tensor, device: torch.device, length: int) -> torch.tensor:
        counts = torch.zeros(length, dtype=torch.long, device=device)
        unique, c = arr.unique(sorted=False, return_counts=True)
        counts[unique] = c
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """

        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

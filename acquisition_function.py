import numpy as np
import scipy.stats as stats


def get_EI_list(train_Y, pred_y, sigma2_pred):
    prediction = pred_y
    sig = sigma2_pred**0.5
    gamma = (prediction - np.max(train_Y)) / sig
    ei = sig*(gamma*stats.norm.cdf(gamma) + stats.norm.pdf(gamma))
    return ei

def calc_EI_overfmax(fmean, fcov, fmax):
    fstd = np.sqrt(fcov)
    temp1 = fmean - fmax
    temp2 = temp1 / fstd
    score = temp1 * stats.norm.cdf(temp2) + fstd * stats.norm.pdf(temp2)
    return score

def calc_PI_overfmax(fmean, fcov, fmax):
    fstd = np.sqrt(fcov)
    temp = (fmean - fmax) / fstd
    score = stats.norm.cdf(temp)
    return score

def calc_PI_underfmin(fmean, fcov, fmin):
    fstd = np.sqrt(fcov)
    temp = (fmin - fmean) / fstd
    score = stats.norm.cdf(temp)
    return score

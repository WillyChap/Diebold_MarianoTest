# Simple 
# Diebold-Mariano test. Modified from code by Adrian Trapletti.
# Then adapted by M. Yousaf Khan for better performance on small samples
# Then Brought to Python from R by Will Chapman 

#' Diebold-Mariano test for predictive accuracy
#'
#' The Diebold-Mariano test compares the forecast accuracy of two forecast
#' method

#' Diebold-Mariano test for predictive accuracy
#'
#' The Diebold-Mariano test compares the forecast accuracy of two forecast
#' methods.
#'
#' This function implements the modified test proposed by Harvey, Leybourne and
#' Newbold (1997). The null hypothesis is that the two methods have the same
#' forecast accuracy. For \code{alternative="less"}, the alternative hypothesis
#' is that method 2 is less accurate than method 1. For
#' \code{alternative="greater"}, the alternative hypothesis is that method 2 is
#' more accurate than method 1. For \code{alternative="two.sided"}, the
#' alternative hypothesis is that method 1 and method 2 have different levels
#' of accuracy.
#'
#' @param e1 Forecast errors from method 1.
#' @param e2 Forecast errors from method 2.
#' @param alternative a character string specifying the alternative hypothesis,
#' must be one of \code{"two.sided"} (default), \code{"greater"} or
#' \code{"less"}.  You can specify just the initial letter.
#' @param h The forecast horizon used in calculating \code{e1} and \code{e2}.
#' @param power The power used in the loss function. Usually 1 or 2.
#' @return A list with class \code{"htest"} containing the following
#' components: \item{statistic}{the value of the DM-statistic.}
#' \item{parameter}{the forecast horizon and loss function power used in the
#' test.} \item{alternative}{a character string describing the alternative
#' hypothesis.} \item{p.value}{the p-value for the test.} \item{method}{a
#' character string with the value "Diebold-Mariano Test".} \item{data.name}{a
#' character vector giving the names of the two error series.}
#' @author George Athanasopoulos
#' @references Diebold, F.X. and Mariano, R.S. (1995) Comparing predictive
#' accuracy. \emph{Journal of Business and Economic Statistics}, \bold{13},
#' 253-263.
#'
#' Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of
#' prediction mean squared errors. \emph{International Journal of forecasting},
#' \bold{13}(2), 281-291.
#' @keywords htest ts
#' @examples



import numpy as np
import pandas as pd 
from statsmodels.tsa.stattools import acf, pacf, acovf
import warnings
from scipy import stats



def DMtest(e1,e2,h=1,power=2,alternative='two.sided'):
    d = np.abs(e1)**power - np.abs(e2)**2
    dcov = acovf(d,nlag=h-1,fft=False,missing='drop')
    dvar = np.sum(np.append(dcov[0],2*dcov[1:]))/len(d)
    dv = dvar
    if dv>0:
        STATISTIC =np.nanmean(d)/np.sqrt(dv)
    elif (h==1):
        raise ValueError("Variance of DM statistic is zero")
    else: 
        warnings.warn('Variance is negative, using horizon h=1') 
        DMtest(e1,e2,h=1,power=power)   
    n = len(d)
    k = ((n + 1 - 2 * h + (h / n) * (h - 1)) / n) **(0.5)
    STATISTIC = STATISTIC *k 
    
    if alternative == 'two.sided':
        PVAL = 2*stats.t.cdf(-np.abs(STATISTIC),df=n-1)
    elif alternative == 'less':
        PVAL = stats.t.cdf(STATISTIC,df=n-1)
    elif alternative == 'greater':
        PVAL = 1-stats.t.cdf(STATISTIC,df=n-1)
        
    PARAMETER = [h,power]
    return_DICT = {"DM":STATISTIC,'forecast_horizon':h,'power':power,'p_val':PVAL,'Method':'Diebold-Mariano Test'
                  }
    return return_DICT


def bh_correction(pval,level=0.05):
    nn = len(pval)
    pv_srt = sorted(pval)
    limits = np.arange(1,nn+1)*level/nn
    
    if len(np.where(pv_srt<limits))==0:
        discoveries = 0
        return discoveries
    else:
        discoveries = np.max(np.where(pv_srt<limits))
        return discoveries/nn 
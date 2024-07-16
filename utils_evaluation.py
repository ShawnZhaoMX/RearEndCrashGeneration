# # Install rpy2
# !pip install rpy2

# # Import rpy2's interface to R
# import rpy2.robjects.packages as rpackages
# from rpy2.robjects.vectors import StrVector

# # Define the package names
# packnames = ('Ecume',)

# # Install the packages
# utils = rpackages.importr('utils')
# utils.chooseCRANmirror(ind=1)
# utils.install_packages(StrVector(packnames))

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as rpyn

from rpy2.robjects.packages import importr

r_Ecume = importr('Ecume')

def weighted_ks_2samp(x1, x2, w1=None, w2=None):
    """
    Compute weighted Kolmogorov-Smirnov two-sample test using R library (Ecume)
    :return statistic
    :return pvalue
    """
    robjects.r.assign('x1', rpyn.numpy2rpy(x1))
    robjects.r.assign('x2', rpyn.numpy2rpy(x2))

    robjects.r("""w1 <- rep(1, length(x1))"""
               ) if w1 is None else robjects.r.assign('w1', rpyn.numpy2rpy(w1))
    robjects.r("""w2 <- rep(1, length(x2))"""
               ) if w2 is None else robjects.r.assign('w2', rpyn.numpy2rpy(w2))
    

    robjects.r("""ks_result <- ks_test(x=as.matrix(x1), y=as.matrix(x2),
           w_x=as.matrix(w1), w_y=as.matrix(w2), thresh = 0)""")
    ks_result = robjects.globalenv['ks_result']
    statistic = rpyn.rpy2py_floatvector(ks_result[0])[0]
    pvalue = rpyn.rpy2py_floatvector(ks_result[1])[0]
    return statistic, pvalue

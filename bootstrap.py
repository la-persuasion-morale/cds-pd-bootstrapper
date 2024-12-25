# import packages
import numpy as np
from numpy import interp


# The bootstrap method
def bootstrap(h, given_haz, s, cds_tenor, yield_curve, prem_per_year, R):
    '''
    Returns the difference between values of payment leg and default leg.
    '''
    a = 1/prem_per_year
    maturities = [0] + list(cds_tenor)    
    pmnt = 0;        dflt = 0;        auc = 0
    
    # 1. Calculate value of payments for given hazard rate curve values
    for i in range(1, len(maturities)-1):
        num_points = int((maturities[i]-maturities[i-1])*prem_per_year + 1)
        t = np.linspace(maturities[i-1], maturities[i], num_points) 
        r = interp(t)
        
        for j in range(1, len(t)):
            surv_prob_prev = np.exp(-given_haz[i-1]*(t[j-1]-t[0]) - auc)
            surv_prob_curr = np.exp(-given_haz[i-1]*(t[j]-t[0]) - auc)
            pmnt += s*a*np.exp(-r[j]*t[j])*0.5*(surv_prob_prev + surv_prob_curr)
            dflt += np.exp(-r[j]*t[j])*(1-R)*(surv_prob_prev - surv_prob_curr)
    
        auc += (t[-1] - t[0])*given_haz[i-1]
    
    # 2. Set up calculations for payments with the unknown hazard rate value
    num_points = int((maturities[-1]-maturities[-2])*prem_per_year + 1)
    t = np.linspace(maturities[-2], maturities[-1], num_points)
    r = interp(t)
    
    for i in range(1, len(t)):
        surv_prob_prev = np.exp(-h*(t[i-1]-t[0]) - auc)
        surv_prob_curr = np.exp(-h*(t[i]-t[0]) - auc)          
        pmnt += s*a*np.exp(-r[i]*t[i])*0.5*(surv_prob_prev + surv_prob_curr)
        dflt += np.exp(-r[i]*t[i])*(1-R)*(surv_prob_prev - surv_prob_curr)
    
    return abs(pmnt-dflt)
# import packages
import numpy as np
import scipy.optimize as optim
import scipy.interpolate as polate
from bootstrap import bootstrap

# the CDS bootstrap method
def CDS_bootstrap(cds_spreads, yield_curve, cds_tenor, yield_tenor, prem_per_year, R):
    '''
    Bootstraps a credit curve from CDS spreads of varying maturities. Returns the hazard 
    rate values and survival probabilities corresponding to the CDS maturities.

    Args:
        cds_spreads :   vector of CDS spreads
        yield_curve :   vector of risk-free bond yields
        cds_tenor :     vector of maturities corresponding to the given CDS spreads
        yield_tenor :   vector of risk-free bond yield tenor matching yield_curve
        prem_per_year : premiums paid per year on the CDS (i.e. annualy=1, semiannually=2, quarterly=4, monthly=12) 
        R :             recovery rate
    '''

    # Checks
    assert len(cds_spreads) == len(cds_tenor), "CDS spread and it's tenor array must be the same."
    assert len(yield_curve) == len(yield_tenor), "Bond yield curve and it's tenor must be the same."
    
    # Interpolation/Extrapolation function  
    interp = polate.interp1d(yield_tenor, yield_curve,'linear', fill_value='extrapolate')
    
    haz_rates = []
    surv_prob = []
    t = [0] + list(cds_tenor)
    
    for i in range(len(cds_spreads)):
        get_haz = lambda x: bootstrap(x, haz_rates, cds_spreads[i], cds_tenor[0:i+1], yield_curve[0:i+1], prem_per_year, R)
        haz = round(optim.minimize(get_haz, cds_spreads[i]/(1-R), method='SLSQP', tol = 1e-10).x[0],8)
        cond_surv = (t[i+1]-t[i])*haz
        haz_rates.append(haz)
        surv_prob.append(cond_surv)
    
    return haz_rates, np.exp(-np.cumsum(surv_prob))
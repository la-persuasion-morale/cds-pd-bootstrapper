# import packages
import numpy as np
import scipy.interpolate as polate


# the binary CDS spread method
def binary_CDS_spread(credit_curve, yield_curve, credit_curve_tenor, yield_tenor, prem_per_year, default_payout, maturity):
    '''
    Returns the spread of a binary CDS using a yield curve and credit curve

    Args:
        credit_curve :  vector of hazard rates that correspond to CDSs of different maturities
        yield_curve :   vector of yields for risk-free bonds
        credit_curve_tenor :    vector of maturities for CDS contracts corresponding to credit_curve
        yield_tenor :   vector of risk-free bond yield maturities corresponding to yield_curve
        prem_per_year : number of premiums paid per year (i.e. annually=1, semiannually=2, quarterly=4, monthly=12)
        default_payout :    amount paid in the event of a default as % of principal
        maturity :      desired CDS maturity
    '''
    # Checks
    assert len(yield_curve) == len(yield_tenor), "Bond yield curve and it's tenor must be the same."
    assert len(credit_curve) == len(credit_curve_tenor), "Credit curve and it's tenor array must be the same."  
    
    # I. Get survival probabilities and default probabilities using hazard rate curve
    a = 1/prem_per_year
    num_points = int(credit_curve_tenor[-1]/a + 1)
    t = np.linspace(0, credit_curve_tenor[-1], num_points)
    h = []
    index = 0;  t_index = credit_curve_tenor[index]

    for i in range(len(t)):
        if t[i] <= t_index:
            h.append(credit_curve[index])
        else:
            index += 1
            t_index = credit_curve_tenor[index]
            h.append(credit_curve[index])
        
    surv_prob = [1.0]
    
    for i in range(1,len(t)):
        surv_prob.append(a*h[i])
        
    surv_prob = np.exp(-np.cumsum(surv_prob))    
    default_prob = np.asarray([0] + list(-np.diff(surv_prob)))    
    
    # II. Interpolate/Extrapolate yield curve values corresponding to payment times and default times    
    interp = polate.interp1d(yield_tenor, yield_curve, 'linear',fill_value='extrapolate')
    pay_periods = np.linspace(0, credit_curve_tenor[-1], num_points)
    mid_periods = np.linspace(a/2, credit_curve_tenor[-1]-a/2, num_points-1)
    yield1 = interp(pay_periods)
    yield2 = interp(mid_periods)
    
    # III. Solve
    PV_pmnt = [np.exp(-yield1[i]*pay_periods[i])*surv_prob[i] for i in range(1,len(pay_periods))]
    PV_payoff = [default_payout*default_prob[i+1]*np.exp(-yield2[i]*mid_periods[i]) for i in range(len(mid_periods))]
    PV_accrual = [np.exp(-yield2[i]*mid_periods[i])*0.5*a*default_prob[i+1] for i in range(len(mid_periods))]
    
    return sum(PV_payoff)/(sum(PV_pmnt) + sum(PV_accrual))
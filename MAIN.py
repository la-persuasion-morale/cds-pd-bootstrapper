# import packages
import numpy as np
import scipy.optimize as optim
import scipy.interpolate as polate
from bootstrap import bootstrap
from cds_bootstrap import CDS_bootstrap
from cds_spread import CDS_spread
from binary_cds_spread import binary_CDS_spread

# parameters
cds_spreads = [0.0110, 0.0120, 0.0130, 0.0140, 0.0150]
yield_curve = [0.01350, 0.01430, 0.0190, 0.02470, 0.02936, 0.03311]
cds_tenor = [1, 2, 3, 4, 5]
yield_tenor = [0.5, 1, 2, 3, 4, 5]
prem_per_year = 4
R = 0.40
maturity = 5
credit_curve = [0.99, 0.98, 0.95, 0.92]
credit_curve_tenor = [1, 3, 5, 7]
default_payout = 0.6


# main method
if __name__ == '__main__':
    CDS_bootstrap(cds_spreads, yield_curve, cds_tenor, yield_tenor, prem_per_year, R)
    CDS_spread(credit_curve, yield_curve, credit_curve_tenor, yield_tenor, prem_per_year, R, maturity)
    binary_CDS_spread(credit_curve, yield_curve, credit_curve_tenor, yield_tenor, prem_per_year, default_payout, maturity)







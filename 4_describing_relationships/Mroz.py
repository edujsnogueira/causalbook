import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from causaldata import Mroz

# Read in data
dt = Mroz.load_pandas().data
# Keep just working women
dt = dt[dt['lfp'] == True]
# Create unlogged earnings
dt.loc[:,'earn'] = dt['lwg'].apply('exp')

# 1. Draw a scatterplot 
sns.scatterplot(x = 'inc',
y = 'earn',
data = dt).set(xscale="log", yscale="log")
# The .set() gives us log scale axes

# 2. Get the conditional mean by college attendance
# wc is the college variable
dt.groupby('wc')[['earn']].mean()

# 3. Get the conditional mean by bins
# Use cut to get 10 bins
dt.loc[:, 'inc_bin'] = pd.cut(dt['inc'],10)
dt.groupby('inc_bin')[['earn']].mean()

# 4. Draw the LOESS and linear regression curves
# Do log beforehand for these axes
dt.loc[:,'linc'] = dt['inc'].apply('log')
sns.regplot(x = 'linc',
            y = 'lwg',
            data = dt,
            lowess = True)
sns.regplot(x = 'linc',
            y = 'lwg',
            data = dt,
            ci = None)
            
# 5. Run a linear regression, by itself and including controls
m1 = sm.ols(formula = 'lwg ~ linc', data = dt).fit()
print(m1.summary())
# k5 is number of kids under 5 in the house
m2 = sm.ols(formula = 'lwg ~ linc + wc + k5', data = dt).fit()
print(m2.summary())
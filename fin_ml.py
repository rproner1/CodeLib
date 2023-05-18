import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def compute_returns(test_returns, probs, thresh=0.5, strategy='long', betsize=False, betsizetype='sharpe'):

    """
    Computes the returns of a strategy given test returns and probabilities of positive return month.

    Parameters
    ----------
    test_returns:  Test set monthly factor returns. pd.DataFrame, np.ndarray, or pd.Series object.
    probs: Predicted probabilities of positive factor returns for each month in the test set.
    thresh: The probability threshold for classifying a return as positive. 
    strategy: trading strategy {'long', 'longshort'}. 'long' will implement a long only strategy.
    betsize: Determines if bets are to be sized according to the estimated probabilites (bool).
    betsizetype: {'sharpe', 'prob'} Betsizing technique to use. Sharpe betsizing willl create betsizes m = 2F(z)-1 where 
        z = (p(x) - 0.5) / [p(x)(1-p(x))]^0.5, p(x) is the probability of positive return given features X=x, and F(.) 
        is the normal CDF. If 'prob', m = 2p(x)-1.

    Returns
    ----------
    strategy_returns: out-of-sample returns using the specified strategy.
    """

    # If a series or single col df is passed, convert it to an array
    if not isinstance(test_returns, np.ndarray):
        test_returns = test_returns.values
    
    if not isinstance(probs, np.ndarray):
        probs = probs.values

    preds = np.copy(probs)
    preds[preds > thresh] = 1
    preds[preds <= thresh] = 0
    if strategy == 'longshort':
        preds[preds == 0] = -1
    
    if betsize:
        # estimated sharpe ratio
        z = ( probs - 0.5 ) / ( probs*(1-probs) )**0.5
        
        # bet size
        m = 2*norm.cdf(z) - 1

        if betsizetype=='prob':
            m = 2*probs-1

        if strategy == 'long':
            # This prevents shorts from turning to buys where m <0 and pred = -1
            m[m < 0] = 0
        
            # print(m.shape)

        preds = m

    preds = preds.reshape(test_returns.shape)

    strategy_returns = test_returns * preds

    return strategy_returns
    
def economic_significance(benchmark_returns, strategy_returns):
    
    """
    Computes a strategies portfolio statistics (mean, std. dev., and Sharpe ratio) as well a summary of the 
    regression of strategy returns on benchmark returns.

    Parameters
    ----------
    benchmark_returns: A 1-D array or array-like of returns of the benchmark strategy.
    strategy_returns: A 1-D array or array-like of returns of the strategy of interest.

    Returns
    ----------
    results: A dictionary containing an economic significance report.
    """

    ols_data = pd.DataFrame({'buy_hold_ret': benchmark_returns, 'strat_ret': strategy_returns})
    
    ols = smf.ols('strat_ret ~ buy_hold_ret', ols_data).fit()
    
    ols_params = ols.params
    ols_t_vals = ols.tvalues
    
    mean = np.mean(strategy_returns) * 12 * 100
    stdev = np.std(strategy_returns) * np.sqrt(12) * 100
    sharpe = mean / stdev 

    # annualize alpha and convert to percentage
    results = {
        'mean': [mean],
        'stdev': [stdev],
        'sharpe': [sharpe],
        'alpha': [ols_params[0]*100*12], 
        't_alpha': [ols_t_vals[0]], 
        'beta': [ols_params[1]], 
        't_beta': [ols_t_vals[1]], 
        'R2': [ols.rsquared]
    }
    
    results = {key: np.round(value, 4) for (key,value) in results.items()}
    
    return results

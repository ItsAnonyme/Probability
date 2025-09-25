import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize
import math
from decimal import Decimal

# Choose your teams
HomeTeam = "Brondby"
AwayTeam = "Aarhus"

max_goals = 20

# Where we get our data
List = pd.read_csv("https://www.football-data.co.uk/new/DNK.csv", nrows=2761)

def bivariate_log_likelihood(lambda3, goals_home, goals_away, lambda1_pred, lambda2_pred):
    total_log_likelihood = 0
    for x, y, l1, l2 in zip(goals_home, goals_away, lambda1_pred, lambda2_pred):
        log_p = calculate_log_bivariate_pmf(x, y, l1, l2, lambda3)
        total_log_likelihood += log_p
    return -total_log_likelihood

def calculate_log_bivariate_pmf(x, y, l1, l2, l3):
    k_min = 0
    k_max = min(x, y)
    total = 0
    log_constant_term = -l1 - l2 - l3 + x * np.log(l1) + y * np.log(l2) - math.lgamma(x + 1) - math.lgamma(y + 1)
    for k in range(k_min, k_max + 1):
        term = (math.lgamma(x+1) + math.lgamma(y+1) - math.lgamma(x - k + 1) - math.lgamma(y - k + 1) - math.lgamma(k+1))
        term += k * np.log(l3 / (l1 * l2))
        total += np.exp(term)
    if total <= 0:
        return -1e10
    log_prob = log_constant_term + np.log(total)
    return log_prob

def BivariatePoisson(x, y, H, A, T):
    P = (math.exp(-H-A-T) * (H**x * A**y)/(math.factorial(x)*math.factorial(y))
         * sum(((math.factorial(x)*math.factorial(y))/(math.factorial(x - k)*math.factorial(y - k)*math.factorial(k)))
               * (T/H*A)**k for k in range(0, min(x, y))))
    return P

# Rewriting the data
home_df = pd.DataFrame(data={"team": List.Home, "opponent": List.Away, "goals": List.HG, "home": 1})
away_df = pd.DataFrame(data={"team": List.Away, "opponent": List.Home, "goals": List.AG, "home": 0})

# Applying Poisson Distribution
model_Poisson = smf.glm(data=pd.concat([home_df, away_df]), family=sm.families.Poisson(), formula="goals ~ home + team + opponent").fit()

# Goals prediction
home_goals_obs = List['HG'].values
away_goals_obs = List['AG'].values
lambda1_preds = model_Poisson.predict(pd.DataFrame(data={"team": List.Home, "opponent": List.Away, "home": 1}, index=[1])).values
lambda2_preds = model_Poisson.predict(pd.DataFrame(data={"team": List.Away, "opponent": List.Home, "home": 0}, index=[1])).values

initial_lambda3_guess = 0.1
result = minimize(bivariate_log_likelihood, x0=initial_lambda3_guess, args=(home_goals_obs, away_goals_obs, lambda1_preds, lambda2_preds), bounds=[(1e-6, None)])

if result.success:
    optimal_lambda3 = result.x[0]
    #print("Optimal lambda3 =", optimal_lambda3)
else:
    print("Optimization failed.", result.message)
    optimal_lambda3 = 0.05

home_goals = model_Poisson.predict(pd.DataFrame(data={"team": HomeTeam, "opponent": AwayTeam, "home": 1}, index=[1])).values[0]
away_goals = model_Poisson.predict(pd.DataFrame(data={"team": AwayTeam, "opponent": HomeTeam, "home": 0}, index=[1])).values[0]

T = optimal_lambda3

Probabilities_Bivariate = np.zeros((max_goals, max_goals))
for i in range(max_goals):
    for j in range(max_goals):
        # Use a non-log version of the PMF for the final probability matrix
        # You can adapt your original function or use an exp(log) version
        Probabilities_Bivariate[i, j] = np.exp(calculate_log_bivariate_pmf(i, j, home_goals, away_goals, T))

# Ensure the matrix is valid (probabilities sum to ~1)
Probabilities_Bivariate /= Probabilities_Bivariate.sum()

# Summing the probability of each outcome
Home_Win = np.sum(np.tril(Probabilities_Bivariate, -1))
Draw = np.sum(np.diag(Probabilities_Bivariate))
Away_Win = np.sum(np.triu(Probabilities_Bivariate, 1))

if __name__ == "__main__":
    print(f"The probability of each team winning the match, according to the Bivariate Poisson Distribution, is:"
          f"\n {HomeTeam}: {round(Decimal(Home_Win * 100), 3)} % \n {AwayTeam}: {round(Decimal(Away_Win * 100), 3)} % \n Draw: {round(Decimal(Draw * 100), 3)} %"
          f"\n Sum of percentages: {Home_Win + Away_Win + Draw}"
          f"\n This is calculated with a max goal of {max_goals}"
          f"\n and lambda values of 1: {lambda1_preds}, 2: {lambda2_preds}, 3: [{optimal_lambda3}]")
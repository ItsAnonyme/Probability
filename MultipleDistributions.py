import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson, nbinom, invgauss
import math

# Choose your teams
HomeTeam = "Brondby"
AwayTeam = "Aarhus"

max_goals = 20

# Where we get our data
List = pd.read_csv("https://www.football-data.co.uk/new/DNK.csv")

# Rewriting the data
home_df = pd.DataFrame(data={"team": List.Home, "opponent": List.Away, "goals": List.HG, "home": 1})
away_df = pd.DataFrame(data={"team": List.Away, "opponent": List.Home, "goals": List.AG, "home": 0})

# Applying Poisson Distribution
model_Poisson = smf.glm(data=pd.concat([home_df, away_df]), family=sm.families.Poisson(), formula="goals ~ home + team + opponent").fit()

# Goals prediction
home_goals_poisson = model_Poisson.predict(pd.DataFrame(data={"team": HomeTeam, "opponent": AwayTeam, "home": 1}, index=[1])).values[0]
away_goals_poisson = model_Poisson.predict(pd.DataFrame(data={"team": HomeTeam, "opponent": AwayTeam, "home": 0}, index=[1])).values[0]

# The matrix containing the probability of each outcome up to max_goals
Probability_matrix_Poisson = [[poisson.pmf(i, team_avg) for i in range(0, max_goals)]
for team_avg in [home_goals_poisson, away_goals_poisson]]

matrix_Poisson = np.outer(np.array(Probability_matrix_Poisson[0]), np.array(Probability_matrix_Poisson[1]))

# Summing the probability of each outcome
Home_Win_Poisson = np.sum(np.tril(matrix_Poisson, -1))
Draw_Poisson = np.sum(np.diag(matrix_Poisson))
Away_Win_Poisson = np.sum(np.triu(matrix_Poisson, 1))

# Applying Negative Binomial Distribution
model_NegBin = smf.glm(data=pd.concat([home_df, away_df]), family=sm.families.Gaussian(), formula="goals ~ home + team + opponent").fit()

# Goals prediction
home_goals_NegBin = model_NegBin.predict(pd.DataFrame(data={"team": HomeTeam, "opponent": AwayTeam, "home": 1}, index=[1])).values[0]
away_goals_NegBin = model_NegBin.predict(pd.DataFrame(data={"team": HomeTeam, "opponent": AwayTeam, "home": 0}, index=[1])).values[0]


# The matrix containing the probability of each outcome up to max_goals
Probability_matrix_NegBin = [[(math.exp(-((i-team_avg)**2)/(2*team_avg**2*i)))/(math.sqrt(2*math.pi*i**3)) for i in range(0, max_goals)]
for team_avg in [home_goals_NegBin, away_goals_NegBin]]

matrix_NegBin = np.outer(np.array(Probability_matrix_NegBin[0]), np.array(Probability_matrix_NegBin[1]))

# Summing the probability of each outcome
Home_Win_NegBin = np.sum(np.tril(matrix_NegBin, -1))
Draw_NegBin = np.sum(np.diag(matrix_NegBin))
Away_Win_NegBin = np.sum(np.triu(matrix_NegBin, 1))

if __name__ == "__main__":
    print(f"The probability of each team winning the match, according to the Poisson Distribution, is:"
          f"\n {HomeTeam}: {Home_Win_Poisson}\n {AwayTeam}: {Away_Win_Poisson}\n Draw: {Draw_Poisson}"
          f"\nThis is calculated with a max goal of {max_goals}\n")

    print(f"The probability of each team winning the match, according to the Inverse Gaussian Distribution, is:"
          f"\n {HomeTeam}: {Home_Win_NegBin}\n {AwayTeam}: {Away_Win_NegBin}\n Draw: {Draw_NegBin}"
          f"\nThis is calculated with a max goal of {max_goals}\n")
    print(home_goals_poisson, away_goals_poisson)
    print(home_goals_NegBin, away_goals_NegBin)


# Bivariate Poisson
























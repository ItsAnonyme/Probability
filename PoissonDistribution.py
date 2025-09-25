import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson
from AvgGoalMethods import *
import matplotlib as plt
import math


# Choose your teams
HomeTeam = "Chelsea"
AwayTeam = "Arsenal"

# Where we get our data
Data = pd.read_csv("Dataset-small.csv", skiprows=range(1, 3042), skipfooter=216, engine='python')

def prediction_poisson(HomeTeam, AwayTeam):
    #print("Predicting for:", HomeTeam, "vs", AwayTeam)
    max_goals = 10

    # Rewriting the data
    home_df = pd.DataFrame(data={"team": Data.HomeTeam, "opponent": Data.AwayTeam, "goals": Data.FTHG, "home": 1, "win": 1})
    away_df = pd.DataFrame(data={"team": Data.AwayTeam, "opponent": Data.HomeTeam, "goals": Data.FTAG, "home": 0, "win": 0})

    # Applying Poisson Distribution
    model_Poisson = smf.glm(data=pd.concat([home_df, away_df]), family=sm.families.Poisson(),
                            formula="goals ~ home + team + opponent + win").fit()
    #print(model_Poisson.summary())
    # Goals prediction
    home_goals = \
    (model_Poisson.predict(pd.DataFrame(data={"team": HomeTeam, "opponent": AwayTeam, "home": 1, "win": 1}, index=[1])).values[0])
    away_goals = \
    model_Poisson.predict(pd.DataFrame(data={"team": AwayTeam, "opponent": HomeTeam, "home": 0, "win": 0}, index=[1])).values[0]

    # The matrix containing the probability of each outcome up to max_goals
    Probability_matrix = [[poisson.pmf(i, team_avg) for i in range(0, max_goals)]
                          for team_avg in [home_goals, away_goals]]

    matrix = np.outer(np.array(Probability_matrix[0]), np.array(Probability_matrix[1]))

    # Summing the probability of each outcome
    Home_Win = np.sum(np.tril(matrix, -1))
    Draw = np.sum(np.diag(matrix))
    Away_Win = np.sum(np.triu(matrix, 1))
    if Home_Win > Away_Win and Home_Win > Draw:
        return "H"
    elif Away_Win > Draw and Away_Win > Home_Win:
        return "A"
    else:
        return "D"

if __name__ == "__main__":
    print(prediction_poisson(HomeTeam, AwayTeam))




























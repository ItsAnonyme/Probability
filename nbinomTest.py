import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import nbinom

List = pd.read_csv("https://www.football-data.co.uk/new/DNK.csv", skiprows=range(2764, 2809))


def prediction_nbinomial2(HomeTeam, AwayTeam):
    max_goals = 10

    # Prepare data - convert to numeric to avoid object dtype issues
    home_df = pd.DataFrame(data={"team": List.Home.astype('category').cat.codes,
                                 "opponent": List.Away.astype('category').cat.codes,
                                 "goals": List.HG, "home": 1})
    away_df = pd.DataFrame(data={"team": List.Away.astype('category').cat.codes,
                                 "opponent": List.Home.astype('category').cat.codes,
                                 "goals": List.AG, "home": 0})
    data = pd.concat([home_df, away_df])

    # Create mapping for team names to codes
    all_teams = pd.concat([List.Home, List.Away]).unique()
    team_mapping = {team: code for code, team in enumerate(all_teams)}

    # Convert team names to codes for prediction
    home_code = team_mapping[HomeTeam]
    away_code = team_mapping[AwayTeam]

    # Fit Negative Binomial model using GLM with formula
    # This automatically handles the dispersion parameter estimation
    model_NegBin = smf.glm(formula="goals ~ home + C(team) + C(opponent)",
                           data=data,
                           family=sm.families.NegativeBinomial(alpha=0.2)).fit()

    # Create prediction data with proper encoding
    pred_data_home = pd.DataFrame({
        "home": [1],
        "team": [home_code],
        "opponent": [away_code]
    })

    pred_data_away = pd.DataFrame({
        "home": [0],
        "team": [away_code],
        "opponent": [home_code]
    })

    # Predict goals
    home_goals_NegBin = model_NegBin.predict(pred_data_home).values[0]
    away_goals_NegBin = model_NegBin.predict(pred_data_away).values[0]

    # Get the estimated alpha parameter from the model
    # For GLM NegativeBinomial, alpha is stored in the scale attribute
    alpha = model_NegBin.scale

    # Calculate probabilities using correct Negative Binomial PMF
    # Using parameterization: nbinom.pmf(k, n=1/alpha, p=1/(1 + alpha * mu))
    prob_matrix_NegBin = [
        [nbinom.pmf(k=i, n=1 / alpha, p=1 / (1 + alpha * mu)) for i in range(max_goals)]
        for mu in [home_goals_NegBin, away_goals_NegBin]
    ]

    matrix_NegBin = np.outer(np.array(prob_matrix_NegBin[0]), np.array(prob_matrix_NegBin[1]))

    # Calculate outcome probabilities
    Home_Win_NegBin = np.sum(np.tril(matrix_NegBin, -1))
    Draw_NegBin = np.sum(np.diag(matrix_NegBin))
    Away_Win_NegBin = np.sum(np.triu(matrix_NegBin, 1))

    if Home_Win_NegBin > Away_Win_NegBin and Home_Win_NegBin > Draw_NegBin:
        return "H"
    elif Away_Win_NegBin > Draw_NegBin and Away_Win_NegBin > Home_Win_NegBin:
        return "A"
    else:
        return "D"
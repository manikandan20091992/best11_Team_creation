import requests
# views.py
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
import pytz

##############################################################
def select_best_players(sorted_players, winning_team, min_from_winning_team=7, max_from_winning_team=8, total_players=11):
    winning_team_players = [player for player in sorted_players if player['country'].lower() == winning_team.lower()]
    other_players = [player for player in sorted_players if player['country'].lower() != winning_team.lower()]

    # Sort winning team players by score in descending order
    winning_team_players_sorted = sorted(winning_team_players, key=lambda x: x['score'], reverse=True)

    # Select players from the winning team (min 7, max 10)
    selected_players = winning_team_players_sorted[:max_from_winning_team]
    winning_team_count = len(selected_players)

    # If we have less than 7 from the winning team, adjust selection
    if winning_team_count < min_from_winning_team:
        needed = min_from_winning_team - winning_team_count
        # Add the highest-scoring players from the winning team until we reach at least 7
        for player in winning_team_players_sorted[winning_team_count:]:
            if needed <= 0:
                break
            selected_players.append(player)
            needed -= 1

    # Check if we have reached the limit of 10 players from the winning team
    selected_players = selected_players[:max_from_winning_team]

    # Fill remaining spots from other players, ensuring category constraints
    category_count = {'wk': 0, 'batsman': 0, 'all': 0, 'bowler': 0}

    for player in selected_players:
        category_count[player['category']] += 1

    # Ensure at least one player from each category
    categories_needed = {key: 1 - category_count[key] for key in category_count.keys() if category_count[key] == 0}

    for player in other_players:
        if len(selected_players) >= total_players:
            break

        # Check if we need a player from this category
        if player['category'] in categories_needed and categories_needed[player['category']] > 0:
            selected_players.append(player)
            category_count[player['category']] += 1
            categories_needed[player['category']] -= 1

    # If we still need players, fill from the remaining sorted list
    for player in other_players:
        if len(selected_players) >= total_players:
            break
        if player not in selected_players:
            selected_players.append(player)

    return selected_players[:total_players]

@api_view(['GET', 'POST'])
def select_best_11_players(request):
    best_11_players = []
    captain = None
    voice_captain = None

    if request.method == 'POST':
        file = request.FILES['file']
        winningTeam = request.data['winningTeam']
        data = pd.read_excel(file)
        data.fillna(0, inplace=True)
        # Composite score calculation (already handled per category)
        data['composite_score'] = 0
        data['new_bowlingavg'] = 0
        data['new_allrounderavg'] = 0


        # Initialize a dictionary to hold category-specific models  decision trees created for each category Bagging
        category_models = {
            'wk': RandomForestRegressor(),
            'batsman': RandomForestRegressor(),
            'all': RandomForestRegressor(),
            'bowler': RandomForestRegressor()
        }

        # Prepare training data for each category
        category_datasets = {
            'wk': {'X': [], 'y': []},
            'batsman': {'X': [], 'y': []},
            'all': {'X': [], 'y': []},
            'bowler': {'X': [], 'y': []}
        }

        # Loop through players, create feature sets for each category
        for index, row in data.iterrows():
            category = row['category'].lower()
            
            if category == 'wk':  # Wicketkeeper
                category_datasets['wk']['X'].append([row['batting_avg']])#, row['strike_rate'], row['runs']
                category_datasets['wk']['y'].append(
                    row['batting_avg'] * 1
                )

            elif category == 'batsman':
                category_datasets['batsman']['X'].append([row['batting_avg']])
                category_datasets['batsman']['y'].append(
                    row['batting_avg'] * 1
                )

            elif category == 'all':
                # Ensure the bowling average falls within the correct range
                if row['bowling_avg'] <= 20:
                    row['bowling_avg'] = 50  # Excellent
                elif row['bowling_avg'] <= 30:
                    row['bowling_avg'] = 40  # Good
                elif row['bowling_avg'] <= 40:
                    row['bowling_avg'] = 30  # Average
                elif row['bowling_avg'] <= 50:
                    row['bowling_avg'] = 20  # Below Average
                else:
                    row['bowling_avg'] = 10  # Poor


                # New scoring logic: Average of batting_avg and bowling_avg
                # Allroundercomposite_score = max(row['batting_avg'], row['bowling_avg'])

                if row['batting_avg'] > row['bowling_avg']:
                    batting_weight = 0.7  # Give more weight to batting if it's better
                    bowling_weight = 0.3
                else:
                    batting_weight = 0.3  # Give more weight to bowling if it's better
                    bowling_weight = 0.7

                composite_score = (row['batting_avg'] * batting_weight) + (row['bowling_avg'] * bowling_weight)
                data.loc[index, 'new_allrounderavg']=composite_score
                category_datasets['all']['X'].append([composite_score])
                category_datasets['all']['y'].append(composite_score)

            elif category == 'bowler':
                if row['bowling_avg'] <= 20:
                    row['bowling_avg'] = 50  # Excellent
                elif row['bowling_avg'] <= 30:
                    row['bowling_avg'] = 40  # Good
                elif row['bowling_avg'] <= 40:
                    row['bowling_avg'] = 30  # Average
                elif row['bowling_avg'] <= 50:
                    row['bowling_avg'] = 20  # Below Average
                else:
                    row['bowling_avg'] = 10  # Poor
                data.loc[index, 'new_bowlingavg']=row['bowling_avg']
                category_datasets['bowler']['X'].append([row['bowling_avg']])
                category_datasets['bowler']['y'].append(
                    row['bowling_avg'] * 1
                )
        # breakpoint()
        # Train models for each category row sampling and feature sampling
        #Random forest classifier means to take maximum count output from model 
        #Random forest regression means to take a mean from model
        for category in category_models:
            
            X_train, X_test, y_train, y_test = train_test_split(
                category_datasets[category]['X'], 
                category_datasets[category]['y'], 
                test_size=0.2, 
                random_state=42
            )
            category_models[category].fit(X_train, y_train)
            # Evaluate model
            y_pred = category_models[category].predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
        # breakpoint()
        # Predict for all players based on their category
        player_scores = []
        
        for index, row in data.iterrows():
            category = row['category'].lower()
            model = category_models[category]
            # breakpoint()
            # Get the input features depending on the category
            if category == 'wk':
                features = [row['batting_avg']]#, row['strike_rate'], row['runs']
            elif category == 'batsman':
                features = [row['batting_avg']]#, row['strike_rate'], row['runs']
            elif category == 'all':
                features = [row['new_allrounderavg']]#, row['strike_rate'], row['runs'], 
            elif category == 'bowler':
                features = [row['new_bowlingavg']]

            score = model.predict([features])[0]
            player_scores.append({'name': row['player_name'], 'score': score, 'category': row['category'], 'country': row['country']})

        # Sort players based on their score
        sorted_players = sorted(player_scores, key=lambda x: x['score'], reverse=True)
        
        
        # Example usage
        if winningTeam != "":
            sorted_best_11 = select_best_players(sorted_players, winningTeam)
            
        else:
            # Ensure at least one player from each category (wk, batsman, all, bowler)
            category_count = {'wk': 0, 'batsman': 0, 'all': 0, 'bowler': 0}
            best_11_players = []

            # Select players while ensuring at least one from each category
            for player in sorted_players:
                if player['category'] in category_count and category_count[player['category']] == 0:
                    best_11_players.append(player)
                    category_count[player['category']] += 1

                if len(best_11_players) == 11:
                    break

            # Fill the remaining spots to make 11 players while respecting the max limit of 10 players per country
            country_count = {country: 0 for country in set(data['country'])}
            for player in sorted_players:
                if len(best_11_players) < 11 and country_count[player['country']] < 10 and player not in best_11_players:
                    best_11_players.append(player)
                    country_count[player['country']] += 1

            # Check if we have at least one player from each category
            if (category_count['wk'] == 0 or category_count['batsman'] == 0 or 
                category_count['all'] == 0 or category_count['bowler'] == 0):
                return Response({"code": 400, "message": "Unable to select best 11 players. Must include at least one player from each category."})

            # Sort the best 11 players by score in descending order
            sorted_best_11 = sorted(best_11_players, key=lambda x: x['score'], reverse=True)
        ##################################################################3

        # Get the top two players
        # top_two = sorted_best_11[:2]

        # Create separate dictionaries for the top two players
        
    
        # breakpoint()
        sorted_players = sorted(sorted_best_11, key=lambda x: x['score'], reverse=True)

        # Label the top two players as captain and vice-captain
        
        
        # Render the template with the results
        
        category_counts = {
            'batsman': 0,
            'wk': 0,
            'all': 0,
            'bowler': 0
        }

        # Loop through players and count each category
        for player in sorted_players:
            category = player['category']
            if category in category_counts:
                category_counts[category] += 1


        # Limit to a maximum of 11 players, preserving fixed players
        sorted_players = sorted(sorted_players, key=lambda x: x['score'], reverse=True)
        if winningTeam !='':
            winningTeamCapVC = [player for player in sorted_players if player['country'] == winningTeam][:2]
            capDict = {}
            # top_two_dicts = [player for player in top_two]
            capDict['captain'] = winningTeamCapVC[0]['name']
            capDict['country'] = winningTeamCapVC[0]['country']
            vcDict = {}
            vcDict['voice_captain'] = winningTeamCapVC[1]['name']
            vcDict['country'] = winningTeamCapVC[1]['country']

            for player in sorted_players:
                if player['name'] == winningTeamCapVC[0]['name'] and player['country'] == winningTeamCapVC[0]['country']:
                    player['name'] = f"{player['name']} (captain)"
                if player['name'] == winningTeamCapVC[1]['name']and player['country'] == winningTeamCapVC[1]['country']:
                    player['name'] = f"{player['name']} (vice captain)"    
            
            # sorted_players[0]['name'] += " (Captain)"
            # sorted_players[1]['name'] += " (Vice-Captain)"

        else:
            capDict = {}
            # top_two_dicts = [player for player in top_two]
            capDict['captain'] = sorted_players[0]['name']
            capDict['country'] = sorted_players[0]['country']
            vcDict = {}
            vcDict['voice_captain'] = sorted_players[1]['name']
            vcDict['country'] = sorted_players[1]['country']
            
            sorted_players[0]['name'] += " (Captain)"
            sorted_players[1]['name'] += " (Vice-Captain)"
        country_counts = {}

        for player in sorted_players:
            country = player['country']
            country_counts[country] = country_counts.get(country, 0) + 1

        
        print(sorted_players)
        return render(request, 'select_best_11.html', {
            'best_11': sorted_players,
            # 'captain': top_two_dicts[0],
            # 'voice_captain': top_two_dicts[1],
            "country_counts":country_counts,
            "category_counts":category_counts,
            "capDict":capDict,
            "vcDict":vcDict,

        })

    # If the request method is GET, render the form
    return render(request, 'select_best_11.html')

@api_view(['POST'])
def ai_team_best11(request):
    import pandas as pd

    file = request.FILES['file']
    data = pd.read_excel(file)

    df = pd.DataFrame(data)

    # Replace NaN values with 0 for players who don't have certain stats (e.g., batsmen without bowling stats)
    df = df.fillna(0)
        # Split the data by roles
    batters = df[df['role'] == 'Batsman']
    bowlers = df[df['role'] == 'Bowler']
    allrounders = df[df['role'] == 'All-rounder']
    wicketkeepers = df[df['role'] == 'WK']
    # Features and target for wicketkeepers (similar to batters)
    X_wicketkeepers = wicketkeepers[['batting_avg', 'runs','catches','stampings']]
    y_wicketkeepers = [1] * len(wicketkeepers)

    # Train-test split
    X_train_wicketkeepers, X_test_wicketkeepers, y_train_wicketkeepers, y_test_wicketkeepers = train_test_split(X_wicketkeepers, y_wicketkeepers, test_size=0.2)

    # Train the model
    model_wicketkeepers = RandomForestClassifier()
    model_wicketkeepers.fit(X_train_wicketkeepers, y_train_wicketkeepers)

    # Predictions
    wicketkeepers['is_selected_prediction'] = model_wicketkeepers.predict(X_wicketkeepers)

    # Features and target for batters (use batting-related metrics)
    X_batters = batters[['batting_avg', 'runs','strike_rate','catches']]
    y_batters = [1] * len(batters)  # For simplicity, assume they should all be selected (you can adjust this logic)

    # Train-test split
    X_train_batters, X_test_batters, y_train_batters, y_test_batters = train_test_split(X_batters, y_batters, test_size=0.2)

    # Train the model
    model_batters = RandomForestClassifier()
    model_batters.fit(X_train_batters, y_train_batters)

    # Predictions
    batters['is_selected_prediction'] = model_batters.predict(X_batters)

    # Features and target for bowlers (use bowling-related metrics)
    X_bowlers = bowlers[['bowling_avg', 'wickets','catches']]
    y_bowlers = [1] * len(bowlers)  # For simplicity, assume they should all be selected (you can adjust this logic)

    # Train-test split
    X_train_bowlers, X_test_bowlers, y_train_bowlers, y_test_bowlers = train_test_split(X_bowlers, y_bowlers, test_size=0.2)

    # Train the model
    model_bowlers = RandomForestClassifier()
    model_bowlers.fit(X_train_bowlers, y_train_bowlers)

    # Predictions
    bowlers['is_selected_prediction'] = model_bowlers.predict(X_bowlers)

    # Features and target for all-rounders
    X_allrounders = allrounders[['batting_avg', 'runs', 'bowling_avg', 'wickets','strike_rate','catches']]
    y_allrounders = [1] * len(allrounders)  # Adjust this as necessary

    # Train-test split
    X_train_allrounders, X_test_allrounders, y_train_allrounders, y_test_allrounders = train_test_split(X_allrounders, y_allrounders, test_size=0.2)

    # Train the model
    model_allrounders = RandomForestClassifier()
    model_allrounders.fit(X_train_allrounders, y_train_allrounders)

    # Predictions
    allrounders['is_selected_prediction'] = model_allrounders.predict(X_allrounders)
    # Features and target for all-rounders
    X_allrounders = allrounders[['batting_avg', 'runs', 'bowling_avg', 'wickets','catches']]
    y_allrounders = [1] * len(allrounders)  # Adjust this as necessary

    # Train-test split
    X_train_allrounders, X_test_allrounders, y_train_allrounders, y_test_allrounders = train_test_split(X_allrounders, y_allrounders, test_size=0.2)

    # Train the model
    model_allrounders = RandomForestClassifier()
    model_allrounders.fit(X_train_allrounders, y_train_allrounders)

    # Predictions
    allrounders['is_selected_prediction'] = model_allrounders.predict(X_allrounders)


        # Combine all the selected players
    final_players = pd.concat([batters, bowlers, allrounders, wicketkeepers])
    # breakpoint()

    # Filter the final players who are predicted to be selected
    final_team = final_players[final_players['is_selected_prediction'] == 1]

    # Apply the role and team constraints (similar to previous examples)
    def apply_constraints(final_team):
        # Ensure at least 1 WK, 1 Batsman, 1 All-rounder, 1 Bowler
        wk = final_team[final_team['role'] == 'WK'].head(1)
        batsman = final_team[final_team['role'] == 'Batsman'].head(1)
        allrounder = final_team[final_team['role'] == 'All-rounder'].head(1)
        bowler = final_team[final_team['role'] == 'Bowler'].head(1)

        # Get the remaining top players to fill the team up to 11 players
        remaining_players = final_team[~final_team['player_name'].isin(wk['player_name'].tolist() +
                                                                    batsman['player_name'].tolist() +
                                                                    allrounder['player_name'].tolist() +
                                                                    bowler['player_name'].tolist())]
        top_remaining_players = remaining_players.head(7)

        # Combine all players into the final team
        final_team_selected = pd.concat([wk, batsman, allrounder, bowler, top_remaining_players])

        # Ensure no more than 10 players from the same team
        team_counts = final_team_selected['team'].value_counts()
        if team_counts.max() > 10:
            excess_team = team_counts.idxmax()  # The team with more than 10 players
            excess_players = final_team_selected[final_team_selected['team'] == excess_team].tail(team_counts.max() - 10)
            final_team_selected = final_team_selected.drop(excess_players.index)

            # Replace the dropped players with the next best players from other teams
            replacement_players = remaining_players[remaining_players['team'] != excess_team].head(len(excess_players))
            final_team_selected = pd.concat([final_team_selected, replacement_players])

        return final_team_selected

    # Get the final team
    final_11_team = apply_constraints(final_team)

    # Display the final selected team
    print(final_11_team[['player_name', 'role', 'team', 'is_selected_prediction']])

    return Response({"code":200,"message":final_11_team})


############################################################


##################################################################################
from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import StandardScaler

@api_view(['POST'])
def chatgpt_canvas_best11(request):
    file = request.FILES['file']
    df = pd.read_excel(file)

    # Step 2: Preprocess the dataset
    # Replace missing values with 0
    df = df.fillna(0)

    # Example features - Adjust columns based on the API response
    df = df[['player_name', 'team', 'role', 'batting_avg', 'runs', 'bowling_avg', 'wickets', 'catches', 'strike_rate', 'stumps']]

    # Step 3: Feature engineering - Convert the role to one-hot encoding
    df = pd.get_dummies(df, columns=['role'])

    # Step 4: Split the data by role to train separate models for each role
    batters = df[df['role_Batsman'] == 1]
    bowlers = df[df['role_Bowler'] == 1]
    allrounders = df[df['role_All-rounder'] == 1]
    wicketkeepers = df[df['role_WK'] == 1]

    def train_role_based_model(role_df, feature_columns, is_bowler=False):
        if len(role_df) == 0:
            return role_df
        
        X = role_df[feature_columns]
        y = np.random.randint(0, 2, size=len(role_df))  # Random labels, replace with real historical data if available
        
        if is_bowler:
            # For bowlers, a lower bowling average is better, so we inverse the value to reflect that in training
            X['bowling_avg'] = X['bowling_avg'].apply(lambda x: 1 / (x + 1e-5))
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        accuracy = model.score(X_test, y_test)
        print(f"Model accuracy for role: {accuracy * 100:.2f}%")
        
        # Make predictions
        role_df['is_selected_prediction'] = model.predict(X)
        return role_df

    # Train separate models for each role
    batter_features = ['batting_avg', 'runs', 'strike_rate', 'catches']
    bowler_features = ['bowling_avg', 'wickets', 'catches']
    allrounder_features = ['batting_avg', 'runs', 'bowling_avg', 'wickets', 'strike_rate', 'catches']
    wicketkeeper_features = ['batting_avg', 'runs', 'catches', 'stumps']

    batters = train_role_based_model(batters, batter_features)
    bowlers = train_role_based_model(bowlers, bowler_features, is_bowler=True)
    allrounders = train_role_based_model(allrounders, allrounder_features)
    wicketkeepers = train_role_based_model(wicketkeepers, wicketkeeper_features)

    # Combine all the selected players
    final_players = pd.concat([batters, bowlers, allrounders, wicketkeepers])
    print(final_players,"111")
    # Filter the final players who are predicted to be selected
    df_selected = final_players[final_players['is_selected_prediction'] == 1].reset_index(drop=True)

    # Step 8: Apply additional constraints to create the best 11 team
    # Ensure that the final selection includes at least one WK, one batsman, one all-rounder, and one bowler
    wk = df_selected[df_selected['role_WK'] == 1].sort_values(by=['batting_avg', 'catches', 'stumps'], ascending=False).head(1).reset_index(drop=True)  # Select 1 WK
    batsman = df_selected[df_selected['role_Batsman'] == 1].sort_values(by=['batting_avg', 'runs', 'strike_rate'], ascending=False).head(1).reset_index(drop=True)  # Select 1 Batsman
    allrounder = df_selected[df_selected['role_All-rounder'] == 1].sort_values(by=['batting_avg', 'wickets'], ascending=False).head(1).reset_index(drop=True)  # Select 1 All-rounder
    bowler = df_selected[df_selected['role_Bowler'] == 1].sort_values(by=['bowling_avg', 'wickets'], ascending=[True, False]).head(1).reset_index(drop=True)  # Select 1 Bowler

    # Ensure at least one player from each role; if not enough players, add more from the remaining pool
    remaining_players = df_selected[~df_selected['player_name'].isin(wk['player_name'].tolist() +
                                                                     batsman['player_name'].tolist() +
                                                                     allrounder['player_name'].tolist() +
                                                                     bowler['player_name'].tolist())].reset_index(drop=True)

    if len(wk) == 0 and len(remaining_players) > 0:
        wk = remaining_players[remaining_players['role_WK'] == 1].head(1)
        remaining_players = remaining_players.drop(wk.index).reset_index(drop=True)
    if len(batsman) == 0 and len(remaining_players) > 0:
        batsman = remaining_players[remaining_players['role_Batsman'] == 1].head(1)
        remaining_players = remaining_players.drop(batsman.index).reset_index(drop=True)
    if len(allrounder) == 0 and len(remaining_players) > 0:
        allrounder = remaining_players[remaining_players['role_All-rounder'] == 1].head(1)
        remaining_players = remaining_players.drop(allrounder.index).reset_index(drop=True)
    if len(bowler) == 0 and len(remaining_players) > 0:
        bowler = remaining_players[remaining_players['role_Bowler'] == 1].head(1)
        remaining_players = remaining_players.drop(bowler.index).reset_index(drop=True)

    # Recalculate the number of remaining players needed to ensure we have 11 players in total
    num_remaining_needed = 11 - (len(wk) + len(batsman) + len(allrounder) + len(bowler))
    if num_remaining_needed > 0:
        top_remaining_players = remaining_players.head(num_remaining_needed).reset_index(drop=True)
    else:
        top_remaining_players = pd.DataFrame()

    # Combine all selected players into the final team
    best_11_players = pd.concat([wk, batsman, allrounder, bowler, top_remaining_players]).reset_index(drop=True)

    # Ensure no more than 10 players from one team in the final selection
    team_counts = best_11_players['team'].value_counts()
    teams_to_limit = team_counts[team_counts > 10].index
    for team in teams_to_limit:
        excess_players = best_11_players[best_11_players['team'] == team].tail(team_counts[team] - 10)
        best_11_players = best_11_players.drop(excess_players.index).reset_index(drop=True)

        # Replace the dropped players with the next best players from other teams
        replacement_players = remaining_players[remaining_players['team'] != team].head(len(excess_players))
        best_11_players = pd.concat([best_11_players, replacement_players]).reset_index(drop=True)

    # Ensure the final team has exactly 11 players
    while len(best_11_players) < 11 and not remaining_players.empty:
        additional_player = remaining_players.head(1)
        best_11_players = pd.concat([best_11_players, additional_player]).reset_index(drop=True)
        remaining_players = remaining_players.drop(additional_player.index).reset_index(drop=True)

    # Step 9: Print the best 11 players
    dream11_team = best_11_players[['player_name', 'team', 'batting_avg', 'runs', 'bowling_avg', 'wickets', 'catches', 'strike_rate', 'stumps'] + [col for col in best_11_players.columns if 'role_' in col]]

    # Step 10: Select Captain and Vice-Captain using Machine Learning Model
    # Define features for captain and vice-captain selection
    captain_features = ['batting_avg', 'runs', 'bowling_avg', 'wickets', 'catches', 'strike_rate', 'stumps']
    X_captain = best_11_players[captain_features]
    y_captain = np.random.randint(0, 2, size=len(best_11_players))  # Generate labels randomly (replace with actual historical data if available)

    # Train-test split for captain selection
    X_train_captain, X_test_captain, y_train_captain, y_test_captain = train_test_split(X_captain, y_captain, test_size=0.2, random_state=42)

    # Train a logistic regression model to predict captain and vice-captain
    scaler = StandardScaler()
    X_train_captain_scaled = scaler.fit_transform(X_train_captain)
    X_test_captain_scaled = scaler.transform(X_test_captain)

    captain_model = LogisticRegression(random_state=42)
    captain_model.fit(X_train_captain_scaled, y_train_captain)

    # Predict probabilities for all players
    captain_probabilities = captain_model.predict_proba(scaler.transform(X_captain))[:, 1]
    best_11_players['captain_score'] = captain_probabilities

    # Sort players by captain score to determine captain and vice-captain
    sorted_players = best_11_players.sort_values(by='captain_score', ascending=False)
    captain = sorted_players.head(1)
    vice_captain = sorted_players.iloc[1:2]

    # Ensure captain and vice-captain are not the same player
    if captain['player_name'].values[0] == vice_captain['player_name'].values[0]:
        vice_captain = sorted_players.iloc[2:3]

    # Add Captain and Vice-Captain information to the final output
    dream11_team['captain'] = dream11_team['player_name'].isin(captain['player_name']).astype(int)
    dream11_team['vice_captain'] = dream11_team['player_name'].isin(vice_captain['player_name']).astype(int)

    print("Best 11 Players for Dream 11:")
    print(dream11_team)
    return Response(dream11_team.to_dict(orient='records'))



###############################################################################################################################
@api_view(['GET'])
def players_squad_list(request,match_id):
    base_url="https://api.cricapi.com/v1/match_squad"
    apikey="*******************************************"  #Give your api key

    response = requests.get(base_url, params={'apikey': apikey, 'id': match_id})
            
    if response.status_code == 200:
        # breakpoint()
        teams = response.json()['data'] 
    else:
        return Response({"code":404,"data":"api error"})
    # return Response({"code":200,"data":data})
    # breakpoint()
    print(teams,"hhhhhhhhhhhhhhhhhhhhhhhhhhhh")
    return render(request, 'match_detail.html', {'teams': teams})


@api_view(['GET'])
def seriesMatchesListApi(request):
    from datetime import datetime,timedelta

    # Get the current date
    current_date_1= datetime.now().date()

    # Add two days
    new_date = current_date_1 + timedelta(days=7)

    # Convert to string if needed
    currentdate = new_date.strftime('%Y-%m-%d')
    print(currentdate)
    base_url="https://api.cricapi.com/v1/series"
    apikey="************************************" #give your api key

    response = requests.get(base_url, params={'apikey': apikey, 'search': currentdate})
            
    if response.status_code == 200:
        data = response.json()['data'] 
        seriesidList = [item['id'] for item in data]
        # print(seriesidList)
    else:
        return Response({"data":"Api not working"})
    baseurl2 = "https://api.cricapi.com/v1/series_info"
    matchesList = []
    for i in seriesidList:
        response = requests.get(baseurl2, params={'apikey': apikey, 'id': i})
        if response.status_code == 200:
            data = response.json()['data']
            # print(data)
            matchesList.extend(data['matchList'])
            # for ii in data:
            # breakpoint()
    matches_not_ended = [match for match in matchesList if not (match["matchStarted"])]
    # matches_not_ended=matchesList


    for match in matches_not_ended:
        # Original UTC datetime string from the match
        utc_time_str = match['dateTimeGMT']
        
        # Convert the string to a datetime object
        utc_time = datetime.strptime(utc_time_str, "%Y-%m-%dT%H:%M:%S")
        
        # Set the timezone to UTC
        utc_time = utc_time.replace(tzinfo=pytz.UTC)
        
        # Convert UTC to IST
        ist_time = utc_time.astimezone(pytz.timezone('Asia/Kolkata'))
        
        # Add the converted IST time to the match dictionary
        match['dateTimeIST'] = ist_time.strftime("%d-%m-%Y %H:%M")

    # Print matches for debugging (optional)
    print("Matches with IST Time:", matches_not_ended)
    # breakpoint()

   
    return render(request, 'matchlist.html', {'matches': matches_not_ended})






@api_view(['POST'])
def cricketRealDataAiteam(request):
    converted_list = request.POST.getlist('player_ids')
    print(converted_list,"converted_list")
    matchtype = request.POST.get('match_type')   
    # API Key and base URL
    apikey = '**************************************' # Give your api key
    base_url = 'https://api.cricapi.com/v1/players_info'
    
    # List to store all player information
    all_players_info = []

    # Loop through each player ID and fetch their details
    for player_id in converted_list:
        try:
            # Make the request to the API for each player
            response = requests.get(base_url, params={'apikey': apikey, 'id': player_id})
            
            if response.status_code == 200:
                player_data = response.json()['data']  # Extract player data
                # print(player_data)
                #HTML side this data come
                role_key = f'player_roles_{player_id}'
                player_role_from_html = request.POST.get(role_key)
                # breakpoint()
                # Initialize a dictionary to store player details
                player_info = {
                    "player_name": player_data["name"],
                    "batting_avg": None,
                    "bowling_avg": None,
                    "strike_rate": None,
                    "runs": None,
                    "wickets": None,
                    "role": player_role_from_html,
                    "team": player_data["country"],
                    "playerImg":player_data["playerImg"]
                }

                # Extract match-specific stats
                for stat in player_data["stats"]:
                    if matchtype == stat["matchtype"]:
                        if stat["fn"] == "batting":
                            if stat["stat"] == "avg":
                                player_info["batting_avg"] = stat["value"]
                            if stat["stat"] == "sr":
                                player_info["strike_rate"] = stat["value"]
                            if stat["stat"] == "runs":
                                player_info["runs"] = stat["value"]
                        if stat["fn"] == "bowling":
                            if stat["stat"] == "avg":
                                player_info["bowling_avg"] = stat["value"]
                            if stat["stat"] == "wkts":
                                player_info["wickets"] = stat["value"]

                # Append player info to the list
                all_players_info.append(player_info)

            else:
                print(f"Failed to fetch details for player ID {player_id}")
        
        except Exception as e:
            print(f"Error occurred for player ID {player_id}: {str(e)}")
    
    df = pd.DataFrame(all_players_info)
    # print(all_players_info)
    # Step 2: Preprocess the dataset
    # Convert the appropriate columns to numeric and handle non-numeric values
    for col in ['batting_avg', 'bowling_avg', 'strike_rate', 'runs', 'wickets']:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert non-numeric values like '-' to NaN

    # Replace NaN values with 0
    df.fillna(0, inplace=True)

    # Replace specific role values in the DataFrame
    # print(df,"df")
    df['role'] = df['role'].replace({
        'Batting Allrounder': 'All-rounder',
        'Bowling Allrounder': 'All-rounder',
        'WK-Batsman': 'WK'
    })

    # Step 3: Feature engineering - Correct role labeling before one-hot encoding
    df['is_batsman'] = df['role'].apply(lambda x: 1 if x == 'Batsman' else 0)
    df['is_bowler'] = df['role'].apply(lambda x: 1 if x == 'Bowler' else 0)
    df['is_allrounder'] = df['role'].apply(lambda x: 1 if x == 'All-rounder' else 0)
    df['is_wicketkeeper'] = df['role'].apply(lambda x: 1 if x == 'WK' else 0)

    # Normalize numerical features to make the model more stable
    scaler = StandardScaler()
    df[['batting_avg', 'runs', 'bowling_avg', 'wickets', 'strike_rate']] = scaler.fit_transform(df[['batting_avg', 'runs', 'bowling_avg', 'wickets', 'strike_rate']])

    # Step 4: Split the data by role to train separate models for each role
    batters = df[df['is_batsman'] == 1]
    bowlers = df[df['is_bowler'] == 1]
    allrounders = df[df['is_allrounder'] == 1]
    wicketkeepers = df[df['is_wicketkeeper'] == 1]

    def train_role_based_model(role_df, feature_columns, is_bowler=False):
        if len(role_df) == 0:
            return role_df

        X = role_df[feature_columns]
        y = np.random.randint(0, 2, size=len(role_df))  # Random labels, replace with real historical data if available

        # If it's a bowler, apply the inverse transformation for bowling average
        if is_bowler and 'bowling_avg' in feature_columns:
            X['bowling_avg'] = X['bowling_avg'].apply(lambda x: 1 / (x + 1e-5) if x != 0 else 0)

        # Check if there are enough samples to perform train-test split
        if len(X) > 1:
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            # Not enough samples for splitting, use all data for both training and testing
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        if len(X_test) > 0:
            accuracy = model.score(X_test, y_test)
            print(f"Model accuracy for role: {accuracy * 100:.2f}%")

        # Make predictions
        role_df['is_selected_prediction'] = model.predict(X)
        
        return role_df

    # Train separate models for each role
    batter_features = ['batting_avg', 'runs', 'strike_rate']
    bowler_features = ['bowling_avg', 'wickets']
    allrounder_features = ['batting_avg', 'runs', 'bowling_avg', 'wickets', 'strike_rate']
    wicketkeeper_features = ['batting_avg', 'runs']

    batters = train_role_based_model(batters, batter_features)
    bowlers = train_role_based_model(bowlers, bowler_features, is_bowler=True)
    allrounders = train_role_based_model(allrounders, allrounder_features)
    wicketkeepers = train_role_based_model(wicketkeepers, wicketkeeper_features)

    # Combine all the selected players
    final_players = pd.concat([batters, bowlers, allrounders, wicketkeepers])
    df_selected = final_players[(final_players['is_selected_prediction'] == 1)].reset_index(drop=True)

    # Ensure at least one player from each role is selected
    def ensure_minimum_role(df, role_df, role_column):
        if len(df[df[role_column] == 1]) == 0 and len(role_df) > 0:
            additional_player = role_df.head(1)
            df = pd.concat([df, additional_player]).reset_index(drop=True)
        return df

    # Make sure we have at least one player from each role
    df_selected = ensure_minimum_role(df_selected, wicketkeepers, 'is_wicketkeeper')
    df_selected = ensure_minimum_role(df_selected, batters, 'is_batsman')
    df_selected = ensure_minimum_role(df_selected, allrounders, 'is_allrounder')
    df_selected = ensure_minimum_role(df_selected, bowlers, 'is_bowler')

    # Ensure there are exactly 11 players in the final selection
    if len(df_selected) < 11:
        remaining_players = final_players[~final_players['player_name'].isin(df_selected['player_name'])].reset_index(drop=True)
        additional_needed = 11 - len(df_selected)
        df_selected = pd.concat([df_selected, remaining_players.head(additional_needed)]).reset_index(drop=True)
    elif len(df_selected) > 11:
        # Prioritize keeping at least one player from each role while reducing to 11 players
        role_priority_order = ['is_wicketkeeper', 'is_batsman', 'is_allrounder', 'is_bowler']
        for role in role_priority_order:
            while len(df_selected) > 11 and len(df_selected[df_selected[role] == 1]) > 1:
                df_selected = df_selected.drop(df_selected[df_selected[role] == 1].tail(1).index).reset_index(drop=True)
        # If still more than 11 players, drop from the end
        if len(df_selected) > 11:
            df_selected = df_selected.head(11)

    # Select Captain and Vice-Captain using Random Forest Classifier with Role-Based Weightage
    df_selected['inverse_bowling_avg'] = df_selected['bowling_avg'].apply(lambda x: 1 / (x + 1e-5) if x != 0 else 0)
    df_selected['weighted_score'] = (
        (df_selected['batting_avg'] * 0.7 + df_selected['runs'] * 0.3) * df_selected['is_batsman'] +
        (df_selected['wickets'] * 0.5 + df_selected['inverse_bowling_avg'] * 0.5) * df_selected['is_bowler'] +
        (df_selected['batting_avg'] * 0.3 + df_selected['bowling_avg'] * 0.3 + df_selected['wickets'] * 0.4) * df_selected['is_allrounder'] +
        (df_selected['batting_avg'] * 0.5 + df_selected['runs'] * 0.5) * df_selected['is_wicketkeeper'] +
        df_selected['strike_rate'] * 0.2
    )

    # Train-test split for captain selection
    X = df_selected[['batting_avg', 'runs', 'wickets', 'strike_rate', 'inverse_bowling_avg', 'is_batsman', 'is_bowler', 'is_allrounder', 'is_wicketkeeper']]
    y = df_selected['weighted_score'] > df_selected['weighted_score'].median()
    y = y.astype(int)

    # Train a Random Forest Classifier model to predict captain and vice-captain
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    captain_model = RandomForestClassifier(n_estimators=100, random_state=42)
    captain_model.fit(X_train, y_train)

    # Predict probabilities for all players
    captain_probabilities = captain_model.predict_proba(X)[:, 1]
    df_selected['captain_score'] = captain_probabilities

    # Sort players by captain score to determine captain and vice-captain
    sorted_players = df_selected.sort_values(by='captain_score', ascending=False)
    captain = sorted_players.head(1)
    vice_captain = sorted_players.iloc[1:2] if len(sorted_players) > 1 else None

    # Assign captain and vice-captain roles
    df_selected['captain'] = df_selected['player_name'].isin(captain['player_name']).astype(int)
    if vice_captain is not None:
        df_selected['vice_captain'] = df_selected['player_name'].isin(vice_captain['player_name']).astype(int)
    else:
        df_selected['vice_captain'] = 0

    # Prepare final team output
    finalteam = df_selected.to_dict(orient='records')
    
    for noneedkeys in finalteam:
        noneedkeys.pop('batting_avg', None)
        noneedkeys.pop('bowling_avg', None)
        noneedkeys.pop('strike_rate', None)
        noneedkeys.pop('runs', None)  # Remove 'bowling_avg' key
        noneedkeys.pop('wickets', None)
        noneedkeys.pop('is_batsman', None)
        noneedkeys.pop('is_bowler', None)
        noneedkeys.pop('is_allrounder', None)
        noneedkeys.pop('is_selected_prediction', None)
        noneedkeys.pop('is_wicketkeeper', None)
        noneedkeys.pop('inverse_bowling_avg', None)
        noneedkeys.pop('weighted_score', None)
        noneedkeys.pop('captain_score', None)
    # print(finalteam)

    return render(request,"final_team.html",{"data": finalteam})






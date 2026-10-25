import pandas as pd

df_left_2015 = pd.read_excel('../data/raw/2015.xlsx')
df_left_2016 = pd.read_excel('../data/raw/2016.xlsx')
df_left_2017 = pd.read_excel('../data/raw/2017.xlsx')
df_left_2018 = pd.read_excel('../data/raw/2018.xlsx')
df_left_2019 = pd.read_excel('../data/raw/2019.xlsx')
df_left_2020 = pd.read_excel('../data/raw/2020.xlsx')
df_left_2021 = pd.read_excel('../data/raw/2021.xlsx')
df_left_2022 = pd.read_excel('../data/raw/2022.xlsx')
df_left_2023 = pd.read_excel('../data/raw/2023.xlsx')
df_left_2024 = pd.read_excel('../data/raw/2024.xlsx')
full_left_df = pd.concat([df_left_2015, df_left_2016, df_left_2017, df_left_2018, df_left_2019, df_left_2020, df_left_2021, df_left_2022, df_left_2023, df_left_2024],
    ignore_index=True)
df_right_2015 = pd.read_csv('../data/raw/atp_matches_2015.csv')
df_right_2016 = pd.read_csv('../data/raw/atp_matches_2016.csv')
df_right_2017 = pd.read_csv('../data/raw/atp_matches_2017.csv')
df_right_2018 = pd.read_csv('../data/raw/atp_matches_2018.csv')
df_right_2019 = pd.read_csv('../data/raw/atp_matches_2019.csv')
df_right_2020 = pd.read_csv('../data/raw/atp_matches_2020.csv')
df_right_2021 = pd.read_csv('../data/raw/atp_matches_2021.csv')
df_right_2022 = pd.read_csv('../data/raw/atp_matches_2022.csv')
df_right_2023 = pd.read_csv('../data/raw/atp_matches_2023.csv')
df_right_2024 = pd.read_csv('../data/raw/atp_matches_2024.csv')
full_right_df = pd.concat([df_right_2015, df_right_2016, df_right_2017, df_right_2018, df_right_2019, df_right_2020, df_right_2021, df_right_2022, df_right_2023, df_right_2024],
    ignore_index=True)

# Create lookup dictionaries for winner stats
winner_lookup = {}
for _, row in full_right_df.iterrows():
    # Convert "firstname lastname" to "Lastname F." format
    full_name = str(row['winner_name']).strip()
    if ' ' in full_name:
        parts = full_name.split()
        first_name = parts[0]
        last_name = ' '.join(parts[1:])  # Handle multi-part last names
        formatted_name = f"{last_name} {first_name[0]}."
    else:
        formatted_name = full_name  # Keep as is if no space found
    
    winner_lookup[formatted_name] = {
        'hand': row.get('winner_hand', None),
        'ioc': row.get('winner_ioc', None),
        'player_id': row.get('winner_id', None),
        'height': row.get('winner_ht', None),
        'age': row.get('winner_age', None)
    }

# Create lookup dictionaries for loser stats
loser_lookup = {}
for _, row in full_right_df.iterrows():
    # Convert "firstname lastname" to "Lastname F." format
    full_name = str(row['loser_name']).strip()
    if ' ' in full_name:
        parts = full_name.split()
        first_name = parts[0]
        last_name = ' '.join(parts[1:])  # Handle multi-part last names
        formatted_name = f"{last_name} {first_name[0]}."
    else:
        formatted_name = full_name  # Keep as is if no space found
    
    loser_lookup[formatted_name] = {
        'hand': row.get('loser_hand', None),
        'ioc': row.get('loser_ioc', None),
        'player_id': row.get('loser_id', None),
        'height': row.get('loser_ht', None),
        'age': row.get('loser_age', None)
    }

# Combine both lookups into one comprehensive player lookup
# (since a player can be both a winner and loser in different matches)
all_player_lookup = {}
for name, stats in winner_lookup.items():
    all_player_lookup[name] = stats
for name, stats in loser_lookup.items():
    if name not in all_player_lookup:
        all_player_lookup[name] = stats

# Add winner stats to matches
full_left_df['Winner_hand'] = full_left_df['Winner'].map(lambda x: all_player_lookup.get(x, {}).get('hand', None))
full_left_df['Winner_ioc'] = full_left_df['Winner'].map(lambda x: all_player_lookup.get(x, {}).get('ioc', None))
full_left_df['Winner_player_id'] = full_left_df['Winner'].map(lambda x: all_player_lookup.get(x, {}).get('player_id', None))
full_left_df['Winner_height'] = full_left_df['Winner'].map(lambda x: all_player_lookup.get(x, {}).get('height', None))
full_left_df['Winner_age'] = full_left_df['Winner'].map(lambda x: all_player_lookup.get(x, {}).get('age', None))

# Add loser stats to matches
full_left_df['Loser_hand'] = full_left_df['Loser'].map(lambda x: all_player_lookup.get(x, {}).get('hand', None))
full_left_df['Loser_ioc'] = full_left_df['Loser'].map(lambda x: all_player_lookup.get(x, {}).get('ioc', None))
full_left_df['Loser_player_id'] = full_left_df['Loser'].map(lambda x: all_player_lookup.get(x, {}).get('player_id', None))
full_left_df['Loser_height'] = full_left_df['Loser'].map(lambda x: all_player_lookup.get(x, {}).get('height', None))
full_left_df['Loser_age'] = full_left_df['Loser'].map(lambda x: all_player_lookup.get(x, {}).get('age', None))

# Dictionary to store all ranking dataframes
ranking_data = {}

# Read all ranking files (2015-2024)
for year in range(2015, 2025):
    try:
        df = pd.read_csv(f'../data/raw/MentalToughness_{year}.csv')
        df.columns = df.columns.str.strip()
        ranking_data[year] = df
    except FileNotFoundError:
        print(f"Warning: MentalToughness_{year}.csv not found")

# Function to convert "firstname lastname" to "Lastname F." format
def format_name(full_name):
    """Convert 'firstname lastname' to 'Lastname F.' format"""
    full_name = str(full_name).strip()
    if ' ' in full_name:
        parts = full_name.split()
        first_name = parts[0]
        last_name = ' '.join(parts[1:])  # Handle multi-part last names
        return f"{last_name} {first_name[0]}."
    return full_name

# Create ranking lookups for each year
ranking_lookups = {}
for year, df in ranking_data.items():
    lookup = {}
    for _, row in df.iterrows():
        player_name = row.get('name', None)
        if player_name:
            formatted_name = format_name(player_name)
            rating = row.get('rating', None)
            lookup[formatted_name] = rating
    ranking_lookups[year] = lookup

# Function to extract year from date string (format: day/month/year)
def extract_year_from_date(date_str):
    """Extract year from date string in format 'day/month/year'"""
    try:
        # Parse the date and extract year
        date_obj = pd.to_datetime(date_str, format='%d/%m/%Y', errors='coerce')
        if pd.notna(date_obj):
            return int(date_obj.year)
    except:
        pass
    return None

# Function to get rating for a player in a specific year
def get_rating(player_name, year):
    """Get player rating for a specific year"""
    if year and year in ranking_lookups:
        return ranking_lookups[year].get(player_name, None)
    return None

# Add winner and loser ratings to full_left_df
# Extract year from date on-the-fly for each row
full_left_df['Winner_rating'] = full_left_df.apply(
    lambda row: get_rating(row['Winner'], extract_year_from_date(row['Date'])), axis=1
)

full_left_df['Loser_rating'] = full_left_df.apply(
    lambda row: get_rating(row['Loser'], extract_year_from_date(row['Date'])), axis=1
)

# Save the enhanced dataset
full_left_df.to_csv('../data/processed/2024_with_player_stats.csv', index=False)

print("\n" + "="*50)
print(f"Total matches: {len(full_left_df)}")
print(f"\nMatches with missing winner rating: {full_left_df['Winner_rating'].isna().sum()}")
print(f"Matches with missing loser rating: {full_left_df['Loser_rating'].isna().sum()}")

print(f"\nTotal matches: {len(full_left_df)}")
print(f"Unique players found: {len(all_player_lookup)}")
print(f"\nNew columns added:")
print("- Winner_hand, Winner_ioc, Winner_player_id, Winner_height, Winner_rating")
print("- Loser_hand, Loser_ioc, Loser_player_id, Loser_height, Loser_rating")

# Check for any missing data
winner_missing = full_left_df['Winner_player_id'].isna().sum()
loser_missing = full_left_df['Loser_player_id'].isna().sum()
print(f"\nMatches with missing winner data: {winner_missing}")
print(f"Matches with missing loser data: {loser_missing}")

# Show a sample of the enhanced data
print(full_left_df.head())
import pandas as pd
from pathlib import Path
import numpy as np

def add_location_to_df(df, location_clean):
    """
    Add location as a new column to the dataframe
    """
    df['location'] = location_clean
    return df

def format_player_name(name):
    """
    Format player names according to tennis naming conventions.

    Rules:
      - If last name is hyphenated (e.g. Ramos-Vinolas), treat as first name.
      - Otherwise, last name = everything after the first name.
      - Each last name part is properly capitalized (e.g. 'del Potro' → 'Del Potro').
      - Hyphens in first name become spaces for multiple initials.

    Examples:
        'Pierre-Hugues Herbert'   -> 'Herbert P.H.'
        'Albert Ramos-Vinolas'    -> 'Ramos-Vinolas A.'
        'Roberto Bautista Agut'   -> 'Bautista Agut R.'
        'Juan Martin del Potro'   -> 'Del Potro J.'
    """
    parts = name.strip().split()

    if len(parts) < 2:
        return name  # handle single-word names

    first_name = parts[0].replace('-', ' ')
    last_name = ""

    # Case 1: Hyphenated last name -> treat as first name (Spanish style)
    if '-' in parts[-1]:
        last_name = parts[-1].title()

    # Case 2: Otherwise, all parts except the first are last name
    else:
        # Capitalize properly (e.g., 'del Potro' -> 'Del Potro')
        last_name_parts = [p.capitalize() for p in parts[1:]]
        last_name = ' '.join(last_name_parts)

    # Build initials from first name (can have multiple parts)
    initials = ''.join(f"{p[0].upper()}." for p in first_name.split())

    return f"{last_name} {initials}"

def process_year_data(year):
    """
    Process all CSV files for a given year and combine them
    """
    data_dir = Path(f"../data/ytd_data/{year}/")
    
    if not data_dir.exists():
        print(f"Directory for {year} doesn't exist")
        return None
    
    print(f"\n--- Processing Year {year} ---")
    
    # List to store all dataframes for this year
    year_dataframes = []
    
    for csv_file in data_dir.glob("*.csv"):
        if csv_file.is_symlink() and not csv_file.exists():
            continue
            
        try:
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Extract and clean location
            filename = csv_file.stem
            location = filename.rsplit('_', 1)[0]
            location_clean = (
                "'s-Hertogenbosch" if location == "s-hertogenbosch"
                else "Winston-Salem" if location == "winston-salem"
                else "St. Petersburg" if location == "st-petersburg" 
                else 'Rio de Janeiro' if location == 'rio-de-janeiro'
                else location.replace('-', ' ').title()
            )
            
            # Add location column to dataframe
            df = add_location_to_df(df, location_clean)
            
            # Optionally add year column too
            df['year'] = year
            
            # Add to list
            year_dataframes.append(df)
            
            # print(f"{location_clean}: {len(df)} rows")
            
        except Exception as e:
            print(f"Error with {csv_file.name}: {e}")
    
    # Concatenate all dataframes for this year
    if year_dataframes:
        combined_df = pd.concat(year_dataframes, ignore_index=True)
        print(f"  Total for {year}: {len(combined_df)} rows from {len(year_dataframes)} tournaments")
        return combined_df
    else:
        print(f"  No data found for {year}")
        return None

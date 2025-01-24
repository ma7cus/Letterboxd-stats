import pandas as pd
from average_rating_scraper import extract_average_ratings
from film_length_scraper import extract_film_lengths
from datetime import datetime

################################################################
# Function to merge Excel data
################################################################
def merge_excel_data(ratings_file_path, analysed_file_path):
    """
    Merges Excel data based on common columns and updates the DataFrame.

    Args:
        ratings_file_path (str): Path to the ratings file.
        analysed_file_path (str, optional): Path to the analysed file. Defaults to None.

    Returns:
        DataFrame: The merged DataFrame.
    """
    # Read ratings and analysed Excel files into pandas DataFrames
    ratings_df = pd.read_csv(ratings_file_path)
    analysed_df = pd.read_csv(analysed_file_path)

    # Merge DataFrames based on common columns
    merged_df = pd.merge(ratings_df, analysed_df, on=['Name', 'Year', 'Letterboxd URI'], how='outer', suffixes=('_ratings', '_new'))
     
    # Fill NaN values in Date and Rating columns from the ratings DataFrame with values from new DataFrame
    merged_df['Date'] = merged_df['Date_ratings'].fillna(merged_df['Date_new'])
    merged_df['Rating'] = merged_df['Rating_ratings'].fillna(merged_df['Rating_new'])
    
    # Drop unnecessary columns
    merged_df.drop(columns=['Date_ratings', 'Rating_ratings', 'Date_new', 'Rating_new'], inplace=True)
    
    # Reorder columns
    column_order = ['Date', 'Name', 'Year', 'Letterboxd URI', 'Rating', 'Average Rating', 'Length (mins)']
    merged_df = merged_df.reindex(columns=column_order)
    
    merged_df.to_csv('starting_df.csv', index=False)
    return merged_df
       
################################################################
# Function to update data
################################################################
def update_data(df):
    """
    Updates data by merging, extracting average ratings and film lengths, and saving to a new file.

    Args:
        ratings_file_path (str): Path to the ratings file.
        analysed_file_path (str, optional): Path to the analysed file. Defaults to None.

    Returns:
        DataFrame: The updated DataFrame.
    """
    average_ratings = extract_average_ratings(df)
    film_lengths = extract_film_lengths(df)

    # Update DataFrame with average ratings and film lengths
    df['Average Rating'] = average_ratings
    df['Length (mins)'] = film_lengths
    df['Length (mins)'] = pd.to_numeric(df['Length (mins)'], errors='coerce')

    # Get current date for output file name
    current_date = datetime.now().strftime('%d_%m_%Y')
    output_file_name = f'ratings_{current_date}.csv'

    # Save DataFrame to CSV file
    df.to_csv(output_file_name, index=False)
    print(f"New rows added and saved to '{output_file_name}'.")
    print("")
    return df


    
    

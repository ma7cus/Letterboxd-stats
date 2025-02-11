###############################################################################
# Inputs
###############################################################################
from conflation import merge_excel_data, update_data
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import skew
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings("ignore")

###############################################################################
# Processing the input to the correct form with updated values
###############################################################################

def import_and_process_data(downloaded_ratings_file, current_run_output_file, old_run_output_file=None):
    """
    Import and process data based on the files available and user inputs.

    Args:
        downloaded_ratings_file (str): Path to the most recently downloaded ratings data file.
        current_run_output_file (str): Path to the output file to create or use for this run.
        old_run_output_file (str, optional): Path to the analysed file from the most recent previous run.

    Returns:
        DataFrame: The processed DataFrame.
    """
    # Case 1: Use existing output file without updates
    if os.path.exists(current_run_output_file):
        print("Current run output file exists. Proceeding with existing dataframe.")
        df = pd.read_csv(current_run_output_file)
        return df

    # Case 2: Merge downloaded ratings with old run data, then update
    if os.path.exists(downloaded_ratings_file):
        if old_run_output_file and os.path.exists(old_run_output_file):
            print("Updating old data frame with new entries...")
            df = merge_excel_data(downloaded_ratings_file, old_run_output_file)
            df = update_data(df)
        else:
            # Case 3: Process only the downloaded ratings file
            print("Adding data to downloaded ratings file...")
            df = pd.read_csv(downloaded_ratings_file)
            df['Average Rating'] = None
            df['Length (mins)'] = None
            df = update_data(df)      

        return df

    # Case 4: Missing essential files
    print("Error: No valid input files found. Ensure the downloaded ratings file exists.")
    return None


###############################################################################  

###############################################################################
# Rating distribution histogram
###############################################################################

## Plotting the ratings distribution ##
def ratings_dist(df):
    """
    Plot the distribution of ratings and compute
    statistics such as mean, standard deviation, and skewness.
    """
    df_ratings = df['Rating']  # Split off just the ratings data from the dataframe
        
    axes = plt.axes()
    x_ticks = np.linspace(0.5, 5, 10)
    axes.set_xticks(x_ticks)
    plt.xlabel('Ratings')
    plt.ylabel('Frequency')
    plt.title('Ratings distribution for ma7cus')
    sns.histplot(data=df_ratings, bins=10, kde=True, label='Smoothed Histogram')  # Plot a histogram of the frequency data split into 10 classes (i.e. the 10 possible ratings)
    mu, std = norm.fit(df_ratings)
    skewness = skew(df_ratings)
    x = np.linspace(df_ratings.min(), df_ratings.max(), 100)
    pdf = norm.pdf(x, mu, std)
    plt.plot(x, pdf * len(df_ratings) / 10, 'r-', linewidth=2, label='Fitted Normal Distribution')
    
    plt.legend()
    plt.savefig('ratings_dist.pdf', format='pdf')
    plt.show()
    
    print("Ratings distribution: ")
    print(f"Mean: {mu:.{2}f} (2dp)")
    print(f"Standard Deviation: {std:.{2}f} (2dp)")
    print(f"Skewness: {skewness:.{2}f} (2dp)")
    print(" ")
    
###############################################################################

###############################################################################
# Year frequency distribution histogram
###############################################################################

## Plotting the year distribution ##
def release_year_dist(df):
    """
    Plot the distribution of years and compute the 
    most watched years.
    """
    df_years = df['Year']  # Split off just the ratings data from the dataframe

    max_year = df['Year'].max()
    min_year = df['Year'].min()
    year_range = max_year - min_year
    
    axes = plt.axes()
    x_ticks = np.linspace(0.5, 5, 10)
    axes.set_xticks(x_ticks)
    plt.xlabel('Years')
    plt.ylabel('Frequency')
    plt.title('Release year distribution')

    sns.histplot(data=df_years, bins=year_range, kde=True)  # Plot a histogram of the frequency data split into 10 classes (i.e. the 10 possible ratings)
    plt.savefig('year_dist.pdf', format='pdf')
    plt.show()
    
    year_counts = df_years.value_counts()
    sorted_counts = year_counts.sort_values(ascending=False)
    
    print("Most watched release years: ")
    print(f'{sorted_counts.index[0]} with {sorted_counts.iloc[0]} watches')
    print(f'{sorted_counts.index[1]} with {sorted_counts.iloc[1]} watches')
    print(f'{sorted_counts.index[2]} with {sorted_counts.iloc[2]} watches')
    print("")


#The function below is not used in the main script, but can be used to plot the cumulative frequency of films watched over time.
#I found it didn't work for my data as the ratings data is just the date at which a film was logged, not watched 
# so there were days where I 'watched' hundreds of films which didn't make sense.
"""    
def watch_frequency_dist(df, time_interval='D'):
    Plots a cumulative frequency graph of the number of films watched over time.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing film watch data with a 'Date' column.
    time_interval (str): Pandas offset alias (e.g., 'D' for daily, 'W' for weekly, 'M' for monthly).
    
    # Ensure the 'Date' column is in datetime format, handling inconsistencies
    df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='mixed', dayfirst=True, errors='coerce')
    
    # Drop rows with NaT values in 'Date'
    df = df.dropna(subset=['Date'])
    
    # Count the number of films watched per specified time interval
    df_grouped = df.groupby(pd.Grouper(key='Date', freq=time_interval)).size()
    
    # Compute cumulative sum
    df_cumulative = df_grouped.cumsum()
    
    # Find the n most watched time intervals
    number_of_elements = 5
    most_watched_days = df_grouped.nlargest(number_of_elements)
    print(f'Top 3 most watched {time_interval}:')
    print(most_watched_days)
    
    # Plot cumulative frequency
    plt.figure(figsize=(10, 5))
    plt.plot(df_cumulative.index, df_cumulative.values, linestyle='-')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Count of Films Watched')
    plt.title(f'Cumulative Films Watched Over Time ({time_interval})')
    plt.legend()
    plt.grid()
    plt.show()
"""
###############################################################################

###############################################################################
# Favourite movies by year spreadsheet
###############################################################################
    
def find_nth_favorite_movie(year_table, n, min_rating): 
    """
    Find the nth favorite movie given a dataframe
    with movies from a specific year and a minimum rating threshold.
    """
    num_films_seen = len(year_table)  # Calculate the number of films seen
    
    if num_films_seen >= n:
        sorted_table = year_table.sort_values(by='Rating', ascending=False)
        nth_favorite_movie_row = sorted_table.iloc[n-1]
        if nth_favorite_movie_row['Rating'] >= min_rating:
            return nth_favorite_movie_row[['Name', 'Rating']]  # Return the row with the correct labels
        else:
            return pd.Series([None, None], index=['Name', 'Rating'])
    else:
        return pd.Series([None, None], index=['Name', 'Rating'])

def print_nth_favourite_movies(df, n, min_rating):
    """
    Print and export to Excel the top n favorite movies
    for each year based on their rating.
    """
    df_year_grouped = df.groupby('Year')  # Splits the full table into a superset, the elements of which are the tables where the year is all the same.

    earliest_year = df['Year'].min()
    latest_year = df['Year'].max()
    
    df_year_favourites = pd.DataFrame(columns=['Year'])
    
    column_labels = ['Year']
    for i in range(1, n + 1):
        column_labels += [f'Favorite film {i}', f'Favorite film rating {i}']

    for i in range(1, n+1):
        df_year_favourites[f'Favorite film {i}'] = ''
        df_year_favourites[f'Favorite film rating {i}'] = ''


    for year in range(earliest_year, latest_year+1):  # Iterates through all years between highest and lowest inclusive
        if year in df_year_grouped.groups:  # If any movies were seen, proceed, else return null row
            year_table = df_year_grouped.get_group(year)  # Pulls the table for the current year
            row_data = [year]
            for i in range(1, n+1):
                favorite_movie_row = find_nth_favorite_movie(year_table, i, min_rating)
                if favorite_movie_row is not None:
                    row_data += list(favorite_movie_row)
                else:
                    row_data += [None, None]
                    
            df_year_favourites.loc[len(df_year_favourites)] = row_data
        else:
            null_row = pd.Series([year] + [None] * (2 * n), index=column_labels)

            df_year_favourites = df_year_favourites._append(null_row, ignore_index=True)


    df_year_favourites.to_csv(f'favourite_{n}_movies_by_year.csv', index=False)
    
    return df_year_favourites


###############################################################################

###############################################################################
# Favourite year by movie spreadsheet
###############################################################################

def print_favourite_year(df, n):
    """
    Print and export to an Excel file the favorite year based
    on the average rating of the top n films for each year.
    """
    df_year_grouped = df.groupby('Year')
    earliest_year = df['Year'].min()
    latest_year = df['Year'].max()
    
    df_favourite_year = pd.DataFrame(columns=['Year', f'Average rating of top {n} films'])

    for year in range(earliest_year, latest_year + 1):
        if year in df_year_grouped.groups:
            year_table = df_year_grouped.get_group(year)
            favourite_movies = []
            for i in range(1, n+1):
                ith_favourite_movie = find_nth_favorite_movie(year_table, i, 0.5)
                ith_favourite_movie_rating = ith_favourite_movie['Rating']
                if ith_favourite_movie_rating is not None:
                    favourite_movies.append(ith_favourite_movie_rating)
                else:
                    favourite_movies.append(None)
            
            # Check if the list contains any None values
            if any(movie is None for movie in favourite_movies):
                average = None
            else:
                average = np.mean(favourite_movies)
           
            df_favourite_year.loc[len(df_favourite_year)] = [year, average]
        else:
            null_row = pd.Series([year, None], index=['Year', f'Average rating of top {n} films'])
            df_favourite_year = df_favourite_year._append(null_row, ignore_index=True)

    # Plotting
    plt.bar(df_favourite_year['Year'], df_favourite_year[f'Average rating of top {n} films'])
    plt.xlabel('Year')
    plt.ylabel('Average Rating')
    plt.title(f'Average Rating of Top {n} Films Over Years')
    plt.savefig(f'average_rating_by_year_top_{n}_films.pdf', format='pdf') 
    plt.show()


    sorted_df = df_favourite_year.sort_values(by=f'Average rating of top {n} films', ascending=False)

    print(f"Highest average year of top {n} films:")
    print(f"Year: {int(round(sorted_df.iloc[0]['Year']))} with average rating of: {sorted_df.iloc[0][f'Average rating of top {n} films']}")
    print(f"Year: {int(round(sorted_df.iloc[1]['Year']))} with average rating of: {sorted_df.iloc[1][f'Average rating of top {n} films']}")
    print(f"Year: {int(round(sorted_df.iloc[2]['Year']))} with average rating of: {sorted_df.iloc[2][f'Average rating of top {n} films']}")
    print("")
    
    df_favourite_year.to_csv(f'favourite_year_by_{n}_movies.csv', index=False)
    
    return df_favourite_year


###############################################################################

##################################################################
# Total length
##################################################################

def print_total_length(df):
    """
    Print the total length of all films in the DataFrame.
    """
    total_length = df["Length (mins)"].sum()
    print("Film length data: ")
    print("Total length of all films:", round(total_length/60,1), "hours")
    print("Total length of all films:", round(total_length/(60*24)), "days")
    print("")

##################################################################

##################################################################
# Biggest differences
##################################################################

def biggest_difference(df, n, higher_or_lower):
    """
    Find the top n movies with the biggest difference between
    their actual rating and their average rating.
    """
    # Get the average ratings and actual ratings
    average_rating = df['Average Rating']
    actual_rating = df['Rating']

    # Calculate the differences between average rating and actual rating
    difference = average_rating - actual_rating
    
    if higher_or_lower == 1:
        sorted_list = sorted(difference)[:n]
    elif higher_or_lower == -1:
        sorted_list = sorted(difference)[-n:]
    else:
        print("Enter a valid higher or lower value")
        return None

    # Initialize an empty DataFrame to store the filtered rows
    higher_or_lower_df = pd.DataFrame(columns=["Name", "Rating", "Average Rating", "Difference"])

    # Create a dictionary to store indices for each unique difference value
    indices_dict = {}

    # Populate the dictionary with indices corresponding to each difference value
    for i in difference.unique():
        indices_dict[i] = difference[difference == i].index.tolist()
        
    
    rows_added = 1
    
    # Loop through each difference value in the sorted list
    for diff in sorted_list:
        if rows_added > 1:
            rows_added -= 1
            continue
        rows_added -= 1
        # Retrieve the indices corresponding to the current difference value
        indices = indices_dict[diff]

        # Append all rows with the current difference value to the result DataFrame
        for idx in indices:
            row = df.loc[idx, ["Name", "Rating", "Average Rating"]]
            row["Difference"] = diff
            higher_or_lower_df = higher_or_lower_df._append(row, ignore_index=True)
            rows_added += 1
            if rows_added > n:
                break
            
        if higher_or_lower == -1:
            higher_or_lower_df = higher_or_lower_df.sort_values(by="Difference", ascending=False)
    
    if higher_or_lower == 1:
        print(f"Top {n} higher than average ratings: ")
        higher_or_lower_df.to_csv(f"top_{n}_differences.csv", index=False)
    elif higher_or_lower == -1:
        print(f"Top {n} lower than average ratings: ")
        higher_or_lower_df.to_csv(f"bottom_{n}_differences.csv", index=False)
    
    print(higher_or_lower_df[['Name','Rating','Average Rating']])

    return higher_or_lower_df
##################################################################

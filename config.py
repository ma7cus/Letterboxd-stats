##################################################################
# File paths
################################################################## 
DOWNLOADED_RATINGS_FILE = "ratings.csv"  
# Path to the raw ratings data file downloaded directly from Letterboxd. 
# This file contains basic movie data in columns "Date, Name, Year, Letterboxd URI, Rating".

OLD_RUN_OUTPUT_FILE = "ratings_24_01_2025.csv"  
# Path to the output file from a previous run if it exists (you need not specify this file if this is the first run of the code).    
# This file contains processed columns like "Average Rating" and "Length (mins)" and is used to merge with new data, saving recalculations.

CURRENT_RUN_OUTPUT_FILE = "ratings_11_02_2025.csv"  
# Path to the output file for the current run. 
# If this file already exists, it will be loaded without further updates; otherwise, it will be created after processing.
##################################################################

##################################################################
# Favourite year by film/film by year variables
##################################################################

FAVOURITES_TOP_N_MOVIES = 8  
# Specifies the number of top-rated movies to include in the output for each year's "favourite movies" DataFrame.
# Example: If set to 5, the script will output the user's top 5 movies for each year.

FAVOURITES_MIN_RATING = 3  
# Minimum rating threshold for a movie to be added to the dataframe when analysing favourite movies.
# Example: If set to 3, the script will only consider movies with a rating of 3 or higher when adding to the dataframe.

YEARLY_TOP_N_MOVIES = 5  
# Specifies the number of top-rated movies to consider when calculating the average rating for each year.
# Example: If set to 5, the script will calculate the average yearly rating based on the user's top 5 movies for that year.
##################################################################

##################################################################
# Biggest difference variables
##################################################################
BIGGEST_DIFFERENCES_TOP_N = 25  
# The number of movies to include when analysing the biggest differences between their average rating 
# (as calculated from metadata) and their actual user-provided rating.
##################################################################
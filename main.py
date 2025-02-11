##################################################################
# Inputs
##################################################################
from main_functions import (
    import_and_process_data,
    ratings_dist,
    release_year_dist,
    print_nth_favourite_movies,
    print_favourite_year,
    print_total_length,
    biggest_difference,
)
from config import (  # Import variables from the config file
    DOWNLOADED_RATINGS_FILE,
    OLD_RUN_OUTPUT_FILE,
    CURRENT_RUN_OUTPUT_FILE,
    FAVOURITES_TOP_N_MOVIES,
    FAVOURITES_MIN_RATING,
    YEARLY_TOP_N_MOVIES,
    BIGGEST_DIFFERENCES_TOP_N,
)
##################################################################

##################################################################
# Main script
##################################################################
def main():
    # Process data
    df = import_and_process_data(
        DOWNLOADED_RATINGS_FILE,
        CURRENT_RUN_OUTPUT_FILE,
        OLD_RUN_OUTPUT_FILE,
    )

    # Generate and display distribution plots
    ratings_dist(df)
    release_year_dist(df)

    # Generate data on favourite movies in each year and favourite years by the top 'n' movie ratings
    df_year_favourites = print_nth_favourite_movies(df, FAVOURITES_TOP_N_MOVIES, FAVOURITES_MIN_RATING)
    df_favourite_year = print_favourite_year(df, YEARLY_TOP_N_MOVIES)

    # Calculate total length of all films
    print_total_length(df)

    # Find biggest differences between average rating and user rating
    higher_df = biggest_difference(df, BIGGEST_DIFFERENCES_TOP_N, 1)
    lower_df = biggest_difference(df, BIGGEST_DIFFERENCES_TOP_N, -1)
##################################################################

##################################################################
# Main
##################################################################
if __name__ == "__main__":
    main()

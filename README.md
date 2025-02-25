Letterboxd Analysis Script
- This script lets you analyse your letterboxd data by a few specified statistics. It combines your ratings data with extra details like average ratings and movie lengths, then creates charts and spreadsheets for analysis.

How It Works
- You provide your downloaded ratings.csv file from Letterboxd (go to settings, data and click the export button)
- The script fetches details like average ratings and runtimes for movies that are missing these fields. (If you have a previous run, it will merge this data to avoid recalculating fields)
- It will then generate spreadsheets and charts showing the data clearly.

What you'll need
- Some python libraries which can be installed using: 
pip install pandas numpy matplotlib seaborn scipy tqdm beautifulsoup4 requests
- Your exported ratings.csv file from Letterboxd.
- Variables like file paths and statistical parameters defined correctly in the config file

Running the Script
- Clone the repository and navigate to the folder: https://github.com/ma7cus/Letterboxd-stats
- git clone 
- cd Letterboxd-stats
- python main.py

What You’ll Get
Charts:
- ratings_dist.pdf: Shows the distribution of your ratings.
- year_dist.pdf: Shows how many movies you watched each year.
- average_rating_by_year_top_N_films.pdf: Highlights the average ratings of your top 'n' favourite movies by year.

Spreadsheets:
- favourite_N_movies_by_year.csv: Lists your favourite 'n' movies for each year.
- favourite_year_by_N_movies.csv: Ranks years by the average rating of your 'n' favourite movies.
- top_N_differences.csv: 'n' movies you rated higher than their average rating.
- bottom_N_differences.csv: 'n' movies you rated lower than their average rating.

Notes
- Make sure your ratings.csv file is in the same directory or update the path in config.py.
- If you have a previous run’s output, use OLD_RUN_OUTPUT_FILE to avoid reprocessing.
- All outputs are saved in the same directory as the script.
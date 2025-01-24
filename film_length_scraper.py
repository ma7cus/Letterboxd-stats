import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

###############################################################################
# Function to extract film length from URL
###############################################################################
def extract_film_length_from_url(url):
    """
    Function to extract the film length (in minutes) from a given URL.

    Parameters:
    url (str): The URL of the webpage containing the film length information.

    Returns:
    str: The film length extracted from the URL, or None if extraction fails.
    """
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        html_content = response.text
    else:
        print(f"Failed to retrieve HTML content from {url}. Status code: {response.status_code}")
        return None

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all <p> tags with class 'text-link text-footer'
    p_tags_with_class = soup.find_all('p', class_='text-link text-footer')

    film_length = None
    # Iterate through the <p> tags to find film length information
    for p_tag in p_tags_with_class:
        p_text = p_tag.get_text(strip=True)
        if 'mins' in p_text:
            film_length = p_text.split()[0]  # Extract the film length
            break

    return film_length

###############################################################################
# Function to extract film lengths for blank entries in 'Length (mins)' column
###############################################################################
def extract_film_lengths(df):
    """
    Function to extract film lengths for blank entries in the 'Length (mins)' column of a DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the film data.

    Returns:
    list: A list containing the extracted film lengths.
    """
    film_lengths = []
    null_count = df['Length (mins)'].isnull().sum()
    # Iterate through the DataFrame rows
    print("Extracting new film lengths: ")
    with tqdm(total=null_count) as pbar:
        for url, length in zip(df["Letterboxd URI"], df["Length (mins)"]):
            if pd.isnull(length):  # Check if the film length is blank
                film_length = extract_film_length_from_url(url)
                film_lengths.append(film_length)
                pbar.update(1)
            else:
                film_lengths.append(length)
        print('New film lengths extracted')
        print('')
    return film_lengths

###############################################################################
# Code for testing the functions
###############################################################################
"""
file_path = 'test_data.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Call the extract_film_lengths function to extract film lengths
film_lengths = extract_film_lengths(df)

# Append the extracted film lengths to the DataFrame
df['Length (mins)'] = film_lengths
df['Length (mins)'] = pd.to_numeric(df['Length (mins)'], errors='coerce')

# Print the last few rows of the DataFrame to verify the appended lengths
print(df[['Average Rating', 'Length (mins)']].tail())
"""
################################################################
# Inputs
################################################################
import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

################################################################
# Function to read metadata from a given URL
################################################################
def read_metadata_from_url(url):
    """
    Reads metadata from a given URL.

    Args:
        url (str): The URL from which to read metadata.

    Returns:
        dict: A dictionary containing the metadata found in the HTML.
    """
    # Send a GET request to the URL to fetch the HTML content
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        html_content = response.text
    else:
        print(f"Failed to retrieve HTML content from {url}. Status code: {response.status_code}")
        return None
    
    # Parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all meta tags in the HTML
    meta_tags = soup.find_all('meta')

    # Create a dictionary to hold the metadata
    metadata = {}

    # Extract metadata from meta tags
    for tag in meta_tags:
        if 'name' in tag.attrs:
            name = tag.attrs['name']
            content = tag.attrs.get('content', '')
            metadata[name] = content
        elif 'property' in tag.attrs:  # For OpenGraph metadata
            property = tag.attrs['property']
            content = tag.attrs.get('content', '')
            metadata[property] = content

    return metadata

################################################################
# Function to extract average ratings from metadata for blank entries in the DataFrame
################################################################
def extract_average_ratings(df):
    """
    Extracts average ratings from metadata for blank entries in the DataFrame.

    Args:
        df (DataFrame): The DataFrame containing movie data.

    Returns:
        list: A list of extracted average ratings.
    """
    urls = df['Letterboxd URI']
    average_ratings_old = df['Average Rating']
    average_ratings = []
    null_count = df['Average Rating'].isnull().sum()
    # Iterate through the DataFrame rows
    print("Extracting new average ratings:")
    with tqdm(total=null_count) as pbar:
        for url, average_rating in zip(urls, average_ratings_old):
            if pd.isnull(average_rating):  # Check if the average rating is blank
                metadata = read_metadata_from_url(url)
                average_rating = metadata.get('twitter:data2')  # Gets average rating in the form a.b out of c
                if average_rating:
                    average_rating = float(re.search(r'\d+\.\d+', average_rating).group())  # Extracts the first number in the string of form a.b
                else:
                    average_rating = None  # Set average_rating to None if not found
                pbar.update(1)
            average_ratings.append(average_rating)
                
        print('New average ratings extracted')
        print('')
    return average_ratings


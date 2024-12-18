from thefuzz import process
import pandas as pd
import os
import pickle
from openai import OpenAI
import csv
from thefuzz import process
import numpy as np

def extract_user_shows(user_input):
    """
    Extracts and normalizes TV show names from user input.

    Parameters:
        user_input (str): Comma-separated TV show names entered by the user.

    Returns:
        list: A cleaned list of TV show names.
    """
    # Normalize and split the input by commas
    user_shows = [show.strip() for show in user_input.split(",") if show.strip()]
    return user_shows

def load_tv_shows_pandas(csv_file):
    """
    Loads TV shows and their descriptions from a CSV file using Pandas.

    Parameters:
        csv_file (str): Path to the CSV file.

    Returns:
        dict: A dictionary where keys are show titles and values are descriptions.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Convert the DataFrame into a dictionary (title: description)
    return pd.Series(df["Description"].values, index=df["Title"].values).to_dict()

def match_user_shows(user_shows, tv_show_list):
    """
    Matches user-entered shows to the closest titles in the TV show list.

    Parameters:
        user_shows (list): List of user-input show names.
        tv_show_list (list): List of actual TV show titles.

    Returns:
        list: A list of matched show titles.
    """
    matched_shows = []
    for show in user_shows:
        # Find the closest matching title in the list
        match, score = process.extractOne(show, tv_show_list)
        matched_shows.append(match)
    return matched_shows


def generate_embeddings(tv_show_data, embeddings_file):
    """
    Generates and saves embeddings for TV show descriptions using OpenAI API.

    Parameters:
        tv_show_data (dict): Dictionary with show titles as keys and descriptions as values.
        embeddings_file (str): Path to save the embeddings using pickle.

    Returns:
        dict: Dictionary with show titles as keys and their embedding vectors as values.
    """
    if os.path.exists(embeddings_file):
        print("Loading existing embeddings...")
        with open(embeddings_file, "rb") as f:
            return pickle.load(f)

    print("Generating embeddings...")
    client = OpenAI()
    embeddings = {}
    for title, description in tv_show_data.items():
        response = client.embeddings.create(input=description, model="text-embedding-ada-002")
        embeddings[title] = response.data[0].embedding

    with open(embeddings_file, "wb") as f:
        pickle.dump(embeddings, f)
    print("Embeddings saved successfully!")
    return embeddings

def calculate_average_vector(show_vectors):
    """
    Calculates the average vector from a list of vectors.

    Parameters:
        show_vectors (list): List of embedding vectors.

    Returns:
        np.array: The average vector.
    """
    if not show_vectors:
        raise ValueError("No vectors provided for averaging.")
    return np.mean(show_vectors, axis=0)

import numpy as np

def cosine_similarity(a, b):
    """
    Returns the cosine similarity between two vectors `a` and `b`.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def recommend_shows(input_shows, average_vector, tv_show_embeddings, top_n=5):
    """
    Recommends TV shows based on similarity to the average vector.

    Parameters:
        input_shows (list): List of shows input by the user.
        average_vector (np.array): The average vector of the user's input shows.
        tv_show_embeddings (dict): Dictionary of TV show embeddings.
        top_n (int): Number of top recommendations to return.

    Returns:
        list: A list of tuples (show_title, similarity, percentage).
    """
    similarities = []

    for show, vector in tv_show_embeddings.items():
        # Exclude the input shows
        if show in input_shows:
            continue

        # Calculate cosine similarity
        similarity = cosine_similarity(average_vector, np.array(vector))
        similarities.append((show, similarity))

    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Select top N recommendations
    top_recommendations = similarities[:top_n]

    # Normalize similarity scores to percentages
    max_similarity = top_recommendations[0][1]
    min_similarity = top_recommendations[-1][1]

    recommendations_with_percentages = []
    for show, similarity in top_recommendations:
        # Scale similarity to a percentage (normalized)
        percentage = 100 * (similarity - min_similarity) / (max_similarity - min_similarity)
        recommendations_with_percentages.append((show, similarity, round(percentage, 2)))

    return recommendations_with_percentages


if __name__ == "__main__":
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
    csv_file_path = os.path.join(script_dir, "imdb_tvshows.csv")
    embeddings_file_path = os.path.join(script_dir, "tv_show_embeddings.pkl")

    # Step 1: Load TV shows and embeddings
    try:
        tv_show_data = load_tv_shows_pandas(csv_file_path)
        print(f"Loaded {len(tv_show_data)} TV shows successfully!\n")
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found. Make sure it exists in the same directory as this script.")
        exit(1)

    try:
        tv_show_embeddings = generate_embeddings(tv_show_data, embeddings_file_path)
        print(f"Loaded embeddings for {len(tv_show_embeddings)} TV shows successfully!\n")
    except FileNotFoundError:
        print(f"Error: The embeddings file '{embeddings_file_path}' was not found. Please generate embeddings first.")
        exit(1)


    # List of TV show titles
    tv_show_list = list(tv_show_data.keys())

    # Step 2: Ask the user for input
    while True:
        # Ask the user for input
        user_input = input(
            "Which TV shows did you really like watching? Separate them by a comma. Make sure to enter more than 1 show\n"
        )

        # Extract shows from user input
        extracted_shows = extract_user_shows(user_input)

        if len(extracted_shows) > 1:
            # Match user-input shows to real titles
            matched_shows = match_user_shows(extracted_shows, tv_show_list)
            # Confirm with the user
            confirmation = input(
                f"Making sure, do you mean {', '.join(extracted_shows)}? (y/n): "
            ).strip().lower()

            if confirmation == 'y':
                print(f"Great! Generating recommendations now...")
                break
            else:
                print("Sorry about that. Let's try again. Please make sure to write the names of the TV shows correctly.")
        else:
            print("Please enter at least two valid TV shows.")
    
    # Step 4: Fetch vectors for matched shows
    input_vectors = [
        np.array(tv_show_embeddings[show]) for show in matched_shows if show in tv_show_embeddings
    ]

    if not input_vectors:
        print("Error: No valid embeddings found for the matched shows.")
        exit(1)

    # Step 5: Calculate the average vector
    average_vector = calculate_average_vector(input_vectors)

    # Step 6: Get recommendations
    recommendations = recommend_shows(matched_shows, average_vector, tv_show_embeddings)

    # Step 7: Output recommendations
    print("\nHere are the TV shows that I think you would love:")
    for show, similarity, percentage in recommendations:
        print(f"{show} ({percentage}%)")
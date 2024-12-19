from thefuzz import process
import pandas as pd
import os
import pickle
from openai import OpenAI
import csv
from thefuzz import process
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import time
import json

# Initialize OpenAI Client
client = OpenAI()

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


def generate_show_name_and_description(prompt):
    """
    Generate a TV show name and description using OpenAI's GPT model.

    Parameters:
        prompt (str): A descriptive prompt for the show.

    Returns:
        tuple: Show name and description.
    """
    # Send the request to OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7
    )
    generated_text = response.choices[0].message.content.strip()

    # Initialize defaults
    show_name = "Unnamed Show"
    description = "No description provided."

    try:
        # Split the response into lines for processing
        lines = generated_text.split("\n")

        # Iterate through lines to extract show name and description
        for line in lines:
            if line.lower().startswith("show name:"):
                # Extract the text after "Show Name:"
                show_name = line.split(":", 1)[1].strip()
            elif line.lower().startswith("description:"):
                # Extract the text after "Description:"
                description = line.split(":", 1)[1].strip()

        # Fallback: If no explicit "Show Name" or "Description" found, use entire text
        if not show_name or show_name == "Unnamed Show":
            show_name = generated_text.split("\n")[0].strip()
        if not description or description == "No description provided.":
            description = "\n".join(generated_text.split("\n")[1:]).strip()

    except Exception as e:
        print(f"Error parsing generated text: {e}. Returning defaults.")
    
    return show_name, description



def generate_show_image(prompt, output_path):
    LIGHTX_API_URL = 'https://api.lightxeditor.com/external/api/v1/text2image'
    LIGHTX_STATUS_URL = 'https://api.lightxeditor.com/external/api/v1/order-status'
    lightx_api_key = os.getenv("LIGHTX_API_KEY")

    if not lightx_api_key:
        print("Error: LIGHTX_API_KEY environment variable not set.")
        return None

    headers = {
        'Content-Type': 'application/json',
        'x-api-key': lightx_api_key
    }

    # Step 1: Submit the image generation request
    data = {
        "textPrompt": prompt
    }

    try:
        print(f"Sending request to LightX API with prompt: {prompt}")
        response = requests.post(LIGHTX_API_URL, headers=headers, json=data)

        if response.status_code == 200:
            try:
                response_data = response.json()
                print("Debug: Response JSON:", response_data)  # Debugging line to inspect the JSON
                if response_data and 'body' in response_data and 'orderId' in response_data['body']:
                    order_id = response_data.get('body', {}).get('orderId')
                    print(f"Order ID: {order_id}")
                else:
                    print("Error: 'body' or 'orderId' not found in the response.")
                    print(f"Response Content: {response_data}")
                    return None
            except ValueError as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Response Content: {response.text}")
                return None

            # Step 2: Poll for the image URL
            status_payload = {"orderId": order_id}
            retries = 0
            max_retries = 5
            image_url = None

            while retries < max_retries:
                time.sleep(5)  # Wait for 5 seconds between polls
                status_response = requests.post(LIGHTX_STATUS_URL, headers=headers, json=status_payload)

                if status_response.status_code == 200:
                    try:
                        status_data = status_response.json().get('body', {})
                        status = status_data.get('status')

                        if status == "active":
                            image_url = status_data.get('output')  # The image URL
                            print(f"Image URL: {image_url}")
                            break
                        elif status == "failed":
                            print("Image generation failed.")
                            return None
                    except ValueError as e:
                        print(f"Error parsing JSON response during status check: {e}")
                        print(f"Response Content: {status_response.text}")
                        break
                else:
                    print(f"Failed to fetch status. Status code: {status_response.status_code}")
                    print(status_response.text)

                retries += 1

            if image_url:
                # Step 3: Download the image
                os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directory exists
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    with open(output_path, "wb") as file:
                        file.write(image_response.content)
                    print(f"Image saved to: {output_path}")
                    return output_path
                else:
                    print(f"Failed to download the image. Status code: {image_response.status_code}")
            else:
                print("Image generation timed out.")
        else:
            print(f"Initial request failed. Status code: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as err:
        print(f"Request error: {err}")

    return None

def display_images(image_paths):
    """
    Display the generated images inside the Python program.

    Parameters:
        image_paths (list): List of image file paths to display.
    """
    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            img.show()  # Opens the image using the default viewer
        except Exception as e:
            print(f"Error displaying image {image_path}: {e}")

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

    # Step 7: Generate custom shows sequentially
    # First custom show based on user input
    input_prompt = (
        "Create a unique TV show name and short description.\n"
        "Please respond in the following format:\n"
        "Show Name: [Your TV Show Name]\n"
        "Description: [A brief description of the show]\n\n"
        "Example:\n"
        "Show Name: Chrono Shadows\n"
        "Description: In a world where time is a malleable force, a rogue timekeeper and a young girl must race against time to save reality.\n\n"
        "The following shows are provided as inspiration: "
        f"{', '.join(matched_shows)}"
    )
    show1_name, show1_description = generate_show_name_and_description(input_prompt)
    print(f"\nShow #1: {show1_name}")
    print(f"Description: {show1_description}")

    show1_visual_prompt = f"An image depicting the TV show '{show1_name}' and its story: {show1_description}"
    show1_image_path = f"generated_ads/{show1_name.replace(' ', '_').replace('*', '')}.png"
    generate_show_image(show1_visual_prompt, show1_image_path)

    # Second custom show based on recommendations
    recommended_prompt = (
        "Create a unique TV show name and short description.\n"
        "Please respond in the following format:\n"
        "Show Name: [Your TV Show Name]\n"
        "Description: [A brief description of the show]\n\n"
        "Example:\n"
        "Show Name: Arcane Legends\n"
        "Description: In a world where ancient magic and modern technology collide, a group of misfits uncovers a secret that could change their world forever.\n\n"
        "The following recommended shows are provided as inspiration: "
        f"{', '.join([rec[0] for rec in recommendations])}"
    )
    
    show2_name, show2_description = generate_show_name_and_description(recommended_prompt)
    print(f"\nShow #2: {show2_name}")
    print(f"Description: {show2_description}")

    show2_visual_prompt = f"An image depicting the TV show '{show2_name}' and its story: {show2_description}"
    show2_image_path = f"generated_ads/{show2_name.replace(' ', '_').replace('*', '')}.png"
    generate_show_image(show2_visual_prompt, show2_image_path)

    # Step 8: Display generated images
    print("\nHere are your two custom TV shows with their posters:")
    print(f"Show #1: {show1_name} | Poster saved at: {show1_image_path}")
    print(f"Show #2: {show2_name} | Poster saved at: {show2_image_path}")
    display_images([show1_image_path, show2_image_path])


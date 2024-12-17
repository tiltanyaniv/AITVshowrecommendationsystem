from thefuzz import process
import pandas as pd
import os

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


def generate_embeddings(tv_show_data, embeddings_file):
    """
    Placeholder implementation for TDD. This function does nothing meaningful yet.
    """
    return {}


if __name__ == "__main__":
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
    csv_file_path = os.path.join(script_dir, "imdb_tvshows")
    # Step 1: Load TV shows using Pandas
    try:
        tv_show_data = load_tv_shows_pandas(csv_file_path)
        print(f"Loaded {len(tv_show_data)} TV shows successfully!\n")
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found. Make sure it exists in the same directory as this script.")
        exit(1)
    while True:
        # Ask the user for input
        user_input = input(
            "Which TV shows did you really like watching? Separate them by a comma. Make sure to enter more than 1 show\n"
        )

        # Extract shows from user input
        extracted_shows = extract_user_shows(user_input)

        if len(extracted_shows) > 1:
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
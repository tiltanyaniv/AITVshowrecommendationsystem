from thefuzz import process

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

if __name__ == "__main__":
    while True:
        # Ask the user for input
        user_input = input(
            "Which TV shows did you really like watching? Separate them by a comma. Make sure to enter more than 1 show:\n"
        )

        # Extract shows from user input
        extracted_shows = extract_user_shows(user_input)

        if len(extracted_shows) > 1:
            print(f"Extracted shows: {extracted_shows}")
            break
        else:
            print("Please enter at least two valid TV shows.")
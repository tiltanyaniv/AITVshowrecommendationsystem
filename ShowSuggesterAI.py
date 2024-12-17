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

def load_tv_shows_pandas(csv_file):
    """
    Placeholder implementation: Make the test fail initially.
    """
    return {}

if __name__ == "__main__":
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
from ShowSuggesterAI import extract_user_shows
import pytest

@pytest.mark.parametrize("user_input, expected_output", [
    ("Game Of Thrones, Lupin, The Witcher", ["Game Of Thrones", "Lupin", "The Witcher"]),
    ("game of thrones, lupin, witcher", ["game of thrones", "lupin", "witcher"]),  # Case sensitivity test
    ("Game Of Thrones, , Witcher", ["Game Of Thrones", "Witcher"]),              # Handles empty input
    ("Breaking Bad, Sherlock, Dark", ["Breaking Bad", "Sherlock", "Dark"]),      # Completely new input
])
def test_validate_user_input(user_input, expected_output):
    assert extract_user_shows(user_input) == expected_output
from ShowSuggesterAI import extract_user_shows, load_tv_shows_pandas
import pytest
import pandas as pd
import pickle
from unittest.mock import MagicMock, patch



@pytest.mark.parametrize("user_input, expected_output", [
    ("Game Of Thrones, Lupin, The Witcher", ["Game Of Thrones", "Lupin", "The Witcher"]),
    ("game of thrones, lupin, witcher", ["game of thrones", "lupin", "witcher"]),  # Case sensitivity test
    ("Game Of Thrones, , Witcher", ["Game Of Thrones", "Witcher"]),              # Handles empty input
    ("Breaking Bad, Sherlock, Dark", ["Breaking Bad", "Sherlock", "Dark"]),      # Completely new input
])

def test_validate_user_input(user_input, expected_output):
    assert extract_user_shows(user_input) == expected_output

def test_load_tv_shows(tmpdir):
    # Create a mock CSV file
    csv_content = "Title,Description\nGame Of Thrones,A fantasy series with dragons.\nBreaking Bad,A chemistry teacher turns drug lord."
    csv_file = tmpdir.join("tv_shows.csv")
    csv_file.write(csv_content)
    
    # Load TV shows
    tv_shows = load_tv_shows_pandas(str(csv_file))
    
    # Assert the output
    assert len(tv_shows) == 2
    assert tv_shows["Game Of Thrones"] == "A fantasy series with dragons."
    assert tv_shows["Breaking Bad"] == "A chemistry teacher turns drug lord."

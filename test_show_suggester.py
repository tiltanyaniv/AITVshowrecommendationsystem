from ShowSuggesterAI import extract_user_shows, load_tv_shows_pandas, generate_embeddings
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


# Test 1: Validate user input
def test_validate_user_input(user_input, expected_output):
    assert extract_user_shows(user_input) == expected_output


# Test 2: Load TV shows from CSV
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

# Test 3: Generate and save embeddings
@patch("openai.OpenAI")
def test_generate_embeddings(mock_openai, tmpdir):
    # Mock the OpenAI client and embedding response
    mock_client = MagicMock()
    mock_embedding = [0.0] * 1536  # Mock embedding of size 1536
    mock_response = MagicMock()
    mock_response.data = [{"embedding": mock_embedding}]  # Simulate real API response structure
    
    # Set the mock response for embeddings.create
    mock_client.embeddings.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    # Prepare test data
    tv_show_data = {"Game Of Thrones": "A fantasy series with dragons."}
    embeddings_file = tmpdir.join("embeddings.pkl")

    # Generate embeddings
    embeddings = generate_embeddings(tv_show_data, str(embeddings_file))
    
    # Assert the embeddings are correct
    assert "Game Of Thrones" in embeddings
    assert len(embeddings["Game Of Thrones"]) == 1536  # Check embedding size
    assert embeddings["Game Of Thrones"] == mock_embedding  # Check exact mock content
    
    # Check that embeddings were saved to the file
    with open(str(embeddings_file), "rb") as f:
        loaded_embeddings = pickle.load(f)
    assert loaded_embeddings == embeddings

# Test 4: Load saved embeddings
def test_load_existing_embeddings(tmpdir):
    # Create a mock embeddings file
    embeddings_file = tmpdir.join("embeddings.pkl")
    mock_embeddings = {"Game Of Thrones": [0.1, 0.2, 0.3]}
    with open(str(embeddings_file), "wb") as f:
        pickle.dump(mock_embeddings, f)
    
    # Load embeddings
    embeddings = generate_embeddings({}, str(embeddings_file))
    
    # Assert the embeddings match
    assert embeddings == mock_embeddings
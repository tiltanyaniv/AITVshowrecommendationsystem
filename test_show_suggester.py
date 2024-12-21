from ShowSuggesterAI import extract_user_shows, load_tv_shows_pandas, cosine_similarity, generate_embeddings, calculate_average_vector
import pytest
import pandas as pd
import pickle
from unittest.mock import MagicMock, patch
import numpy as np



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


# Test 3: Load saved embeddings
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

@pytest.mark.parametrize("vector_a, vector_b, expected_similarity", [
    (np.array([1, 0, 0]), np.array([1, 0, 0]), 1.0),  # Perfect similarity
    (np.array([1, 0, 0]), np.array([0, 1, 0]), 0.0),  # Orthogonal vectors
    (np.array([1, 2, 3]), np.array([1, 2, 3]), 1.0),  # Same vectors
    (np.array([1, 0, 0]), np.array([-1, 0, 0]), -1.0),  # Opposite direction
    (np.array([1, 2, 3]), np.array([3, 2, 1]), pytest.approx(0.714, rel=1e-3)),  # Non-orthogonal vectors
])

# Test 4: check cosine similarity  
def test_cosine_similarity(vector_a, vector_b, expected_similarity):
    similarity = cosine_similarity(vector_a, vector_b)
    assert similarity == pytest.approx(expected_similarity, rel=1e-3)  # Allow for slight floating-point differences

@pytest.mark.parametrize("vectors, expected_average", [
    ([np.array([1, 2, 3]), np.array([4, 5, 6])], np.array([2.5, 3.5, 4.5])),  # Simple average
    ([np.array([0, 0, 0]), np.array([0, 0, 0])], np.array([0, 0, 0])),        # Zero vectors
    ([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])], np.array([1/3, 1/3, 1/3])),  # Orthogonal vectors
    ([np.array([10, 20]), np.array([30, 40]), np.array([50, 60])], np.array([30, 40])),  # Larger numbers
])
# Test 5: check average ector
def test_calculate_average_vector(vectors, expected_average):
    average_vector = calculate_average_vector(vectors)
    assert np.allclose(average_vector, expected_average, atol=1e-6), f"Expected {expected_average}, got {average_vector}"

def test_calculate_average_vector_empty():
    with pytest.raises(ValueError, match="No vectors provided for averaging."):
        calculate_average_vector([])  # Empty list should raise ValueError
from ShowSuggesterAI import validate_user_input
import pytest 

def test_validate_user_input():
    user_input = "gem of throns, lupan, witcher"
    expected_output = ["Game Of Thrones", "Lupin", "The Witcher"]
    assert validate_user_input(user_input) == expected_output
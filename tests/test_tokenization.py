from src.tokenization import Tokenizer


def test_tokenizer_generates_typed_token():

    tokenizer = Tokenizer()

    token = tokenizer.tokenize("SSN")

    assert token == "[SSN_1]"


def test_tokenizer_increments_counter():

    tokenizer = Tokenizer()

    token_1 = tokenizer.tokenize("SSN")
    token_2 = tokenizer.tokenize("SSN")

    assert token_1 == "[SSN_1]"
    assert token_2 == "[SSN_2]"


def test_tokenizer_uses_separate_counters():

    tokenizer = Tokenizer()

    ssn = tokenizer.tokenize("SSN")
    email = tokenizer.tokenize("EMAIL")

    assert ssn == "[SSN_1]"
    assert email == "[EMAIL_1]"


def test_tokenizer_reset():

    tokenizer = Tokenizer()

    tokenizer.tokenize("SSN")
    tokenizer.reset()

    token = tokenizer.tokenize("SSN")

    assert token == "[SSN_1]"
from memonative.auth.keys import KEY_PREFIX, generate_key, hash_key


def test_generated_key_has_prefix_and_minimum_entropy():
    k = generate_key()
    assert k.startswith(KEY_PREFIX)
    # urlsafe_b64(32 bytes) is 43 chars; total >= 47.
    assert len(k) >= len(KEY_PREFIX) + 43


def test_generated_keys_are_unique():
    keys = {generate_key() for _ in range(100)}
    assert len(keys) == 100


def test_hash_is_64_hex_chars_and_stable():
    h1 = hash_key("mn_example")
    h2 = hash_key("mn_example")
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_hash_changes_with_input():
    assert hash_key("a") != hash_key("b")

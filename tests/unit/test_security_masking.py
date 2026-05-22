"""Unit tests for security masking helpers (T057)."""

from rulekiln.observability.security import mask_dict_values, mask_url


def test_mask_url_credentials() -> None:
    url = "postgresql://user:supersecret@localhost/db"
    masked = mask_url(url)
    assert "supersecret" not in masked
    assert "***:***" in masked
    assert "localhost/db" in masked


def test_mask_url_api_key_querystring() -> None:
    url = "https://api.example.com/v1?api_key=sk-secret123"
    masked = mask_url(url)
    assert "sk-secret123" not in masked
    assert "***MASKED***" in masked


def test_mask_url_no_credentials_unchanged() -> None:
    url = "https://example.com/path?foo=bar"
    assert mask_url(url) == url


def test_mask_dict_values_redacts_secrets() -> None:
    data = {"api_key": "sk-123", "token": "tok-abc", "name": "rulekiln"}
    masked = mask_dict_values(data)
    assert masked["api_key"] == "***REDACTED***"
    assert masked["token"] == "***REDACTED***"  # noqa: S105
    assert masked["name"] == "rulekiln"


def test_mask_dict_values_case_insensitive() -> None:
    data = {"API_KEY": "val", "Password": "pw", "normal": "ok"}
    masked = mask_dict_values(data)
    assert masked["API_KEY"] == "***REDACTED***"
    assert masked["Password"] == "***REDACTED***"  # noqa: S105
    assert masked["normal"] == "ok"

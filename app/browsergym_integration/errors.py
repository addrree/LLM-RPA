class BrowserGymIntegrationError(Exception):
    """Base exception for BrowserGym integration layer."""


class UnsupportedBrowserGymActionError(BrowserGymIntegrationError):
    """Raised when an internal action cannot be represented as BrowserGym action string."""

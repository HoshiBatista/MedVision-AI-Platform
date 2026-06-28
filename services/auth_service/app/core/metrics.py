from prometheus_client import Counter, Histogram

AUTH_LOGIN_TOTAL = Counter(
    "auth_login_total",
    "Total login attempts",
    ["result"],  # success | invalid_credentials | account_disabled
)

AUTH_REGISTER_TOTAL = Counter(
    "auth_register_total",
    "Total registration attempts",
    ["result"],  # success | email_exists
)

AUTH_LOGOUT_TOTAL = Counter(
    "auth_logout_total",
    "Total logout calls",
)

AUTH_REFRESH_TOTAL = Counter(
    "auth_refresh_total",
    "Total refresh-token rotations",
    ["result"],  # success | invalid | reuse_detected | inactive
)

AUTH_PASSWORD_HASH_DURATION = Histogram(
    "auth_password_hash_duration_seconds",
    "Time spent hashing passwords (bcrypt)",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
)

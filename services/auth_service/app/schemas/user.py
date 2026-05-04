from datetime import datetime

from pydantic import BaseModel, EmailStr, field_validator


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str | None = None

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    username: str
    password: str


class UpdateUserRequest(BaseModel):
    full_name: str | None = None


class AdminUpdateUserRequest(BaseModel):
    full_name: str | None = None
    role: str | None = None
    is_active: bool | None = None

    @field_validator("role")
    @classmethod
    def role_valid(cls, v: str | None) -> str | None:
        if v is not None and v not in ("admin", "user", "radiologist"):
            raise ValueError("role must be one of: admin, user, radiologist")
        return v


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str | None
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}

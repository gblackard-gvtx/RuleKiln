# Python Coding Standards for Cursor Agents

## 📚 Core Principles

**Python Version**: 3.12+ (Modern syntax only, no backwards compatibility)
**Required Libraries**: Pydantic 2.0+, SQLAlchemy 2.0+, SQLModel, Structlog

### Essential Rules

1. **Type Safety First**: Always use proper type annotations, never bare types
2. **Smart Data Structures**: Use `dict[str, primitive]` for simple data, Pydantic models for complex data
3. **No Magic Strings**: Use object attributes (`.property`) instead of dictionary access (`["key"]`)
4. **Modern Python**: Use `str | None` not `Optional[str]`, `list[str]` not `List[str]`
5. **No Type Shortcuts**: Create proper types instead of using `# pyright: ignore` or `Any`

---

## 🚨 CRITICAL VIOLATIONS - NEVER DO THESE

### ❌ ABSOLUTELY FORBIDDEN PATTERNS

```python
# ❌ FORBIDDEN - Bare dict without types
def process_data(obj: dict) -> dict:
    pass

# ❌ FORBIDDEN - Lazy type ignoring
def bad_function(data: dict) -> dict:  # pyright: ignore[reportMissingTypeArgument]
    pass

# ❌ FORBIDDEN - Using Any to bypass typing
from typing import Any
def process_json(data: Any) -> Any:
    pass

# ❌ FORBIDDEN - Magic string dictionary access
result = data["confidence_score"]
errors = data.get("validation_errors", [])
```

### ✅ REQUIRED SOLUTIONS

```python
# ✅ ACCEPTABLE - Simple primitive types
user_settings: dict[str, str] = {"theme": "dark", "lang": "en"}
scores: dict[str, int] = {"math": 95, "english": 87}

# ✅ REQUIRED - Complex data structures use Pydantic
class ErrorData(BaseModel):
    message: str | None = None
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

def process_data(data: ErrorData) -> ProcessedResult:
    return ProcessedResult(...)
```

---

## 🎯 Decision Tree: Dict vs Pydantic

```
Is your data...?

Simple key-value with primitives only (str, int, bool)?
├── ✅ Use dict[str, primitive]

Complex with nesting, lists, or validation?
├── 🚨 MUST use Pydantic model

From external API/JSON?
├── 🚨 MUST use Pydantic model

Used across multiple functions?
└── 🚨 MUST use Pydantic model
```

**Primitive types**: `str`, `int`, `float`, `bool` only

---

## ✅ Type Annotation Requirements

### Function Signatures

```python
# ✅ GOOD - All parameters and return types specified
def process_image(image_data: ImageSchema, session: Session) -> ProcessResult:
    """Process an image and return results."""
    pass

# ✅ GOOD - Modern syntax
def handle_data(data: str | int | None) -> bool:
    pass

# ❌ FORBIDDEN - Missing types
def process_data(data):  # ❌ No type hints
    pass

# ❌ FORBIDDEN - Old syntax
from typing import Union, Optional, List, Dict  # ❌ Don't import these
def handle_data(data: Union[str, int]) -> Optional[bool]:  # ❌ Use | instead
```

### Modern Python 3.12+ Syntax

- Use `str | None` not `Optional[str]`
- Use `list[str]` not `List[str]`
- Use `dict[str, int]` not `Dict[str, int]`
- Use `collections.abc.Sequence` not `typing.Sequence`

### Generic Collections

```python
# ✅ GOOD - Use collections.abc for interfaces
from collections.abc import Sequence, Mapping

def process_sequence(items: Sequence[str]) -> list[str]:
    return list(items)

def process_mapping(data: Mapping[str, int]) -> dict[str, str]:
    return {k: str(v) for k, v in data.items()}
```

### Type Variables for Generics

```python
from typing import TypeVar, Generic

T = TypeVar('T')

class Repository(Generic[T]):
    def get(self, id: int) -> T | None:
        pass
    
    def save(self, entity: T) -> T:
        pass
```

### When to Use Type Ignores

```python
# ✅ ACCEPTABLE - Pydantic model configuration
model_config = ConfigDict(  # pyright: ignore[reportUnannotatedClassAttribute]
    from_attributes=True,
    populate_by_name=True,
)

# ✅ ACCEPTABLE - Third-party library issues
import some_untyped_library
result = some_untyped_library.function()  # pyright: ignore[reportUnknownMemberType]

# ❌ FORBIDDEN - Never use generic ignores
def process_data(obj: dict) -> dict:  # type: ignore  # ❌ Too generic
    pass
```

**Note**: This project uses **Pyright** (not mypy). Always specify the rule name: `# pyright: ignore[ruleName]`

---

## 🏗️ Pydantic Model Requirements

### When to Use Pydantic

- ✅ Nested structures (lists of objects, nested dicts)
- ✅ Complex validation requirements
- ✅ Multiple related fields
- ✅ Business logic or computed properties
- ✅ External API/JSON responses
- ✅ Data used across multiple functions

### Pydantic Best Practices

```python
class ApiResponse(BaseModel):
    """Clear docstring describing the model."""
    
    # Required fields first
    status: str
    message: str
    
    # Optional fields with defaults
    data: dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Validation
    @field_validator('status')
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in {'success', 'error'}:
            raise ValueError('Status must be success or error')
        return v
```

### ❌ Dataclasses are FORBIDDEN

```python
# ❌ NEVER use dataclasses
from dataclasses import dataclass

@dataclass  # ❌ FORBIDDEN
class UserInfo:
    name: str
    age: int

# ✅ ALWAYS use Pydantic
class UserInfo(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)
```

---

## 🛡️ Security Standards

### ✅ Safe Logging

```python
import structlog
import re

logger = structlog.get_logger(__name__)

# ✅ SAFE - Log IDs, not sensitive data
logger.info(
    "User authentication successful",
    user_id=user.id,  # ✅ ID is safe
    timestamp=auth_time,
)

# ✅ SAFE - Mask sensitive data in URLs
def mask_sensitive_url(url: str) -> str:
    """Mask sensitive parts of URL for logging."""
    patterns = [
        (r'([?&]api_key=)[^&]*', r'\1***MASKED***'),
        (r'([?&]token=)[^&]*', r'\1***MASKED***'),
        (r'://[^:]+:[^@]+@', r'://***:***@'),  # username:password
    ]
    masked_url = url
    for pattern, replacement in patterns:
        masked_url = re.sub(pattern, replacement, masked_url)
    return masked_url

logger.info("API request completed", url=mask_sensitive_url(api_url))

# ❌ FORBIDDEN - Never log passwords, tokens, keys
logger.info("User login", password=password)  # ❌ PASSWORD IN LOGS
logger.debug("API response", api_key=api_key)  # ❌ SECRET IN LOGS
logger.error("Database error", connection_string=db_url)  # ❌ May contain credentials

# ❌ DANGEROUS - Exposing internal errors
try:
    sensitive_operation()
except Exception as e:
    return {"error": str(e)}  # ❌ May expose internal details
```

### ✅ Secret Management

```python
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Secure settings management."""
    
    database_url: str = Field(alias="DATABASE_URL")
    api_key: str = Field(alias="API_KEY")
    secret_key: str = Field(alias="SECRET_KEY")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

# ✅ GOOD - Validate secrets exist
settings = Settings()
if not settings.database_url:
    raise ValueError("DATABASE_URL not configured")

# ❌ NEVER hardcode secrets
API_KEY = "sk-1234567890abcdef"  # ❌ HARDCODED SECRET
DATABASE_URL = "postgresql://user:pass@localhost/db"  # ❌ CREDENTIALS IN CODE
```

### ✅ Input Validation with Pydantic

```python
import re
from pydantic import BaseModel, field_validator

class SecureUserInput(BaseModel):
    """Model with built-in data sanitization."""
    
    username: str
    email: str
    comment: str
    
    @field_validator('username')
    @classmethod
    def sanitize_username(cls, v: str) -> str:
        """Remove potentially dangerous characters."""
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', v)
        if not sanitized:
            raise ValueError('Username contains only invalid characters')
        return sanitized.strip()
    
    @field_validator('comment')
    @classmethod
    def sanitize_comment(cls, v: str) -> str:
        """Remove potentially dangerous characters."""
        sanitized = re.sub(r'[<>"\';]', '', v)
        return sanitized.strip()
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email format."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError('Invalid email format')
        return v.lower()
```

### ✅ File Upload Security

```python
import os
from pydantic import BaseModel, field_validator

class FileUpload(BaseModel):
    """Secure file upload validation."""
    
    filename: str
    content: bytes
    content_type: str
    
    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Validate filename is safe."""
        safe_name = os.path.basename(v)  # Remove path traversal attempts
        
        dangerous_exts = {'.exe', '.bat', '.cmd', '.com', '.scr', '.php', '.jsp'}
        _, ext = os.path.splitext(safe_name.lower())
        if ext in dangerous_exts:
            raise ValueError(f'File type {ext} not allowed')
        return safe_name
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: bytes) -> bytes:
        """Validate file content is safe."""
        max_size = 10 * 1024 * 1024  # 10MB
        if len(v) > max_size:
            raise ValueError('File too large')
        
        dangerous_patterns = [b'<script', b'<?php', b'#!/bin']
        for pattern in dangerous_patterns:
            if pattern in v:
                raise ValueError('Potentially malicious file content detected')
        return v
```

### ✅ API Input Validation

```python
from pydantic import BaseModel, Field, field_validator
import re

class APIRequest(BaseModel):
    """Secure API request validation."""
    
    query: str = Field(max_length=1000)
    page: int = Field(ge=1, le=1000)
    limit: int = Field(ge=1, le=100)
    
    @field_validator('query')
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """Sanitize search query."""
        dangerous_patterns = [
            r'union\s+select', r'drop\s+table', r'delete\s+from',
            r'\$where', r'\$regex', r'javascript:'
        ]
        query_lower = v.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, query_lower):
                raise ValueError('Query contains potentially dangerous content')
        return v.strip()
```

### ❌ Security Violations

```python
# ❌ VULNERABLE to SQL injection
query = f"SELECT * FROM users WHERE id = {user_id}"  # ❌ Use SQLModel instead

# ✅ SAFE - Use parameterized queries
query = select(User).where(User.id == user_id)  # ✅ SQLModel safe

# ❌ EXTREMELY DANGEROUS
def process_user_data(raw_input: str):
    exec(raw_input)  # ❌ CODE INJECTION RISK

# ❌ DANGEROUS - Unvalidated input
def evaluate_expression(expr: str):
    return eval(expr)  # ❌ CODE EXECUTION RISK
```

### Security Best Practices

- Always use Pydantic validators for data sanitization
- Never use `eval()` or `exec()` with user input
- Use parameterized queries (SQLModel), never string concatenation
- Validate file uploads (filename, content type, size)
- Mask sensitive data in URLs/logs
- Store secrets in environment variables, never in code

---

## 🧹 Cleanup Requirements

**ALWAYS clean up temporary files and artifacts after completing work:**

- [ ] Remove any `test_*.py` files created for experimentation
- [ ] Delete temporary scripts used for testing approaches
- [ ] Remove backup files (e.g., `*-old.py`, `*-backup.py`, `*-temp.py`)
- [ ] Clean up renamed files when trying different approaches
- [ ] Remove any debugging print statements or temporary logging
- [ ] Delete unused imports that were added during development

**Before finishing any task** - Check for temporary files and remove them.

---

## 📋 Pre-Submission Checklist

Before submitting code, verify:

### Type Safety
- [ ] No bare `dict` usage without proper types
- [ ] Used `dict[str, primitive]` only for simple key-value data
- [ ] Created Pydantic models for complex/nested data structures
- [ ] No `# pyright: ignore` shortcuts to avoid creating proper types
- [ ] No `Any` usage to bypass proper typing
- [ ] All function signatures have explicit type annotations
- [ ] Used modern Python 3.12+ syntax (`str | None` not `Optional[str]`)

### Data Structure
- [ ] Used dot notation (`.property`) instead of dict access (`["key"]`)
- [ ] Added field validation where appropriate in Pydantic models
- [ ] Used `Field(default_factory=list)` for mutable defaults
- [ ] Created models for JSON responses from external APIs
- [ ] Added proper docstrings to Pydantic models

### Standards Compliance
- [ ] Followed structured logging with key-value pairs
- [ ] Added comprehensive docstrings with Args/Returns/Raises
- [ ] Used proper exception handling patterns
- [ ] Followed database session management patterns
- [ ] Used proper imports (no old `Union`, `Optional`, etc.)

---

## 📚 Reference Documentation

This document consolidates key standards. For detailed information, see:

- **[🚨 Critical Violations](./critical-violations.md)** - MUST READ FIRST
- **[🏗️ Data Models & Pydantic](./data-models.md)** - When to use dict vs Pydantic
- **[🔧 Type Annotations](./type-annotations.md)** - Modern typing patterns
- **[🛡️ Security Standards](./security.md)** - Safe coding practices
- **[📋 Self-Check Checklist](./checklist.md)** - Pre-submission validation
- **[📖 Main README](./README.md)** - Quick reference and overview

---

## 🎯 Quick Reference Summary

**Core Principles:**
1. Modern Python Only - Use Python 3.12+ features, no backwards compatibility
2. Type Safety First - Explicit typing for complex objects
3. No Magic Strings - Use object attributes over dictionary access
4. Smart Data Structures - Use appropriate types for complexity level
5. Latest Standards - Follow newest PEP releases
6. No Type Shortcuts - Create proper types instead of ignoring

**Decision Quick Reference:**
- Simple key-value with primitives? → `dict[str, primitive]`
- Complex/nested/validation needed? → Pydantic model
- From external API/JSON? → Pydantic model
- Used across functions? → Pydantic model

**Type Checking:**
- Use **Pyright** (not mypy)
- Rule-specific ignores: `# pyright: ignore[ruleName]`
- Never use generic `# type: ignore`

---

**This document should be read and followed for ALL code generation and review. Any deviation requires explicit justification.**


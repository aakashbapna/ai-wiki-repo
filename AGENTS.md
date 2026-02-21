# Agent Guidelines

## Code Style Rules

### No Inline Imports

All imports must be placed at the top of the file. Do not import modules inside functions, methods, or conditional blocks.

**Bad:**
```python
def get_data():
    import json
    return json.loads(raw)
```

**Good:**
```python
import json

def get_data():
    return json.loads(raw)
```

### Always Use Type Annotations

All functions and methods must have type annotations for parameters and return values. All variables where the type is not immediately obvious should also be annotated.

**Bad:**
```python
def process(items, limit):
    result = []
    for item in items:
        result.append(item)
    return result
```

**Good:**
```python
def process(items: list[str], limit: int) -> list[str]:
    result: list[str] = []
    for item in items:
        result.append(item)
    return result
```

### Avoid `Any`

Do not use `typing.Any`. Use specific types, `Union`, `TypeVar`, or `Protocol` instead. If a type is genuinely unknown, use `object` or define a proper type.

**Bad:**
```python
from typing import Any

def transform(value: Any) -> Any:
    ...
```

**Good:**
```python
from typing import Union

def transform(value: Union[str, int]) -> str:
    ...
```

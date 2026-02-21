"""Detect project/root config files and scan-excluded files by filename."""

# Binary file extensions to exclude from scanning (images, video, audio, compiled, etc.)
_EXCLUDED_EXTENSIONS: frozenset[str] = frozenset({
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp",
    ".ico", ".svg", ".heic", ".heif", ".avif", ".raw", ".psd", ".ai",
    # Video
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v",
    ".mpeg", ".mpg", ".3gp",
    # Audio
    ".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a", ".wma", ".aiff",
    # Fonts
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    # Archives / compressed
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".tgz",
    # Compiled / binary
    ".exe", ".dll", ".so", ".dylib", ".a", ".lib", ".o", ".obj",
    ".pyc", ".pyo", ".class", ".jar", ".war", ".ear",
    # Documents / data blobs
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".db", ".sqlite", ".sqlite3",
    # Other binary formats
    ".bin", ".dat", ".iso", ".img", ".dmg",
    #data files
    ".csv", ".xls",".xlsx",
    ".ipynb"
})

# Files to exclude from scanning: lock files, generated files, etc.
_SCAN_EXCLUDED_FILE_NAMES: frozenset[str] = frozenset({
    # JavaScript / Node lock files
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "bun.lockb",
    # Python lock files
    "Pipfile.lock",
    "poetry.lock",
    # Rust lock files
    "Cargo.lock",
    # Go checksum
    "go.sum",
    # Ruby lock files
    "Gemfile.lock",
    # PHP lock files
    "composer.lock",
    "LICENSE",
    "LICENSE.txt",
})


# Project files: describe sub-projects, how to build, run, test (NOT lock files).
# Case-sensitive where the ecosystem expects it (e.g. Makefile), lowercase otherwise.
_PROJECT_FILE_NAMES: frozenset[str] = frozenset({
    # JavaScript / Node
    "package.json",
    "tsconfig.json",
    # Python
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "requirements.txt",
    # Rust
    "Cargo.toml",
    # Go
    "go.mod",
    # Java / JVM
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
    # .NET
    "project.json",
    "packages.config",
    "Directory.Build.props",
    "Directory.Packages.props",
    # Ruby
    "Gemfile",
    # PHP
    "composer.json",
    # Build / C/C++
    "Makefile",
    "makefile",
    "CMakeLists.txt",
    "meson.build",
    "SConstruct",
    "Justfile",
    # Docker / Ops
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "dockerfile",
})



def is_project_file(file_name: str) -> bool:
    """
    Return True if the file describes sub-projects, how to build, run, or test.
    Includes: package.json, Makefile, pyproject.toml, Cargo.toml, go.mod, pom.xml, etc.
    Excludes: lock files (use is_scan_excluded_file for those).
    """
    if not file_name or not file_name.strip():
        return False
    name = file_name.strip()
    if name in _PROJECT_FILE_NAMES:
        return True
    # .NET project files
    if name.endswith(".csproj") or name.endswith(".vbproj") or name.endswith(".fsproj"):
        return True
    return False


def is_scan_excluded_file(file_name: str) -> bool:
    """
    Return True if the file should be excluded from scanning.
    Excludes: lock files (yarn.lock, Cargo.lock, etc.) and binary files
    (images, video, audio, fonts, archives, compiled objects, etc.) by extension.
    """
    if not file_name or not file_name.strip():
        return False
    name = file_name.strip()
    if name.startswith("."):
        return True
    if name in _SCAN_EXCLUDED_FILE_NAMES:
        return True
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return ext in _EXCLUDED_EXTENSIONS

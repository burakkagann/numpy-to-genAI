@echo off
REM NumPy-to-GenAI Windows Build Script
REM Windows-native replacement for Unix Makefile

setlocal

set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set BUILDDIR=build

if "%1" == "" goto help
if "%1" == "help" goto help
if "%1" == "clean" goto clean
if "%1" == "html" goto html
if "%1" == "serve" goto serve

echo Unknown target: %1
goto help

:help
echo NumPy-to-GenAI Build System (Windows)
echo.
echo Usage: make.bat [target]
echo.
echo Available targets:
echo   html    Build HTML documentation
echo   clean   Remove build directory
echo   serve   Start local HTTP server (port 8000)
echo   help    Show this help message
goto end

:clean
echo Cleaning build directory...
if exist %BUILDDIR% (
    rmdir /s /q %BUILDDIR%
    echo [92m✓ Build directory cleaned[0m
) else (
    echo [93m⚠ Build directory does not exist[0m
)
goto end

:html
echo Building HTML documentation...
echo.

REM Check if sphinx-build is available
where %SPHINXBUILD% >nul 2>&1
if errorlevel 1 (
    echo [91m✗ ERROR: sphinx-build not found![0m
    echo.
    echo Please install Sphinx:
    echo   pip install -r dev_requirements.txt
    echo.
    echo Or run: scripts\setup_env.bat
    exit /b 1
)

REM Run Sphinx build
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR%
if errorlevel 1 (
    echo.
    echo [91m✗ Build failed with errors[0m
    exit /b 1
) else (
    echo.
    echo [92m✓ Build completed successfully![0m
    echo.
    echo Documentation available at: %BUILDDIR%\html\index.html
    echo To view locally, run: make.bat serve
)
goto end

:serve
if not exist %BUILDDIR%\html\index.html (
    echo [91m✗ ERROR: HTML build not found[0m
    echo.
    echo Please run: make.bat html
    exit /b 1
)

echo Starting local HTTP server on port 8000...
echo.
echo Open your browser to: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

cd %BUILDDIR%\html
python -m http.server 8000
goto end

:end
endlocal

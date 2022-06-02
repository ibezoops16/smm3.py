@echo off

if not exist .venv (
    echo Setting up python virtual environment for the project.
    python -m venv .venv
)

echo Activating python virtual environment
call .venv\Scripts\activate

echo Installing project requirements
call python -m pip install -r requirements.txt

echo Setup done.

porosity_detection.bat
A = ImageLoad("C:\a.bmp")
D = Fun1(A, ..)
imshow(D);
E = Fun2(D, ..)
imshow(E)
F = Fun3(E, ...)

dfp.bat

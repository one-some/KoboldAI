@echo off
cd /D %~dp0

:Isolation
call conda deactivate 2>NUL
set Path=%windir%\system32;%windir%;C:\Windows\System32\Wbem;%windir%\System32\WindowsPowerShell\v1.0\;%windir%\System32\OpenSSH\
SET CONDA_SHLVL=
SET PYTHONNOUSERSITE=1
SET PYTHONPATH=

rmdir /S /Q flask_session 2>NUL

TITLE KoboldAI - Server
SET /P M=<loader.settings
IF %M%==1 GOTO drivemap
IF %M%==2 GOTO subfolder
IF %M%==3 GOTO drivemap_B

:subfolder
ECHO Runtime launching in subfolder mode
call miniconda3\condabin\activate
pip install git+https://github.com/neonbjb/tortoise-tts progressbar inflect librosa rotary-embedding-torch unidecode lazy_loader llvmlite numba joblib decorator audioread msgpack pooch scikit-learn soundfile soxr platformdirs threadpoolctl pydantic-core annotated-types pydantic --no-dependencies
pip install torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --no-dependencies
cmd /k
pause
exit

:drivemap
ECHO Runtime launching in K: drive mode
subst /D K: >nul
subst K: miniconda3 >nul
call K:\python\condabin\activate
pip install git+https://github.com/neonbjb/tortoise-tts progressbar inflect librosa rotary-embedding-torch unidecode lazy_loader llvmlite numba joblib decorator audioread msgpack pooch scikit-learn soundfile soxr platformdirs threadpoolctl pydantic-core annotated-types pydantic --no-dependencies
pip install torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --no-dependencies
cmd /k
pause
exit

:drivemap_B
ECHO Runtime launching in B: drive mode
subst /D B: >nul
subst B: miniconda3 >nul
call B:\python\condabin\activate
pip install git+https://github.com/neonbjb/tortoise-tts progressbar inflect librosa rotary-embedding-torch unidecode lazy_loader llvmlite numba joblib decorator audioread msgpack pooch scikit-learn soundfile soxr platformdirs threadpoolctl pydantic-core annotated-types pydantic --no-dependencies
pip install torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --no-dependencies
cmd /k
pause
exit
Quick build instructions

1. Open PowerShell and go to project root:

```
cd c:\Users\vicza\Desktop\TFG
.\.venv\Scripts\Activate.ps1
```

2. Run the helper script:

```
build_exes.bat
```

Notes:
- The batch script installs `pyinstaller` and creates four single-file executables in `dist\`.
- If some modules are missing during the build, install them in the venv first (`py -m pip install -r requirements.txt`).
- To include additional files, edit `build_exes.bat` and add more `--add-data` flags.

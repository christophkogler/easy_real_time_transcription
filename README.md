# Easy [Real Time Whisper Transcription](https://github.com/davabase/whisper_real_time)

This repository is meant to make using the Real Time Whisper Transcription repo easier by providing a simple installation method for all of its requirements as well as CUDA accelerated PyTorch. 

It also includes some minor script improvements:
- Implements [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) to reduce transcription delay further
- Minor adjustment to script logic to make the transcription better
- Transcription log saving

### Conda Torch environments are quite large. 
The virtual environment and Whisper model together is over 10GB!

To install the dependencies, install Anaconda and create an isolated Conda environment with 
```
conda env create -p <env path> --file <environment.yml path>
``` 
Then, activate the environment with 
```
conda activate <env path>
```

Now you should be able to execute `python transcribe_demo.py` or `python transcribe_gui.py`and have real-time transcription!
>On first transcription run, the script will download the Whisper model to the script's directory.
>Whisper 'medium' is the default; it is ~1.5GB on disk and needs ~4GB of VRAM to run.

# Building to an executable
Building the GUI script to an executable can make it much more portable and user friendly, as well as slightly smaller at ~7GB (w/ medium Whisper model).

To convert this from a Python script into a packaged and portable standalone executable, follow these steps after creation of the conda environment:
1) Install ['pyinstaller'](https://pyinstaller.org/en/stable/) to your Conda environment, via `conda install -c conda-forge pyinstaller` (when the environment is active).
> Running pyinstaller on a script should produce two folders named `build` and `dist` in your working directory, each containing a folder named after the script.  
> `dist\transcribe_gui` is the directory we compile into, in this case.
2) Run pyinstaller on the 'transcribe_gui.py' script, ensuring it includes necessary DLLs for CUDA acceleration (which it would otherwise miss), via
```
pyinstaller transcribe_gui.py -D --add-data <venv_path>\Library\bin\cudnn_cnn_infer64_8.dll:. --add-data <venv_path>\Library\bin\cudnn_ops_infer64_8.dll:.
```
3) Run your executable, and test out the real time transcription!

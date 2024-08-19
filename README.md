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

Now you should be able to execute `python transcribe_demo.py` and have real-time transcription!
>On first run, the script will download the Whisper model to the script's directory.
>Whisper 'medium' is the default; it is ~1.5GB on disk and needs ~4GB of VRAM to run.

import os
import sys
import numpy
import speech_recognition as sr
import whisper
import torch
import threading

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
from faster_whisper import WhisperModel
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    # Add more languages as needed... any on https://github.com/openai/whisper/blob/main/whisper/tokenizer.py.
}

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Transcription App")
        self.stop_transcription_flag = False  # Flag to stop transcription
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)  # Handle close event
        self.setup_ui()
    def setup_ui(self):
        # Configure grid to allow resizing
        for i in range(3):
            self.root.grid_columnconfigure(i, weight=1)
        for i in range(12):
            self.root.grid_rowconfigure(i, weight=1)

        # Create a Notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, columnspan=3, sticky="nsew")

        # Create a frame for the transcription tab
        self.transcription_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.transcription_frame, text="Transcription")
		
        # Create a frame for the options tab
        self.options_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.options_frame, text="Options")

        # Options Tab Widgets
        self.create_options_widgets()

        # Transcription Tab Widgets
        self.create_transcription_widgets()

    def create_options_widgets(self):
        # Configure grid for resizing in the options frame
        for i in range(3):
            self.options_frame.grid_columnconfigure(i, weight=1)
        for i in range(9):
            self.options_frame.grid_rowconfigure(i, weight=1)
        # Model selection
        tk.Label(self.options_frame, text="Model").grid(row=0, column=0, sticky="ew")
        self.model_var = tk.StringVar(value="medium")
        model_options = ["tiny", "base", "small", "medium", "large"]
        tk.OptionMenu(self.options_frame, self.model_var, *model_options).grid(row=0, column=1, sticky="ew")

        # Energy threshold
        tk.Label(self.options_frame, text="Energy Threshold").grid(row=1, column=0, sticky="ew")
        self.energy_threshold_var = tk.IntVar(value=1000)
        tk.Entry(self.options_frame, textvariable=self.energy_threshold_var).grid(row=1, column=1, sticky="ew")

        # Record timeout
        tk.Label(self.options_frame, text="Record Timeout (s)").grid(row=2, column=0, sticky="ew")
        self.record_timeout_var = tk.DoubleVar(value=1.0)
        tk.Entry(self.options_frame, textvariable=self.record_timeout_var).grid(row=2, column=1, sticky="ew")

        # Phrase timeout
        tk.Label(self.options_frame, text="Phrase Timeout (s)").grid(row=3, column=0, sticky="ew")
        self.phrase_timeout_var = tk.DoubleVar(value=3.0)
        tk.Entry(self.options_frame, textvariable=self.phrase_timeout_var).grid(row=3, column=1, sticky="ew")

        # Output option
        self.output_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.options_frame, text="Output Transcription to Text File", variable=self.output_var).grid(row=4, column=0, columnspan=2, sticky="ew")

        # Output path
        tk.Label(self.options_frame, text="Output Path").grid(row=4, column=0, sticky="ew")
        executing_script_folder = os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))
        self.output_path_var = tk.StringVar(value=os.path.join(executing_script_folder, "output.txt"))
        tk.Entry(self.options_frame, textvariable=self.output_path_var).grid(row=4, column=1, sticky="ew")
        tk.Button(self.options_frame, text="Browse", command=self.browse_output_path).grid(row=4, column=2, sticky="ew")

        # Model directory
        tk.Label(self.options_frame, text="Model Directory").grid(row=5, column=0, sticky="ew")
        self.model_dir_var = tk.StringVar(value=executing_script_folder)
        tk.Entry(self.options_frame, textvariable=self.model_dir_var).grid(row=5, column=1, sticky="ew")
        tk.Button(self.options_frame, text="Browse", command=self.browse_model_dir).grid(row=5, column=2, sticky="ew")

        # Translate option
        self.do_translation = tk.BooleanVar()
        tk.Checkbutton(self.options_frame, text="Translating and transcribing from different language?", variable=self.do_translation).grid(row=6, column=0, columnspan=2, sticky="ew")

        # Language selection dropdown
        tk.Label(self.options_frame, text="Select spoken language").grid(row=7, column=0, sticky="ew")
        self.translate_from = tk.StringVar(value="english")
        language_options = {value: key for key, value in LANGUAGES.items()}
        tk.OptionMenu(self.options_frame, self.translate_from, *language_options).grid(row=7, column=1, sticky="ew")

        # Default microphone (only for Linux)
        if 'linux' in platform:
            tk.Label(self.options_frame, text="Default Microphone").grid(row=8, column=0, sticky="ew")
            self.default_microphone_var = tk.StringVar(value='pulse')
            tk.Entry(self.options_frame, textvariable=self.default_microphone_var).grid(row=8, column=1, sticky="ew")

    def create_transcription_widgets(self):
        # Start transcription button
        tk.Button(self.transcription_frame, text="Start Transcription", command=self.start_transcription_thread).grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Button(self.transcription_frame, text="Stop Transcription", command=self.stop_transcription).grid(row=0, column=2, sticky="ew")

        # Transcription output text box
        tk.Label(self.transcription_frame, text="Transcription Output:").grid(row=1, column=0, columnspan=3, sticky="ew")
        self.output_text = scrolledtext.ScrolledText(self.transcription_frame, wrap=tk.WORD)
        self.output_text.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

        # Configure grid for resizing
        self.transcription_frame.grid_columnconfigure(0, weight=1)
        self.transcription_frame.grid_columnconfigure(1, weight=1)
        self.transcription_frame.grid_columnconfigure(2, weight=1)
        self.transcription_frame.grid_rowconfigure(2, weight=1)

    def browse_output_path(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path:
            self.output_path_var.set(path)

    def browse_model_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.model_dir_var.set(directory)

    def start_transcription_thread(self):
        # Start the transcription in a separate thread to prevent GUI freezing
        self.stop_transcription_flag = False  # Reset the stop flag
        transcription_thread = threading.Thread(target=self.transcription_process)
        transcription_thread.start()

    def stop_transcription(self):
        self.stop_transcription_flag = True

    def transcription_process(self):
        script_directory = os.path.dirname(__file__)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Got CUDA?: {torch.cuda.is_available()}")

        output_file_path = None
        if self.output_var.get():
            if not self.output_path_var.get():
                output_file_path = os.path.join(script_directory, "output.txt")
            else:
                output_file_path = os.path.normpath(self.output_path_var.get())

        if 'linux' in platform:
            mic_name = self.default_microphone_var.get()
            if not mic_name or mic_name == 'list':
                print("Available microphone devices are: ")
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    print(f"Microphone with name \"{name}\" found")
                return
            else:
                found_microphone = False
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        source = sr.Microphone(sample_rate=16000, device_index=index)
                        found_microphone = True
                        break
                if not found_microphone:
                    print(f"Microphone with name \"{mic_name}\" not found, please enter a name from the following:")
                    for index, name in enumerate(sr.Microphone.list_microphone_names()):
                        print(f"Microphone with name \"{name}\" found")
                    return
        else:
            source = sr.Microphone(sample_rate=16000)

        model = self.model_var.get()
        if model != "large" and not self.do_translation.get():
            model = model + ".en"
        if self.model_dir_var.get():
            audio_model = whisper.load_model(model, download_root=self.model_dir_var.get()).to(device)
        else:
            audio_model = whisper.load_model(model, download_root=script_directory).to(device)
        print("Whisper model loaded.\n")

        audio_model = WhisperModel(model, device=device, cpu_threads=8)
        print("'faster-whisper' modified model loaded.\n")

        last_response_time = None
        data_queue = Queue()

        def record_callback(_, audio: sr.AudioData):
            data = audio.get_raw_data()
            data_queue.put(data)

        recorder = sr.Recognizer()
        recorder.energy_threshold = self.energy_threshold_var.get()
        recorder.dynamic_energy_threshold = False

        print("Adjusting for ambient sound level of microphone...\n")
        with source:
            recorder.adjust_for_ambient_noise(source)
        print("Complete!\n")

        recorder.listen_in_background(source, record_callback, phrase_time_limit=self.record_timeout_var.get())

        phrase_timeout = self.phrase_timeout_var.get()
        audio_data = b''
        transcription = ['']
        start_date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        print("Beginning transcription.\n")

        if output_file_path:
            output_file = open(output_file_path, 'a')
            output_file.write(start_date_time + '\n')

        while not self.stop_transcription_flag:
            try:
                now = datetime.now()
                if not data_queue.empty():
                    phrase_complete = False
                    if last_response_time and now - last_response_time > timedelta(seconds=phrase_timeout):
                        phrase_complete = True
                        audio_data = b''

                    last_response_time = now

                    audio_data = audio_data + b''.join(data_queue.queue)
                    data_queue.queue.clear()

                    audio_np = numpy.frombuffer(audio_data, dtype=numpy.int16).astype(numpy.float32) / 32768.0

					#create a list from languages.keys and select the same index where translate_from is located in the list made from languages.values
                    selected_language_key = list(LANGUAGES.keys())[list(LANGUAGES.values()).index(self.translate_from.get())]

                    if self.do_translation.get(): # Attempt to translate to english and transcribe
                        segments, info = audio_model.transcribe(audio_np, beam_size=12, task="translate", language = selected_language_key)
                        detected_language = info.language
                        if detected_language != selected_language_key:
                            print(f"Attempting to translate from {self.translate_from.get()} but detected language is {detected_language}, quality may be degraded.")
                        text = ''.join(segment.text for segment in segments).strip()
                    else: # Attempt english transcription
                        segments, info = audio_model.transcribe(audio_np, beam_size=12)
                        text = ''.join(segment.text for segment in segments).strip()

                    if phrase_complete:
                        if output_file_path:
                            output_file.write(transcription[-1] + '\n')
                        transcription.append(text)
                    else:
                        transcription[-1] = text

                    # Update the text box with the transcription (done on the main thread)
                    self.output_text.after(0, self.update_output_text, transcription)

                    os.system('cls' if os.name=='nt' else 'clear') # Clear the console
                    for line in transcription:
                        print(line, flush=True)    # Reprint the updated transcription.
                    print('', end='', flush=True)    # Flush stdout.

                else:
                    sleep(0.25)
            except KeyboardInterrupt:
                break

        # End of transcription handling
        if output_file_path:
            #output_file.write("\nTranscription ended.\n")
            output_file.close()

        if output_file_path:
            self.output_text.after(0, self.update_output_text, transcription + ["\nTranscription ended. Log appended to file."])
        else:
            self.output_text.after(0, self.update_output_text, transcription + ["\nTranscription ended."])
        print("Transcription ended. Log saved.")

    def update_output_text(self, transcription):
        # Clear the text box and insert the latest transcription text
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "\n".join(transcription))

    def on_close(self):
        # Gracefully stop transcription before closing
        self.stop_transcription()
        # Wait for the transcription thread to exit
        self.root.after(500, self.root.destroy)

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()

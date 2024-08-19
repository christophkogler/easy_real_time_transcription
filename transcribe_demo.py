import argparse					# Import the argparse module for parsing command line arguments
import os						# Import the os module for working with operating system functionality, such as file paths and directory management
import numpy					# Import NumPy (a library for efficient numerical computation) under the alias 'np'
import speech_recognition as sr	# Import SpeechRecognition (a library for recognizing spoken words from audio data) under the alias 'sr'
import whisper 					# Import openai's Whisper handling library (a state-of-the-art speech recognition model)
import torch					# Import PyTorch (open-source machine learning framework)

from datetime import datetime, timedelta	# Import the datetime and timedelta classes from the datetime module
from queue import Queue						# Import the Queue class for thread-safe data passing between threads
from time import sleep						# Import the sleep function to pause execution of a program
from sys import platform					# Import the platform variable that holds information about the operating system
from faster_whisper import WhisperModel		# Import the altered WhisperModel from faster_whisper for accelerated transcription

def main():
    parser = argparse.ArgumentParser()  # Create an ArgumentParser object for parsing command line arguments.

    parser.add_argument("--model", default="medium", help="Model to use", 	
                        choices=["tiny", "base", "small", "medium", "large"])#medium is ~1.5gb on disk and uses ~4gb of vram
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=2000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=1,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=1,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    parser.add_argument("--output", default=True, help="Output transcription as text file?", type=bool) 
    parser.add_argument("--output_path", default=None, help="Output text file path. Defaults to script directory.", type=str)
    parser.add_argument("--model_dir", default=None, help="Whisper model save directory path. Defaults to script directory.", type=str)
	
    if 'linux' in platform: # Only Linux people need this
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)

    args = parser.parse_args() # Parse all command line arguments using the ArgumentParser object

    script_directory = os.path.dirname(__file__) # Get the directory path of this Python file
    device = "cuda" if torch.cuda.is_available() else "cpu" #Determine whether or not we have access to a CUDA-capable GPU and attempt use it as our default tensor device; otherwise fall back on CPU.
    print(f"Got CUDA?: {torch.cuda.is_available()}")# Print the result of determining CUDA availability

    
    output_file_path = None
    if args.output: # If output enabled, get output file path
        if args.output_path == None: # Default
            output_file_path = os.path.join(script_directory, "output.txt") # The default path for saving the transcription.
        else: # User specified output file
            output_file_path = os.path.normpath(args.output_path)  # Attempt to use a user-specified file path for saving the transcription.

    #print(sr.Microphone.list_working_microphones()) # Print a list of available microphones

    if 'linux' in platform:    # Important for linux users: prevents permanent application hang and crash by using the wrong Microphone
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list': # no default microphone or list requested
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()): # Enumerate through the list of microphones and for each,
                print(f"Microphone with name \"{name}\" found") # print their names.
			return # exit early to prevent hang
        else: # user entered a microphone name
			found_microphone = False # Assume they didnt enter a valid name, until we find a match
            for index, name in enumerate(sr.Microphone.list_microphone_names()): # Iterate over all available microphone devices and for each,
                if mic_name in name: #  check if it matches the specified name.
                    source = sr.Microphone(sample_rate=16000, device_index=index)
					found_microphone = True
                    break # break out of for() loop
			if not found_microphone: # if they didnt input a valid name
                print(f"Microphone with name \"{mic_name}\" not found, please enter a name from the following:") # inform user of error
                for index, name in enumerate(sr.Microphone.list_microphone_names()): # Enumerate through the list of microphones and for each,
                    print(f"Microphone with name \"{name}\" found") # print their names.
				return # exit early to prevent hang
    else:
        source = sr.Microphone(sample_rate=16000)

    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    if args.model_dir: # if the user set a custom download directory
        audio_model = whisper.load_model(model, download_root=args.model_dir).to(device) # load/download in custom dir
    else: # else
        audio_model = whisper.load_model(model, download_root=script_directory).to(device) # load/download in same directory as script
    print("Whisper model loaded.\n")

    # compute_type = 'int8' if 'distil' in model else 'auto' # Use a lower precision, if available.
    audio_model = WhisperModel(model, 
                               device=device, 
                               #compute_type=compute_type, # seems to degrade quality too much
                               cpu_threads=8)
    print("'faster-whisper' modified model loaded.\n")
    
    last_response_time = None # The last time a recording was retrieved from the queue.
    
    data_queue = Queue() # Thread safe Queue for passing data from the threaded recording callback.
    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        data = audio.get_raw_data() # Grab the raw bytes...
        data_queue.put(data) # and push them into the thread safe queue.

    recorder = sr.Recognizer() # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False    # Dynamic energy compensation often lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
	
    print("Adjusting for ambient sound level of microphone...\n")
    with source:
        recorder.adjust_for_ambient_noise(source) # Adjust recorder for ambient noise level of the microphone.
    print("Complete!\n")
	
    record_timeout = args.record_timeout
    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)    #call record_callback whenever listen_in_background gets sounds from source. (split by phrase_time_limit?)

    phrase_timeout = args.phrase_timeout
    audio_data = b''
    transcription = ['']
    start_date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S") # for logging
	
    print("Beginning transcription.\n")
	
    while True:
        try:
            now = datetime.now() # get current time for the loop
            if not data_queue.empty(): # if there is data in the queue (recognizer detected 'speech')
                phrase_complete = False # reset phrase_complete
                if last_response_time and now - last_response_time > timedelta(seconds=phrase_timeout): # If enough time has passed between recordings, consider the phrase complete.
                    phrase_complete = True 	# Start new phrase...
                    audio_data = b'' 		# Clear the phrase audio buffer...

                last_response_time = now # The last time we received new audio data from the queue is now, now
                
                audio_data = audio_data + b''.join(data_queue.queue) # Combine our 'current phrase' audio data with new incoming audio data from queue
                data_queue.queue.clear() # Clear out the queue, so we don't re-append the same data.
                
                # Convert in-ram buffer to something the model can use directly without needing a temp file.
                # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                audio_np = numpy.frombuffer(audio_data, dtype=numpy.int16).astype(numpy.float32) / 32768.0

                # Read the transcription.
                #result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                #text = result['text'].strip()
                
                segments, info = audio_model.transcribe(audio_np, beam_size=12) # Get the transcription(s)
                text = ''.join(segment.text for segment in segments).strip()	# Concat the text and clean any trailing/leading whitespace(s)
                
                if phrase_complete: 			# If we detected a pause between recordings
                    transcription.append(text) 		# add a new item to our transcription.
                else: 							# Otherwise
                    transcription[-1] = text 		# edit the last one.

                os.system('cls' if os.name=='nt' else 'clear') # Clear the console
                for line in transcription:
                    print(line, flush=True)    # Reprint the updated transcription.
                print('', end='', flush=True)    # Flush stdout.
            else: # If the data queue is empty
                sleep(0.25)  #Sleep a short time. (No infinite loops)
        except KeyboardInterrupt: # If we get a keyboard interrupt
            break # Break the infinite loop
	
    if output_file_path: # If output is enabled
        output_file = open(output_file_path, 'a') # Open the file in append mode
        output_file.write(start_date_time + '\n') # Add date&time for easy searching
        for line in transcription:
            output_file.write(line + '\n')	# Write transcription to output file, line by line
        print(f"\n\nTranscription saved to {output_file_path}")
        for _ in range(5):
            output_file.write('\n') # write 5 newlines after log
        output_file.close() #close output file
		
    print("\n\nTranscription:")
	
    for line in transcription:
        print(line) # print transcription to console, line by line
#end of main.

if __name__ == "__main__":
    main()

from pydub import AudioSegment
import os

def split_audio(file_path, output_folder, starting_time_in_sec, ending_time_in_sec):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    
    starting_time_of_segment = starting_time_in_sec * 1000
    ending_time_of_segment = ending_time_in_sec * 1000
        
    # Extract segment
    segment = audio[starting_time_of_segment:ending_time_of_segment]
    
    # Define output file path
    output_file_path = os.path.join(output_folder, f'segment_output.mp3')
    
    # Export segment
    segment.export(output_file_path, format="mp3")

    
    print("Audio splitting completed!")

# Example usage
input_file = 'Every Transgender Joke That Made Twitter Mob Crazy [TubeRipper.com].mp3'
output_directory = 'audio_outputs'
split_audio(input_file, output_directory, 25, 165)

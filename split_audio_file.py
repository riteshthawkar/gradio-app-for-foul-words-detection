from pydub import AudioSegment
import os

def split_audio(file_path, output_folder):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    
    # Calculate the duration of the audio in milliseconds
    duration_ms = len(audio)
    
    # Define segment length (1 minute in milliseconds)
    segment_length_ms = 120 * 1000
    
    # Initialize start position and segment counter
    start_pos = 0
    segment_counter = 1
    
    while start_pos < duration_ms:
        end_pos = start_pos + segment_length_ms
        
        # Check if we are at the last segment
        if end_pos > duration_ms:
            last_segment_length = duration_ms - start_pos
            if last_segment_length < 30 * 1000:  # Less than 30 seconds
                # Add to previous segment
                end_pos = start_pos + last_segment_length + segment_length_ms
            else:
                # Make it a separate segment
                end_pos = duration_ms
        
        # Extract segment
        segment = audio[start_pos:end_pos]
        
        # Define output file path
        output_file_path = os.path.join(output_folder, f'segment_{segment_counter}.mp3')
        
        # Export segment
        segment.export(output_file_path, format="mp3")
        
        # Update start position and segment counter
        start_pos = end_pos
        segment_counter += 1
    
    print("Audio splitting completed!")

# Example usage
input_file = 'arabic_english_audios/audios/english_audio_27.mp3'
output_directory = 'audio_outputs'
split_audio(input_file, output_directory)
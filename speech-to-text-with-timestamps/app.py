import gradio as gr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import pandas as pd
import re
from pydub import AudioSegment
from pydub.generators import Sine
import io
# import torch

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, low_cpu_mem_usage=True, use_safetensors=True
)
# model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=8,
    # device=device,
)


arabic_bad_Words = pd.read_csv("arabic_bad_words_dataset.csv")
english_bad_Words = pd.read_csv("english_bad_words_dataset.csv")


def clean_english_word(word):
    cleaned_text = re.sub(r'^[\s\W_]+|[\s\W_]+$', '', word)
    return cleaned_text.lower()

def clean_arabic_word(word):
    pattern = r'[^\u0600-\u06FF]'
    cleaned_word = re.sub(pattern, '', word)
    return cleaned_word

def classifier(word_list_with_timestamp, language):

    foul_words = []
    negative_timestamps = []

    if language == "English":
        list_to_search = set(english_bad_Words["words"])
        for item in word_list_with_timestamp:
            word = clean_english_word(item['text'])
            if word.lower() in list_to_search:
                foul_words.append(word)
                negative_timestamps.append(item['timestamp'])
    else:
        list_to_search = list(arabic_bad_Words["words"])
        for item in word_list_with_timestamp:
            word = clean_arabic_word(item['text'])
            for word_in_list in list_to_search:
                if word_in_list == word:
                    foul_words.append(word)
                    negative_timestamps.append(item['timestamp'])   
        
    return [foul_words, negative_timestamps]

def generate_bleep(duration_ms, frequency=1000):
    sine_wave = Sine(frequency)
    bleep = sine_wave.to_audio_segment(duration=duration_ms)
    return bleep

def mute_audio_range(audio_filepath, ranges, bleep_frequency=800):
    audio = AudioSegment.from_file(audio_filepath)
    for range in ranges:
        start_time = range[0]
        end_time = range[-1]
        start_ms = start_time * 1000  # pydub works with milliseconds
        end_ms = end_time * 1000
        duration_ms = end_ms - start_ms
        
        # Generate the bleep sound
        bleep_sound = generate_bleep(duration_ms, bleep_frequency)
        
        # Combine the original audio with the bleep sound
        audio = audio[:start_ms] + bleep_sound + audio[end_ms:]

    return audio

def resample_audio(audio_segment, target_sample_rate=16000):
    return audio_segment.set_frame_rate(target_sample_rate).set_channels(1).set_sample_width(2)

def format_output_to_list(data):
    formatted_list = "\n".join([f"{item['timestamp'][0]}s - {item['timestamp'][1]}s \t : {item['text']}" for item in data])
    return formatted_list

def transcribe(input_audio, audio_language, task, timestamp_type):
    if input_audio is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")

    if timestamp_type == "sentence":
        timestamp_type = True
    else:
        timestamp_type = "word"

    output = pipe(input_audio, return_timestamps=timestamp_type, generate_kwargs={"task": task})
    text = output['text']

    timestamps = format_output_to_list(output['chunks'])

    foul_words, negative_timestamps = classifier(output['chunks'], audio_language)
    foul_words = ", ".join(foul_words)


    audio_output = mute_audio_range(input_audio, negative_timestamps)

    # Resample the output audio to 16kHz
    audio_output = resample_audio(audio_output, 16000)

    # Save the output audio to a BytesIO object
    output_buffer = io.BytesIO()
    audio_output.export(output_buffer, format="wav")
    output_buffer.seek(0)
    
    # Read the audio data from the BytesIO buffer
    sample_rate = audio_output.frame_rate
    audio_data = np.frombuffer(output_buffer.read(), dtype=np.int16)

    return  [text, timestamps, foul_words, (sample_rate, audio_data)]

cache_examples = [
    ["arabic_english_audios/audios/english_audio_18.mp3", 'English', 'transcribe', 'word'],
    ["arabic_english_audios/audios/english_audio_20.mp3", 'English', 'transcribe', 'word'],
    ["arabic_english_audios/audios/english_audio_21.mp3", 'English', 'transcribe', 'word'],
    ["arabic_english_audios/audios/english_audio_22.mp3", 'English', 'transcribe', 'word'],
    ["arabic_english_audios/audios/english_audio_27.mp3", 'English', 'transcribe', 'word'],
    ["arabic_english_audios/audios/english_audio_29.mp3", 'English', 'transcribe', 'word'],
]
examples = [
        ["arabic_english_audios/audios/arabic_audio_11.mp3", 'Arabic', 'transcribe', 'word'],
        ["arabic_english_audios/audios/arabic_audio_12.mp3", 'Arabic', 'transcribe', 'word'],
        ["arabic_english_audios/audios/arabic_audio_13.mp3", 'Arabic', 'transcribe', 'word'],
    
        # ["arabic_english_audios/audios/english_audio_18.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_19.mp3", 'English', 'transcribe', 'word'],
        # ["arabic_english_audios/audios/english_audio_20.mp3", 'English', 'transcribe', 'word'],
        # ["arabic_english_audios/audios/english_audio_21.mp3", 'English', 'transcribe', 'word'],
        # ["arabic_english_audios/audios/english_audio_22.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_23.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_24.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_25.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_26.mp3", 'English', 'transcribe', 'word'],
        # ["arabic_english_audios/audios/english_audio_27.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_28.mp3", 'English', 'transcribe', 'word'],
        # ["arabic_english_audios/audios/english_audio_29.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_30.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_31.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_32.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_33.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_34.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_35.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_36.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_37.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_38.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_39.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_40.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_41.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_42.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_43.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_44.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_45.mp3", 'English', 'transcribe', 'word'],
    ]

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.HTML("<h2 style='text-align: center;'>Transcribing Audio with Timestamps using whisper-large-v3</h2>")
    # gr.Markdown("")
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["upload", 'microphone'], type="filepath", label="Audio file")
            audio_language = gr.Radio(["Arabic", "English"], label="Audio Language")
            task = gr.Radio(["transcribe", "translate"], label="Task")
            timestamp_type = gr.Radio(["sentence", "word"], label="Timestamp Type")
            with gr.Row():
                clear_button = gr.ClearButton(value="Clear")
                submit_button = gr.Button("Submit", variant="primary", )

        with gr.Column():
            transcript_output = gr.Text(label="Transcript")
            timestamp_output = gr.Text(label="Timestamps")
            foul_words = gr.Text(label="Foul Words")
            output_audio = gr.Audio(label="Output Audio", type="numpy")
            
    cache_examples = gr.Examples(cache_examples, inputs=[audio_input, audio_language, task, timestamp_type], outputs=[transcript_output, timestamp_output,  foul_words, output_audio], fn=transcribe, examples_per_page=10,  cache_examples=True, label="Sample Examples")
    non_cache_examples = gr.Examples(examples, inputs=[audio_input, audio_language, task, timestamp_type], outputs=[transcript_output, timestamp_output,  foul_words, output_audio], fn=transcribe, examples_per_page=50,  cache_examples=False, label="Other Examples")

    submit_button.click(fn=transcribe, inputs=[audio_input, audio_language, task, timestamp_type], outputs=[transcript_output, timestamp_output, foul_words, output_audio])
    clear_button.add([audio_input, audio_language, task, timestamp_type, transcript_output, timestamp_output, foul_words, output_audio])


if __name__ == "__main__":
    demo.launch()

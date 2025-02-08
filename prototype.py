import whisper
import ffmpeg
import pysrt
import spacy
import os
import boto3
from botocore.exceptions import ClientError
from transformers import MarianMTModel, MarianTokenizer

# Step 1: Extract audio from video
def extract_audio(video_path, audio_path="audio.mp3"):
    ffmpeg.input(video_path).output(
        audio_path, 
        ac=1, 
        ar=16000,
        af="highpass=f=200,lowpass=f=3000",  # Filter noise
        loglevel="quiet"  # Suppress FFmpeg output
    ).run(overwrite_output=True)
    return audio_path

# Step 2: Transcribe audio using Whisper
def transcribe_audio(audio_path):
    # First pass with small model
    model_small = whisper.load_model("small")
    result = model_small.transcribe(
        audio_path,
        fp16=False,
        verbose=False,
        temperature=0,
        beam_size=1,
        compression_ratio_threshold=2.4
    )
    
    # Identify low-confidence segments
    low_confidence_segments = [
        (seg['start'], seg['end'])
        for seg in result['segments']
        if seg['no_speech_prob'] > 0.5
    ]

    if low_confidence_segments:
        model_medium = whisper.load_model("medium")
        
        for start, end in low_confidence_segments:
            refined = model_medium.transcribe(
                audio_path,
                fp16=False,
                verbose=False,
                temperature=0,
                condition_on_previous_text=False,
                word_timestamps=False,
                initial_prompt=result['text'],
                start_time=start,
                end_time=end
            )
            # Replace original segment with refined version
            result['segments'] = [
                refined_seg if (refined_seg['start'] >= start and refined_seg['end'] <= end)
                else seg
                for seg in result['segments']
                for refined_seg in refined['segments']
            ]
    return result["segments"]

# Initialize AWS Translate client
def get_translate_client():
    return boto3.client(
        'translate',
        region_name=os.getenv("AWS_DEFAULT_REGION")
    )

# Enhanced translation function with error handling
def translate_text(text, target_lang="hi"):
    translate = get_translate_client()
    try:
        result = translate.translate_text(
            Text=text,
            SourceLanguageCode='en',
            TargetLanguageCode=target_lang,
            Settings={
                'Formality': 'INFORMAL',  # Better for subtitles
                'Profanity': 'MASK'  # Filter inappropriate content
            }
        )
        return result['TranslatedText']
    except ClientError as e:
        print(f"AWS Translation error: {e.response['Error']['Message']}")
        return text  # Fallback to original text
    except Exception as e:
        print(f"General translation error: {str(e)}")
        return text

# Load SpaCy model for English
nlp = spacy.load("en_core_web_sm")

# Correct punctuation and grammar
def correct_text(text):
    doc = nlp(text)
    corrected_text = " ".join([sent.text for sent in doc.sents])
    return corrected_text

# Step 4: Generate SRT subtitles
def generate_subtitles(segments, target_lang, output_path="output.srt"):
    base_name = os.path.splitext(output_path)[0]
    output_path = f"{base_name}_{target_lang}.srt"
    subtitles = pysrt.SubRipFile()
    for i, segment in enumerate(segments):
        start_time = int(segment["start"] * 1000)
        end_time = int(segment["end"] * 1000)
        text = segment["text"]

        # Translate text using AWS
        translated_text = translate_text(text, target_lang)
        
        # Add to subtitles
        subtitles.append(pysrt.SubRipItem(
            index=i+1,
            start=pysrt.SubRipTime(milliseconds=start_time),
            end=pysrt.SubRipTime(milliseconds=end_time),
            text=translated_text
        ))
    
    subtitles.save(output_path)
    os.sync()  # Force write to disk
    return output_path

# Main function
def main(video_path, target_lang="hi"):
    # Step 1: Extract audio
    audio_path = extract_audio(video_path)
    print("Audio extracted successfully.")

    # Step 2: Transcribe audio
    segments = transcribe_audio(audio_path)
    print("Audio transcribed successfully.")

    # Step 3: Generate subtitles with AWS Translate
    segments = merge_short_segments(segments)
    subtitle_path = generate_subtitles(segments, target_lang)
    print(f"Subtitles generated and saved to {subtitle_path}.")

    # Return the subtitle path
    return subtitle_path

def marianmt_translate(segments, target_lang="hi"):
    model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    translated_segments = []
    for segment in segments:
        inputs = tokenizer(segment["text"], return_tensors="pt", padding=True)
        outputs = model.generate(**inputs)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated_segments.append({**segment, "text": translated_text})
    return translated_segments

def get_secret():
    client = boto3.client('secretsmanager')
    return client.get_secret_value(SecretId='subtitle-app-credentials')['SecretString']

def whisper_transcribe(audio_path):
    model = whisper.load_model("medium")
    return model.transcribe(audio_path, language="en")["segments"]

def cleanup_files(*paths):
    """Securely delete generated files with single-pass overwrite"""
    for path in paths:
        try:
            if os.path.exists(path) and os.path.isfile(path):
                # Single overwrite with random data (balance security/speed)
                with open(path, "wb") as f:
                    file_size = os.path.getsize(path)
                    f.write(os.urandom(file_size))  # Overwrite with random bytes
                os.remove(path)
                print(f"Securely deleted: {path}")
        except Exception as e:
            print(f"Error deleting {path}: {str(e)}")
            # Optional: Add error notification to UI

def merge_short_segments(segments, min_duration=1.0):
    merged = []
    for segment in segments:
        if not merged:
            merged.append(segment)
        else:
            last = merged[-1]
            # Merge if gap < 0.5s and total < 5s
            if (segment['start'] - last['end'] < 0.5 and 
                segment['end'] - last['start'] < 5.0):
                last['end'] = segment['end']
                last['text'] += " " + segment['text']
            else:
                merged.append(segment)
    return merged

# main.py

import whisper
import torchaudio
from textblob import TextBlob
from transformers import pipeline
import os

# -------- CONFIG --------
AUDIO_PATH = "sample_audio/test_call.m4a"  # Change this to your file path
MODEL_SIZE = "base"
# ------------------------

print("🔊 Loading Whisper model...")
model = whisper.load_model(MODEL_SIZE)

print("🎧 Transcribing call...")
result = model.transcribe(AUDIO_PATH)
transcript = result["text"].lower()
print("\n📄 Transcript:\n", transcript)

# -------- QA ANALYSIS --------

# 1. Greeting detection
greetings = ["hello", "hi", "good morning", "good evening", "namaste"]
greeting_present = any(word in transcript for word in greetings)
greeting_score = 2 if greeting_present else 0

# 2. Keyword detection
keywords = ["help", "support", "problem", "issue", "thank you", "resolved", "sorry", "feedback"]
found_keywords = [word for word in keywords if word in transcript]
keyword_score = min(len(found_keywords), 4)

# 3. Sentiment analysis
sentiment = TextBlob(transcript).sentiment
sentiment_score = 2 if sentiment.polarity > 0 else 0

# 4. Duration
waveform, sample_rate = torchaudio.load(AUDIO_PATH)
duration = waveform.shape[1] / sample_rate
duration_score = 2 if duration > 5 else 0

# 5. Total scoring
total_score = greeting_score + keyword_score + sentiment_score + duration_score

print("\n🧠 QA ANALYSIS")
print("✅ Greeting:", "Yes" if greeting_present else "No")
print("🔑 Keywords found:", found_keywords)
print("😊 Sentiment:", sentiment.polarity, "-",
      "Positive" if sentiment.polarity > 0 else "Negative" if sentiment.polarity < 0 else "Neutral")
print(f"⏱ Duration: {duration:.1f} sec")
print("\n📊 FINAL SCORE: ", total_score, "/ 10")

# 6. Summary
print("\n📝 Generating summary...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

if len(transcript) > 1000:
    transcript = transcript[:1000]

summary = summarizer(transcript, max_length=60, min_length=20, do_sample=False)
print("📋 Summary:", summary[0]['summary_text'])

# 📞 Call Quality Analyst AI (Open Source)

This is an open-source AI tool that listens to call recordings and behaves like a human call quality analyst. It:
- Converts call audio to text
- Detects greetings, keywords, sentiment
- Scores the call (out of 10)
- Summarizes the call using AI

---

## 🧠 How It Works

1. Uses [Whisper](https://github.com/openai/whisper) for speech-to-text
2. Uses TextBlob for sentiment analysis
3. Uses Hugging Face’s `bart-large-cnn` to summarize the call
4. Outputs a full QA-style report with scoring

---

## 🛠️ Setup

```bash
git clone https://github.com/Mrtiget008/call-quality-analyst-ai.git
cd call-quality-analyst-ai
pip install -r requirements.txt
python main.py

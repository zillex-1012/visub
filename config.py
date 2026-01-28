"""
VietDub Solo - Configuration Constants
"""

# Whisper Models
WHISPER_MODELS = {
    "tiny": "Tiny (Nhanh nhất, kém chính xác)",
    "base": "Base (Cân bằng - Khuyên dùng)",
    "small": "Small (Tốt hơn, chậm hơn)",
    "medium": "Medium (Rất tốt, cần GPU)",
    "large-v3": "Large V3 (Tốt nhất, cần GPU mạnh)"
}

# Translation Models via OpenRouter
TRANSLATION_MODELS = {
    # Free Models
    "meta-llama/llama-3.3-70b-instruct:free": "Llama 3.3 70B (Free - Khuyên dùng)",
    "allenai/molmo-2-8b:free": "Molmo 2 8B (Free)",
    
    # Paid Models
    "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B ($0.02/$0.05 per 1M)",
    "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite ($0.10/$0.40 per 1M)"
}

# TTS Providers
TTS_PROVIDERS = {
    "fpt": "FPT.AI (Giọng Việt tốt nhất)",
    "elevenlabs": "ElevenLabs (Multilingual)",
    "openai": "OpenAI TTS (Dễ dùng)"
}

# FPT.AI Voices
FPT_VOICES = {
    "banmai": "Ban Mai (Nữ Bắc)",
    "leminh": "Lê Minh (Nam Bắc)",
    "thuminh": "Thu Minh (Nữ Bắc)",
    "giahuy": "Gia Huy (Nam Bắc)",
    "myan": "Mỹ An (Nữ Nam)",
    "lannhi": "Lan Nhi (Nữ Nam)",
    "linhsan": "Linh San (Nữ Trung)",
    "minhquang": "Minh Quang (Nam Trung)"
}

# ElevenLabs Voices (Vietnamese support)
ELEVENLABS_VOICES = {
    "21m00Tcm4TlvDq8ikWAM": "Rachel (Female)",
    "AZnzlk1XvdvUeBnXmlld": "Domi (Female)", 
    "EXAVITQu4vr4xnSDxMaL": "Bella (Female)",
    "ErXwobaYiN019PkySvjV": "Antoni (Male)",
    "MF3mGyEYCl7XYWbV9V6O": "Elli (Female)",
    "TxGEqnHWrfWFTfGW9XjX": "Josh (Male)"
}

# OpenAI TTS Voices
OPENAI_VOICES = {
    "alloy": "Alloy (Neutral)",
    "echo": "Echo (Male)",
    "fable": "Fable (British)",
    "onyx": "Onyx (Male Deep)",
    "nova": "Nova (Female)",
    "shimmer": "Shimmer (Female)"
}

# Default Values
DEFAULTS = {
    "whisper_model": "tiny",
    "translation_model": "meta-llama/llama-3.3-70b-instruct:free",
    "tts_provider": "fpt",
    "voice": "banmai",
    "speed": 1.0,
    "original_volume": 0.1,
    "dubbing_volume": 1.0
}

# Translation Prompt Template
TRANSLATION_PROMPT = """You are a professional English to Vietnamese translator.

TASK: Translate the following English sentences to Vietnamese.

CRITICAL RULES:
1. KEEP English terms, specialized concepts, and proper names (e.g., AI, Machine Learning, blockchain, YouTube, ICT, stop rate, setup, order flow, etc.). DO NOT translate them.
2. Translate ONLY the surrounding context to Vietnamese.
3. Use natural Vietnamese as Vietnamese people speak.
4. Keep the meaning and emotion of the original.

OUTPUT FORMAT: Return a JSON array with this exact format:
[{{"id": 1, "vietnamese": "bản dịch tiếng Việt"}}]

EXAMPLE:
Input: [{{"id": 1, "english": "In this video, we will discuss the ICT entry checklist and stop rate."}}]
Output: [{{"id": 1, "vietnamese": "Trong video này, chúng ta sẽ thảo luận về ICT entry checklist và stop rate."}}]

INPUT TO TRANSLATE:
{segments}

Remember: Output MUST be in Vietnamese language, BUT keep English terms as is!
"""

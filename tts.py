"""
VietDub Solo - Text-to-Speech Module
Hỗ trợ nhiều TTS providers: FPT.AI, ElevenLabs, OpenAI
"""

import requests
import os
import tempfile
from typing import Optional, List, Dict
from pydub import AudioSegment
import time


class TTSProvider:
    """Base class for TTS providers"""
    
    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> Optional[str]:
        """
        Synthesize text to speech
        
        Args:
            text: Text to synthesize
            voice: Voice ID
            speed: Speech speed (0.8 - 1.2)
        
        Returns:
            Path to audio file or None if failed
        """
        raise NotImplementedError


class FPTProvider(TTSProvider):
    """FPT.AI TTS Provider - Best for Vietnamese"""
    
    API_URL = "https://api.fpt.ai/hmi/tts/v5"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def synthesize(self, text: str, voice: str = "banmai", speed: float = 1.0) -> Optional[str]:
        if not self.api_key:
            raise ValueError("FPT.AI API key is required")
        
        headers = {
            "api-key": self.api_key,
            "speed": str(speed),
            "voice": voice
        }
        
        try:
            response = requests.post(
                self.API_URL,
                headers=headers,
                data=text.encode('utf-8'),
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get("async"):
                return None
            
            # FPT.AI trả về URL, cần download
            audio_url = data["async"]
            
            # Đợi file sẵn sàng
            time.sleep(1)
            
            # Download audio
            audio_response = requests.get(audio_url, timeout=30)
            audio_response.raise_for_status()
            
            # Lưu file
            output_path = tempfile.mktemp(suffix=".mp3")
            with open(output_path, 'wb') as f:
                f.write(audio_response.content)
            
            return output_path
            
        except Exception as e:
            print(f"FPT.AI TTS error: {e}")
            return None


class ElevenLabsProvider(TTSProvider):
    """ElevenLabs TTS Provider - Good for multilingual"""
    
    API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def synthesize(self, text: str, voice: str = "21m00Tcm4TlvDq8ikWAM", speed: float = 1.0) -> Optional[str]:
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        try:
            response = requests.post(
                f"{self.API_URL}/{voice}",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            output_path = tempfile.mktemp(suffix=".mp3")
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            # Adjust speed if needed
            if speed != 1.0:
                output_path = adjust_audio_speed(output_path, speed)
            
            return output_path
            
        except Exception as e:
            print(f"ElevenLabs TTS error: {e}")
            return None


class OpenAIProvider(TTSProvider):
    """OpenAI TTS Provider - Easy to use"""
    
    API_URL = "https://api.openai.com/v1/audio/speech"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def synthesize(self, text: str, voice: str = "nova", speed: float = 1.0) -> Optional[str]:
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "speed": speed
        }
        
        try:
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            output_path = tempfile.mktemp(suffix=".mp3")
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return output_path
            
        except Exception as e:
            print(f"OpenAI TTS error: {e}")
            return None


def get_tts_provider(provider_name: str, api_key: str) -> TTSProvider:
    """
    Factory function để lấy TTS provider
    
    Args:
        provider_name: 'fpt', 'elevenlabs', hoặc 'openai'
        api_key: API key tương ứng
    
    Returns:
        TTSProvider instance
    """
    providers = {
        "fpt": FPTProvider,
        "elevenlabs": ElevenLabsProvider,
        "openai": OpenAIProvider
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unknown TTS provider: {provider_name}")
    
    return providers[provider_name](api_key)


def adjust_audio_speed(audio_path: str, speed: float) -> str:
    """
    Điều chỉnh tốc độ audio bằng thuật toán chất lượng cao (giữ pitch)
    
    Args:
        audio_path: Đường dẫn file audio
        speed: Tốc độ mới (1.0 = không đổi, >1.0 = nhanh hơn)
    
    Returns:
        Đường dẫn file audio mới
    """
    output_path = tempfile.mktemp(suffix=".mp3")
    
    # Sử dụng FFmpeg atempo filter để giữ pitch khi thay đổi tốc độ
    # atempo chỉ hỗ trợ từ 0.5 đến 2.0. Nếu speed > 2.0, cần chain filter
    atempo_filter = f"atempo={speed}"
    if speed > 2.0:
        atempo_filter = f"atempo=2.0,atempo={speed/2.0}"
        
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-filter:a", atempo_filter,
        "-vn",
        output_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        # Cleanup old file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return output_path
    except Exception as e:
        print(f"Error adjusting speed: {e}")
        return audio_path


def get_audio_duration(audio_path: str) -> float:
    """Lấy duration của audio file (seconds)"""
    try:
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0
    except:
        return 0.0


def fit_audio_to_duration(
    audio_path: str,
    target_duration: float,
    max_speed: float = 1.5  # Giới hạn max speed 1.5 theo yêu cầu
) -> str:
    """
    Điều chỉnh audio để fit vào duration mục tiêu một cách triệt để
    Chỉ tăng tốc khi cần thiết (audio dài hơn target)
    """
    current_duration = get_audio_duration(audio_path)
    
    # Nếu audio đã ngắn hơn target, không cần làm gì (giữ tốc độ gốc)
    if current_duration <= target_duration:
        return audio_path
    
    # Tính speed cần thiết + buffer 15% để đảm bảo không bị lấn
    required_speed = (current_duration / target_duration) * 1.15
    
    # Áp dụng speed up
    actual_speed = min(required_speed, max_speed)
    new_audio_path = adjust_audio_speed(audio_path, actual_speed)
    
    # Kiểm tra lại duration sau khi speed up
    new_duration = get_audio_duration(new_audio_path)
    
    # Nếu vẫn còn dài hơn target (do chạm trần max_speed), cần biện pháp mạnh hơn
    if new_duration > target_duration:
        try:
            # Cắt bớt phần dư thừa (thường là silence cuối câu)
            # Hoặc force cắt nếu chênh lệch không quá lớn (< 0.5s)
            diff = new_duration - target_duration
            if diff < 1.0: # Chấp nhận cắt bớt 1s cuối nếu cần thiết
                final_path = tempfile.mktemp(suffix=".mp3")
                cmd = [
                    "ffmpeg", "-y",
                    "-i", new_audio_path,
                    "-t", str(target_duration), # Force duration
                    "-c", "copy",
                    final_path
                ]
                subprocess.run(cmd, capture_output=True, check=True)
                if os.path.exists(new_audio_path):
                    os.remove(new_audio_path)
                return final_path
        except Exception as e:
            print(f"Error trimming audio: {e}")
            
    return new_audio_path


import subprocess

def generate_all_audio(
    segments: List[Dict],
    provider_name: str,
    api_key: str,
    voice: str,
    speed: float = 1.0,
    fit_duration: bool = True,
    progress_callback=None
) -> List[Dict]:
    """
    Generate audio cho tất cả segments
    """
    provider = get_tts_provider(provider_name, api_key)
    
    for i, seg in enumerate(segments):
        if progress_callback:
            progress_callback(f"Generating audio {i+1}/{len(segments)}...")
        
        text = seg.get("vietnamese") or seg.get("text", "")
        
        if not text:
            continue
        
        try:
            # Generate audio
            audio_path = provider.synthesize(text, voice, speed)
            
            if audio_path and fit_duration:
                # Tính khoảng trống cho phép
                # Nếu có segment tiếp theo, duration = start_next - start_current
                # Nếu là segment cuối, duration = end_current - start_current
                
                start_time = seg["start"]
                if i < len(segments) - 1:
                    next_start = segments[i+1]["start"]
                    # Trừ đi 0.1s làm buffer an toàn để tránh dính nhau
                    available_duration = max(0.5, next_start - start_time - 0.1) 
                else:
                    available_duration = seg["end"] - start_time
                
                audio_path = fit_audio_to_duration(audio_path, available_duration)
            
            seg["audio_path"] = audio_path or ""
            
        except Exception as e:
            print(f"Error generating audio for segment {seg['id']}: {e}")
            seg["audio_path"] = ""
    
    return segments

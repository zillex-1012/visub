"""
VietDub Solo - File Utilities
"""

import os
import tempfile
import shutil
from typing import Optional
import yt_dlp


TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")


def ensure_temp_dir():
    """Đảm bảo thư mục temp tồn tại"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    return TEMP_DIR


def download_youtube(url: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Download video từ YouTube
    
    Args:
        url: YouTube URL
        output_path: Output path (optional)
    
    Returns:
        Đường dẫn file video hoặc None nếu lỗi
    """
    ensure_temp_dir()
    
    if output_path is None:
        output_path = os.path.join(TEMP_DIR, "%(title)s.%(ext)s")
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'merge_output_format': 'mp4',
        # Anti-bot options
        'nocheckcertificate': True,
        'ignoreerrors': False,
        'logtostderr': False,
        'quiet': True,
        'no_warnings': True,
        'default_search': 'auto',
        'source_address': '0.0.0.0',
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            # Get actual filename
            if 'requested_downloads' in info:
                return info['requested_downloads'][0]['filepath']
            
            # Fallback
            filename = ydl.prepare_filename(info)
            return filename
            
    except Exception as e:
        print(f"YouTube download error: {e}")
        return None


def get_video_info(video_path: str) -> dict:
    """
    Lấy thông tin video
    
    Args:
        video_path: Đường dẫn video
    
    Returns:
        Dict với thông tin video
    """
    from moviepy.editor import VideoFileClip
    
    try:
        video = VideoFileClip(video_path)
        info = {
            "duration": video.duration,
            "fps": video.fps,
            "size": video.size,
            "width": video.size[0],
            "height": video.size[1]
        }
        video.close()
        return info
    except Exception as e:
        print(f"Error getting video info: {e}")
        return {}


def cleanup_temp_files():
    """Dọn dẹp thư mục temp"""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR)


def save_uploaded_file(uploaded_file) -> Optional[str]:
    """
    Lưu file được upload từ Streamlit
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        Đường dẫn file đã lưu
    """
    ensure_temp_dir()
    
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def is_youtube_url(url: str) -> bool:
    """Kiểm tra có phải YouTube URL không"""
    youtube_patterns = [
        'youtube.com/watch',
        'youtu.be/',
        'youtube.com/shorts/'
    ]
    return any(pattern in url for pattern in youtube_patterns)

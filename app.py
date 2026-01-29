"""
VietDub - Main Streamlit Application
Công cụ dubbing video cá nhân với AI
"""

import streamlit as st
import pandas as pd
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    WHISPER_MODELS, TRANSLATION_MODELS, TTS_PROVIDERS,
    FPT_VOICES, ELEVENLABS_VOICES, OPENAI_VOICES, DEFAULTS
)
from core.transcriber import transcribe_video, format_timecode_range
from core.translator import translate_segments, estimate_cost
from core.tts import generate_all_audio, get_tts_provider
from core.merger import export_video, create_srt_file, check_ffmpeg_installed
from utils.file_utils import (
    save_uploaded_file, download_youtube, is_youtube_url,
    get_video_info, cleanup_temp_files, TEMP_DIR, ensure_temp_dir
)


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="VietDub",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    code {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Deep Space Theme Background */
    .stApp {
        background-color: #0F172A;
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%);
        color: #E2E8F0;
    }

    /* Main Header */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #A5B4FC 0%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        text-shadow: 0 0 30px rgba(192, 132, 252, 0.3);
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #94A3B8;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 3rem;
    }

    /* Step Indicator - Wizard Style */
    .step-indicator-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 3rem;
        position: relative;
    }
    
    .step-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        z-index: 2;
        width: 120px;
    }
    
    .step-circle {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: #1E293B;
        border: 2px solid #334155;
        display: flex;
        justify-content: center;
        align-items: center;
        font-weight: 600;
        color: #64748B;
        transition: all 0.3s ease;
        margin-bottom: 0.5rem;
    }
    
    .step-item.active .step-circle {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        border-color: #8B5CF6;
        color: white;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.5);
    }
    
    .step-item.completed .step-circle {
        background: #10B981;
        border-color: #10B981;
        color: white;
    }
    
    .step-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .step-item.active .step-label {
        color: #E2E8F0;
        font-weight: 700;
    }
    
    .step-line {
        height: 2px;
        background: #334155;
        flex-grow: 1;
        max-width: 100px;
        margin: 0 -30px 25px -30px;
        z-index: 1;
    }
    
    .step-line.active {
        background: linear-gradient(90deg, #10B981 0%, #6366F1 100%);
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        margin-bottom: 2rem;
    }

    /* Upload Zone */
    .upload-zone {
        border: 2px dashed #475569;
        border-radius: 20px;
        padding: 4rem 2rem;
        text-align: center;
        background: rgba(15, 23, 42, 0.5);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    
    .upload-zone:hover {
        border-color: #8B5CF6;
        background: rgba(139, 92, 246, 0.05);
        transform: translateY(-2px);
    }

    /* Buttons */
    .stButton > button {
        background: #1E293B;
        color: #E2E8F0;
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        border-color: #8B5CF6;
        color: #8B5CF6;
        background: #0F172A;
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Primary Action Buttons (e.g., Translate All, Export) */
    div[data-testid="stColumn"] > div > div > div > div > div > button {
        /* Generic selector backup, specifics handled by Python layout */
    }

    /* UI Components Overrides */
    
    /* Input Fields */
    .stTextInput > div > div > input {
        background-color: #1E293B;
        border-color: #475569;
        color: #E2E8F0;
        border-radius: 10px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #8B5CF6;
        box-shadow: 0 0 0 1px #8B5CF6;
    }

    /* Select Box */
    .stSelectbox > div > div > div {
        background-color: #1E293B;
        border-color: #475569;
        color: #E2E8F0;
        border-radius: 10px;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1E293B;
        border-radius: 10px;
        border: 1px solid #334155;
    }

    /* Data Editor */
    div[data-testid="stDataEditor"] {
        border-radius: 15px;
        border: 1px solid #334155;
        overflow: hidden;
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366F1 0%, #EC4899 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0B1120;
        border-right: 1px solid #1E293B;
    }
    
    /* Hide Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================
# SESSION STATE INITIALIZATION
# ============================================

def init_session_state():
    """Khởi tạo session state"""
    defaults = {
        'current_step': 1,
        'video_path': None,
        'audio_path': None,
        'segments': [],
        'video_info': {},
        'processing': False,
        'openrouter_key': '',
        'tts_key': '',
        'whisper_model': DEFAULTS['whisper_model'],
        'translation_model': DEFAULTS['translation_model'],
        'tts_provider': DEFAULTS['tts_provider'],
        'voice': DEFAULTS['voice'],
        'speed': DEFAULTS['speed'],
        'original_volume': DEFAULTS['original_volume'],
        'dubbed_volume': DEFAULTS.get('dubbed_volume', 1.0)
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ============================================
# SIDEBAR - SETTINGS
# ============================================

def render_sidebar():
    """Render sidebar với settings"""
    with st.sidebar:
        st.markdown("## ⚙️ Cài đặt")
        
        # API Keys
        with st.expander("🔑 API Keys", expanded=True):
            st.session_state.openrouter_key = st.text_input(
                "OpenRouter API Key",
                value=st.session_state.openrouter_key,
                type="password",
                help="Dùng để dịch thuật. Lấy key tại openrouter.ai"
            )
            
            st.session_state.tts_provider = st.selectbox(
                "TTS Provider",
                options=list(TTS_PROVIDERS.keys()),
                format_func=lambda x: TTS_PROVIDERS[x],
                index=list(TTS_PROVIDERS.keys()).index(st.session_state.tts_provider)
            )
            
            tts_label = {
                'fpt': 'FPT.AI API Key',
                'elevenlabs': 'ElevenLabs API Key',
                'openai': 'OpenAI API Key'
            }
            
            st.session_state.tts_key = st.text_input(
                tts_label.get(st.session_state.tts_provider, 'TTS API Key'),
                value=st.session_state.tts_key,
                type="password"
            )
        
        # Model Settings
        with st.expander("🧠 Model Settings", expanded=False):
            st.session_state.whisper_model = st.selectbox(
                "Whisper Model",
                options=list(WHISPER_MODELS.keys()),
                format_func=lambda x: WHISPER_MODELS[x],
                index=list(WHISPER_MODELS.keys()).index(st.session_state.whisper_model),
                help="Model lớn hơn = chính xác hơn nhưng chậm hơn"
            )
            
            st.session_state.translation_model = st.selectbox(
                "Translation Model",
                options=list(TRANSLATION_MODELS.keys()),
                format_func=lambda x: TRANSLATION_MODELS[x],
                index=list(TRANSLATION_MODELS.keys()).index(st.session_state.translation_model)
            )
        
        # Voice Settings
        with st.expander("🎤 Voice Settings", expanded=False):
            # Get voices based on provider
            voices = {
                'fpt': FPT_VOICES,
                'elevenlabs': ELEVENLABS_VOICES,
                'openai': OPENAI_VOICES
            }.get(st.session_state.tts_provider, FPT_VOICES)
            
            st.session_state.voice = st.selectbox(
                "Giọng đọc",
                options=list(voices.keys()),
                format_func=lambda x: voices[x]
            )
            
            st.session_state.speed = st.slider(
                "Tốc độ đọc",
                min_value=0.8,
                max_value=1.3,
                value=st.session_state.speed,
                step=0.1
            )
        
        # Audio Mixing
        with st.expander("🔊 Audio Mixing", expanded=False):
            st.session_state.original_volume = st.slider(
                "Volume tiếng gốc",
                min_value=0.0,
                max_value=0.3,
                value=st.session_state.original_volume,
                step=0.05,
                help="Để nhỏ để làm nền"
            )
            
            st.session_state.dubbed_volume = st.slider(
                "Volume giọng lồng",
                min_value=0.5,
                max_value=1.0,
                value=st.session_state.dubbed_volume,
                step=0.1
            )
        
        # System Info
        st.markdown("---")
        st.markdown("### 📊 Trạng thái")
        
        # Check FFmpeg
        if check_ffmpeg_installed():
            st.success("✅ FFmpeg: Đã cài đặt")
        else:
            st.error("❌ FFmpeg: Chưa cài đặt")
        
        # Show current video info
        if st.session_state.video_path:
            info = st.session_state.video_info
            if info:
                st.info(f"📹 Duration: {info.get('duration', 0):.1f}s")


# ============================================
# STEP 1: INPUT & TRANSCRIBE
# ============================================

def render_step1():
    """Step 1: Upload video và transcribe"""
    st.markdown("### 📥 Bước 1: Nhập Video")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Upload zone
        st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Kéo thả video vào đây",
            type=['mp4', 'mkv', 'avi', 'mov', 'webm'],
            help="Hỗ trợ MP4, MKV, AVI, MOV, WebM"
        )
        
        st.markdown("**hoặc**")
        
        youtube_url = st.text_input(
            "Dán YouTube URL",
            placeholder="https://www.youtube.com/watch?v=..."
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ⚡ Quick Settings")
        
        # Quick model selection
        quick_model = st.radio(
            "Whisper Model",
            options=['base', 'small', 'large-v3'],
            format_func=lambda x: {
                'base': '🚀 Base (Nhanh)',
                'small': '⚖️ Small (Cân bằng)',
                'large-v3': '🎯 Large (Chính xác)'
            }[x],
            horizontal=False
        )
        st.session_state.whisper_model = quick_model
    
    # Process button
    st.markdown("---")
    
    can_process = uploaded_file is not None or (youtube_url and is_youtube_url(youtube_url))
    
    if st.button("🎯 Analyze & Transcribe", disabled=not can_process, use_container_width=True):
        with st.status("Đang xử lý...", expanded=True) as status:
            try:
                # Handle upload or download
                if uploaded_file:
                    st.write("📁 Đang lưu file...")
                    st.session_state.video_path = save_uploaded_file(uploaded_file)
                elif youtube_url:
                    st.write("📥 Đang tải video từ YouTube...")
                    st.session_state.video_path = download_youtube(youtube_url)
                
                if not st.session_state.video_path:
                    st.error("Không thể xử lý video!")
                    return
                
                # Get video info
                st.write("📊 Đang phân tích video...")
                st.session_state.video_info = get_video_info(st.session_state.video_path)
                
                # Transcribe
                st.write(f"🎙️ Đang transcribe với {st.session_state.whisper_model}...")
                st.session_state.segments = transcribe_video(
                    st.session_state.video_path,
                    model_name=st.session_state.whisper_model,
                    language='en'
                )
                
                status.update(label=f"✅ Hoàn tất! Tìm thấy {len(st.session_state.segments)} đoạn.", state="complete")
                
                # Move to step 2
                st.session_state.current_step = 2
                st.rerun()
                
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")


# ============================================
# STEP 2: EDITOR (GOD MODE)
# ============================================

def render_step2():
    """Step 2: Editor - Dịch và chỉnh sửa"""
    st.markdown("### ✏️ Bước 2: Chỉnh sửa & Dịch thuật")
    
    if not st.session_state.segments:
        st.warning("Chưa có dữ liệu transcription. Quay lại Bước 1.")
        if st.button("⬅️ Quay lại Bước 1"):
            st.session_state.current_step = 1
            st.rerun()
        return
    
    # Action buttons
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🛠️ Công cụ")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🌐 Translate All", use_container_width=True):
            if not st.session_state.openrouter_key:
                st.error("Vui lòng nhập OpenRouter API Key trong Sidebar!")
            else:
                with st.spinner("Đang dịch..."):
                    try:
                        # Estimate cost
                        cost = estimate_cost(
                            st.session_state.segments,
                            st.session_state.translation_model
                        )
                        st.info(f"💰 Chi phí ước tính: ${cost:.4f}")
                        
                        # Translate
                        st.session_state.segments = translate_segments(
                            st.session_state.segments,
                            st.session_state.openrouter_key,
                            st.session_state.translation_model
                        )
                        st.success("✅ Dịch xong!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Lỗi dịch: {str(e)}")
    
    with col2:
        if st.button("🎵 Generate All Audio", use_container_width=True):
            if not st.session_state.tts_key:
                st.error("Vui lòng nhập TTS API Key trong Sidebar!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    total = len(st.session_state.segments)
                    for i, seg in enumerate(st.session_state.segments):
                        status_text.text(f"Generating audio {i+1}/{total}...")
                        progress_bar.progress((i + 1) / total)
                        
                        # Generate audio for this segment
                        provider = get_tts_provider(
                            st.session_state.tts_provider,
                            st.session_state.tts_key
                        )
                        text = seg.get("vietnamese") or seg.get("text", "")
                        if text:
                            audio_path = provider.synthesize(
                                text,
                                st.session_state.voice,
                                st.session_state.speed
                            )
                            seg["audio_path"] = audio_path or ""
                    
                    st.success("✅ Generate audio xong!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi TTS: {str(e)}")
    
    with col3:
        if st.button("⬅️ Quay lại Bước 1", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()
    
    with col4:
        # Check if ready for step 3
        has_translations = any(seg.get("vietnamese") for seg in st.session_state.segments)
        has_audio = any(seg.get("audio_path") for seg in st.session_state.segments)
        
        if st.button("➡️ Tiếp tục Export", disabled=not (has_translations and has_audio), use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Create DataFrame for editing
    df_data = []
    for seg in st.session_state.segments:
        df_data.append({
            "ID": seg["id"],
            "Timecode": format_timecode_range(seg["start"], seg["end"]),
            "English": seg["text"],
            "Vietnamese": seg.get("vietnamese", ""),
            "Has Audio": "✅" if seg.get("audio_path") else "❌"
        })
    
    df = pd.DataFrame(df_data)
    
    # Editable dataframe
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "ID": st.column_config.NumberColumn("ID", disabled=True, width="small"),
            "Timecode": st.column_config.TextColumn("⏱️ Timecode", disabled=True, width="medium"),
            "English": st.column_config.TextColumn("🇬🇧 English", disabled=True),
            "Vietnamese": st.column_config.TextColumn("🇻🇳 Vietnamese", width="large"),
            "Has Audio": st.column_config.TextColumn("🎵", disabled=True, width="small")
        },
        hide_index=True,
        height=500
    )
    
    # Update segments from edited DataFrame
    for i, row in edited_df.iterrows():
        if i < len(st.session_state.segments):
            st.session_state.segments[i]["vietnamese"] = row["Vietnamese"]
    
    # Statistics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Tổng đoạn", len(st.session_state.segments))
    with col2:
        translated = sum(1 for seg in st.session_state.segments if seg.get("vietnamese"))
        st.metric("🌐 Đã dịch", f"{translated}/{len(st.session_state.segments)}")
    with col3:
        has_audio = sum(1 for seg in st.session_state.segments if seg.get("audio_path"))
        st.metric("🎵 Có audio", f"{has_audio}/{len(st.session_state.segments)}")


# ============================================
# STEP 3: PREVIEW & EXPORT
# ============================================

def render_step3():
    """Step 3: Preview và Export"""
    st.markdown("### 🎬 Bước 3: Preview & Export")
    
    if not st.session_state.segments:
        st.warning("Chưa có dữ liệu. Quay lại Bước 1.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📹 Video Preview")
        
        if st.session_state.video_path and os.path.exists(st.session_state.video_path):
            st.video(st.session_state.video_path)
        else:
            st.info("Video không khả dụng để preview")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Thống kê")
        
        # Stats
        total_segments = len(st.session_state.segments)
        translated = sum(1 for seg in st.session_state.segments if seg.get("vietnamese"))
        has_audio = sum(1 for seg in st.session_state.segments if seg.get("audio_path"))
        
        st.metric("Tổng đoạn", total_segments)
        st.metric("Đã dịch", f"{translated}/{total_segments}")
        st.metric("Có audio", f"{has_audio}/{total_segments}")
        
        if st.session_state.video_info:
            duration = st.session_state.video_info.get('duration', 0)
            st.metric("Duration", f"{duration:.1f}s")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        burn_subs = st.checkbox("Burn Subtitles vào video", value=True)
    
    with col2:
        output_format = st.selectbox("Format", ["MP4", "MKV"])
    
    with col3:
        export_srt_only = st.checkbox("Chỉ export SRT")
    
    # Export buttons
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🎬 Xuất File")
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("⬅️ Quay lại Bước 2", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()
    
    with col2:
        if st.button("👁️ Quick Preview 60s", use_container_width=True):
            if not check_ffmpeg_installed():
                st.error("FFmpeg chưa được cài đặt!")
                return
            
            with st.status("Đang tạo preview 60s...", expanded=True) as status:
                try:
                    ensure_temp_dir()
                    preview_path = os.path.join(TEMP_DIR, "preview_60s.mp4")
                    
                    st.write("🎬 Đang render 60 giây đầu tiên...")
                    
                    # Filter segments trong 60s đầu
                    preview_segments = [
                        seg for seg in st.session_state.segments 
                        if seg["start"] < 60
                    ]
                    
                    success = export_video(
                        st.session_state.video_path,
                        preview_segments,
                        preview_path,
                        original_volume=st.session_state.original_volume,
                        dubbed_volume=st.session_state.dubbed_volume,
                        burn_subtitles=burn_subs,
                        preview_duration=60  # Chỉ render 60s
                    )
                    
                    if success and os.path.exists(preview_path):
                        status.update(label="✅ Preview sẵn sàng!", state="complete")
                        st.session_state.preview_path = preview_path
                        st.rerun()
                    else:
                        st.error("Tạo preview thất bại.")
                        
                except Exception as e:
                    st.error(f"Lỗi preview: {str(e)}")
    
    with col3:
        if st.button("📄 Export SRT", use_container_width=True):
            try:
                ensure_temp_dir()
                srt_path = os.path.join(TEMP_DIR, "subtitles.srt")
                create_srt_file(st.session_state.segments, srt_path)
                
                with open(srt_path, 'r', encoding='utf-8') as f:
                    srt_content = f.read()
                
                st.download_button(
                    label="⬇️ Download SRT",
                    data=srt_content,
                    file_name="vietdub_subtitles.srt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Lỗi export SRT: {str(e)}")
    
    with col4:
        if st.button("🎬 Export Full Video", use_container_width=True, type="primary"):
            if not check_ffmpeg_installed():
                st.error("FFmpeg chưa được cài đặt. Vui lòng cài FFmpeg trước.")
                return
            
            with st.status("Đang render video...", expanded=True) as status:
                try:
                    ensure_temp_dir()
                    # Determine extension based on format
                    ext = output_format.lower()
                    output_filename = f"vietdub_output.{ext}"
                    output_path = os.path.join(TEMP_DIR, output_filename)
                    
                    st.write("📹 Đang ghép audio...")
                    
                    success = export_video(
                        st.session_state.video_path,
                        st.session_state.segments,
                        output_path,
                        original_volume=st.session_state.original_volume,
                        dubbed_volume=st.session_state.dubbed_volume,
                        burn_subtitles=burn_subs
                    )
                    
                    if success and os.path.exists(output_path):
                        status.update(label="✅ Export thành công!", state="complete")
                        
                        # Determine mime type
                        mime_type = "video/mp4" if ext == "mp4" else "video/x-matroska"
                        
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label=f"⬇️ Download Video ({output_format})",
                                data=f.read(),
                                file_name=output_filename,
                                mime=mime_type
                            )
                    else:
                        st.error("Export thất bại. Kiểm tra FFmpeg và thử lại.")
                        
                except Exception as e:
                    st.error(f"Lỗi export: {str(e)}")
                    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Hiển thị preview nếu có
    if st.session_state.get('preview_path') and os.path.exists(st.session_state.preview_path):
        st.markdown("---")
        st.markdown("#### 🎬 Preview Video (60s đầu tiên)")
        st.video(st.session_state.preview_path)


# ============================================
# MAIN APP
# ============================================

def main():
    """Main application"""
    # Render sidebar
    render_sidebar()
    
    # Header
    st.markdown('<h1 class="main-header">🎬 VietDub</h1>', unsafe_allow_html=True)
    
    # Step indicator
    steps = ["Input", "Edit", "Export"]
    
    # Generate HTML for step indicator
    html_steps = '<div class="step-indicator-container">'
    
    for i, step in enumerate(steps, 1):
        # Determine status
        if i < st.session_state.current_step:
            status_class = "completed"
            icon = "✓"
        elif i == st.session_state.current_step:
            status_class = "active"
            icon = str(i)
        else:
            status_class = ""
            icon = str(i)
            
        # Add connection line (except for first item)
        if i > 1:
            line_class = "active" if i <= st.session_state.current_step else ""
            html_steps += f'<div class="step-line {line_class}"></div>'
            
        # Add step item
        html_steps += f"""
        <div class="step-item {status_class}">
            <div class="step-circle">{icon}</div>
            <div class="step-label">{step}</div>
        </div>
        """
        
    html_steps += '</div>'
    st.markdown(html_steps, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render current step
    if st.session_state.current_step == 1:
        render_step1()
    elif st.session_state.current_step == 2:
        render_step2()
    elif st.session_state.current_step == 3:
        render_step3()


if __name__ == "__main__":
    main()

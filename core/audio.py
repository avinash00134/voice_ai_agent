import numpy as np
import sounddevice as sd
import whisper
from TTS.api import TTS
from typing import Optional, AsyncGenerator, Tuple
from pathlib import Path
import wave
import asyncio
import logging
import tempfile
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Configuration for audio processing"""
    sample_rate: int = settings.sample_rate
    channels: int = settings.channels
    chunk_size: int = settings.chunk_size
    dtype: str = 'float32'
    silence_threshold: float = settings.silence_threshold
    max_duration: int = settings.max_recording_duration

class AudioProcessor:
    """
    Handles all audio processing including:
    - Real-time audio capture
    - Speech-to-text transcription
    - Text-to-speech synthesis
    - Audio file operations
    """
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.is_recording = False
        self.audio_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize models
        self._init_whisper()
        self._init_tts()
        
        # Audio stream state
        self.stream = None
        self.audio_buffer = np.array([], dtype=np.float32)
        
    def _init_whisper(self):
        """Initialize Whisper STT model"""
        try:
            logger.info(f"Loading Whisper model: {settings.whisper_model}")
            self.whisper_model = whisper.load_model(
                settings.whisper_model,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def _init_tts(self):
        """Initialize TTS model"""
        try:
            logger.info(f"Loading TTS model: {settings.tts_model}")
            self.tts = TTS(
                model_name=settings.tts_model,
                progress_bar=False,
                gpu=torch.cuda.is_available()
            )
            logger.info("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise
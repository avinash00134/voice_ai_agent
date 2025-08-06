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
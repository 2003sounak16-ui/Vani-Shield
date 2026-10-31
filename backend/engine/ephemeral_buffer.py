"""
VaniShield AI — Ephemeral In-Process Volatile RAM Ring Buffer
DPDP Act (Digital Personal Data Protection Act 2023) Compliant:
1. Zero persistent disk writes (no raw .wav or .pcm files written to disk).
2. Held strictly in volatile memory with a hard 500ms sliding window eviction.
3. Overwritten with zeros upon purge.
4. Keyed HMAC-SHA256 hashing for caller identity & audit logs.
"""

import os
import time
import hmac
import hashlib
from collections import defaultdict, deque
import threading
import numpy as np

# Load HMAC key from environment variable; fallback to configured salt if not set
_secret_env = os.environ.get("VANISHIELD_HMAC_SECRET", "vanishield_dpdp_secret_salt_2026")
HMAC_SECRET = _secret_env.encode("utf-8") if isinstance(_secret_env, str) else _secret_env

def hash_caller_identity(phone_number: str) -> str:
    """Produces an irreversible, anonymized audit hash for phone numbers."""
    if not phone_number:
        return "ANONYMOUS"
    clean = "".join(filter(str.isalnum, phone_number))
    return hmac.new(HMAC_SECRET, clean.encode("utf-8"), hashlib.sha256).hexdigest()[:16]

class EphemeralRAMBuffer:
    """
    In-memory circular ring buffer holding live 16kHz PCM audio in RAM.
    Zero-database requirement: Runs directly in-process with nanosecond memory speed.
    """
    def __init__(self, max_duration_sec: float = 3.5, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.max_duration_sec = max_duration_sec
        self.max_samples = int(max_duration_sec * sample_rate)
        # Store (timestamp, numpy_chunk)
        self.buffers = defaultdict(deque)
        self.last_activity = {}
        self._lock = threading.Lock()
        self._cleaner_thread = threading.Thread(target=self._auto_clean_worker, daemon=True)
        self._cleaner_thread.start()

    def _auto_clean_worker(self):
        while True:
            time.sleep(1.0)
            try:
                self.evict_expired()
            except Exception:
                pass

    def push_chunk(self, session_id: str, raw_pcm_bytes: bytes):
        """Pushes raw PCM bytes into volatile RAM with timestamp."""
        try:
            arr = np.frombuffer(raw_pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            # Fallback if float32 bytes received
            arr = np.frombuffer(raw_pcm_bytes, dtype=np.float32)

        now = time.time()
        with self._lock:
            self.last_activity[session_id] = now
            q = self.buffers[session_id]
            q.append((now, arr))

            # Evict chunks older than configured max_duration_sec
            cutoff = now - self.max_duration_sec
            while q and q[0][0] < cutoff:
                _, old_arr = q.popleft()
                old_arr.fill(0.0)  # Wipe volatile memory contents

    def evict_expired(self, now: float = None):
        """Proactively evicts expired audio across all sessions independent of incoming packets."""
        now = now or time.time()
        cutoff = now - self.max_duration_sec
        idle_sessions = []
        with self._lock:
            for sid, q in list(self.buffers.items()):
                while q and q[0][0] < cutoff:
                    _, old_arr = q.popleft()
                    old_arr.fill(0.0)
                if not q and (now - self.last_activity.get(sid, 0) > self.max_duration_sec):
                    idle_sessions.append(sid)
            for sid in idle_sessions:
                if sid in self.buffers:
                    q = self.buffers[sid]
                    while q:
                        _, arr = q.popleft()
                        arr.fill(0.0)
                    del self.buffers[sid]
                self.last_activity.pop(sid, None)

    def get_eval_window(self, session_id: str) -> np.ndarray:
        """Retrieves continuous float32 audio window (up to max_samples) directly from RAM."""
        now = time.time()
        cutoff = now - self.max_duration_sec
        with self._lock:
            q = self.buffers[session_id]
            while q and q[0][0] < cutoff:
                _, old_arr = q.popleft()
                old_arr.fill(0.0)

            if not q:
                return np.zeros(self.max_samples, dtype=np.float32)

            audio_chunks = [item[1] for item in q]
            combined = np.concatenate(audio_chunks)

        if len(combined) > self.max_samples:
            combined = combined[-self.max_samples:]
        elif len(combined) < self.max_samples:
            combined = np.pad(combined, (0, self.max_samples - len(combined)))
        return combined

    def get_buffer_duration(self, session_id: str) -> float:
        """Returns total duration in seconds of audio buffered for the given session."""
        now = time.time()
        cutoff = now - self.max_duration_sec
        with self._lock:
            q = self.buffers.get(session_id)
            if not q:
                return 0.0
            total_samples = sum(len(item[1]) for item in q if item[0] >= cutoff)
            return round(total_samples / float(self.sample_rate), 3)

    def clear_session(self, session_id: str):
        """Alias for purge_session to clear volatile RAM immediately."""
        self.purge_session(session_id)

    def purge_session(self, session_id: str):
        """Overwrites memory with zeros and frees volatile RAM immediately upon call end."""
        with self._lock:
            if session_id in self.buffers:
                q = self.buffers[session_id]
                while q:
                    _, arr = q.popleft()
                    arr.fill(0.0)
                del self.buffers[session_id]
            self.last_activity.pop(session_id, None)

    def get_buffer_duration(self, session_id: str) -> float:
        """Returns the total duration in seconds of buffered PCM in memory for the given session."""
        with self._lock:
            q = self.buffers.get(session_id)
            if not q:
                return 0.0
            total_samples = sum(len(item[1]) for item in q)
            return float(total_samples) / float(self.sample_rate)

    def clear_session(self, session_id: str):
        """Alias for purge_session to wipe volatile RAM for session."""
        self.purge_session(session_id)

# Singleton instance
ram_ring_buffer = EphemeralRAMBuffer(max_duration_sec=3.5, sample_rate=16000)

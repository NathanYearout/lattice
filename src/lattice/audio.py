"""A tiny NumPy mixer on one sounddevice stream (any number of voices, live volume, loops), and a microphone recorder."""

import threading

import numpy as np

RATE = 44100
FILES = (".wav", ".ogg", ".flac", ".mp3", ".aiff")


class Voice:
    def __init__(self, data, volume, loop, preview=False):
        self.data, self.volume, self.loop, self.at, self.playing, self.preview = data, volume, loop, 0, True, preview

    def stop(self):
        self.playing = False


class Mixer:
    def __init__(self):
        self.voices, self.keyed, self.lock, self.stream, self.paused, self.enabled = [], {}, threading.Lock(), None, False, True

    def play(self, data, volume=1.0, loop=False, key=None, preview=False):
        """``preview`` voices (editor auditioning) keep playing while the game is paused."""
        v = Voice(data, volume, loop, preview)
        if self.enabled and self.stream is None:
            try:
                import sounddevice as sd

                self.stream = sd.OutputStream(RATE, channels=2, dtype="float32", callback=self._fill)
                self.stream.start()
            except Exception as e:  # no audio device: games still run, silently
                self.enabled = False
                print(f"[lattice] audio disabled: {e}")
        with self.lock:
            self.voices.append(v) if self.enabled else None
            self.keyed[key] = v
        return v

    def stop_all(self, previews=True):
        with self.lock:
            stopping = [v for v in self.voices if previews or not v.preview]
            [v.stop() for v in stopping]
            self.voices = [v for v in self.voices if v.playing]
            self.keyed = {k: v for k, v in self.keyed.items() if v.playing}

    def _fill(self, out, frames, *_):
        out.fill(0)
        with self.lock:
            for v in self.voices:
                if self.paused and not v.preview:
                    continue
                idx = np.arange(v.at, v.at + frames) % len(v.data) if v.loop else np.arange(v.at, min(v.at + frames, len(v.data)))
                out[: len(idx)] += v.data[idx] * v.volume
                v.at += frames
                v.playing &= v.loop or v.at < len(v.data)
            self.voices = [v for v in self.voices if v.playing]
        np.clip(out, -1, 1, out)


mixer = Mixer()


class Recorder:
    """Records the default microphone. ``stop(path)`` trims the silence at both ends, normalises and writes the file."""

    def __init__(self):
        self.stream, self.chunks, self.level = None, [], 0.0

    @property
    def recording(self):
        return self.stream is not None

    @property
    def seconds(self):
        return sum(map(len, self.chunks)) / RATE

    def start(self):
        import sounddevice as sd

        self.chunks, self.level = [], 0.0
        self.stream = sd.InputStream(RATE, channels=1, dtype="float32", callback=self._take)
        self.stream.start()

    def _take(self, data, *_):
        self.chunks.append(data[:, 0].copy())
        self.level = max(float(np.abs(data).max()), self.level * 0.85)  # a meter that falls back smoothly

    def stop(self, path):
        import soundfile

        self.stream.stop(), self.stream.close()
        self.stream, data = None, np.concatenate(self.chunks) if self.chunks else np.zeros(1, np.float32)
        loud = np.flatnonzero(np.abs(data) > max(0.02, np.abs(data).max() * 0.05))
        pad = int(0.05 * RATE)
        data = data[max(loud[0] - pad, 0): loud[-1] + pad] if len(loud) else data
        path.parent.mkdir(parents=True, exist_ok=True)
        peak = float(np.abs(data).max())
        soundfile.write(str(path), data * (0.9 / peak if peak > 0.02 else 1.0), RATE)  # never amplify a silent take's hiss
        return path


def load(path):
    """Decode any file libsndfile reads (wav, ogg, flac, mp3) into stereo float32 at the mixer rate."""
    import soundfile

    data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return as_stereo(data, rate)


def as_stereo(data, rate=RATE):
    data = np.asarray(data, np.float32).reshape(len(data), -1)
    if rate != RATE:  # linear resample: plenty for game sfx
        t = np.arange(int(len(data) * RATE / rate)) * rate / RATE
        data = np.stack([np.interp(t, np.arange(len(data)), c) for c in data.T], -1).astype(np.float32)
    return np.ascontiguousarray(np.broadcast_to(data, (len(data), 2)) if data.shape[1] == 1 else data[:, :2])

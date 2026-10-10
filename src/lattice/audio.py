"""A tiny NumPy mixer on one sounddevice stream: any number of voices, live volume, loops."""

import threading

import numpy as np

RATE = 44100


class Voice:
    def __init__(self, data, volume, loop):
        self.data, self.volume, self.loop, self.at, self.playing = data, volume, loop, 0, True

    def stop(self):
        self.playing = False


class Mixer:
    def __init__(self):
        self.voices, self.keyed, self.lock, self.stream, self.paused, self.enabled = [], {}, threading.Lock(), None, False, True

    def play(self, data, volume=1.0, loop=False, key=None):
        v = Voice(data, volume, loop)
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

    def stop_all(self):
        with self.lock:
            [v.stop() for v in self.voices]
            self.voices.clear(), self.keyed.clear()

    def _fill(self, out, frames, *_):
        out.fill(0)
        if self.paused:
            return
        with self.lock:
            for v in self.voices:
                idx = np.arange(v.at, v.at + frames) % len(v.data) if v.loop else np.arange(v.at, min(v.at + frames, len(v.data)))
                out[: len(idx)] += v.data[idx] * v.volume
                v.at += frames
                v.playing &= v.loop or v.at < len(v.data)
            self.voices = [v for v in self.voices if v.playing]
        np.clip(out, -1, 1, out)


mixer = Mixer()


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

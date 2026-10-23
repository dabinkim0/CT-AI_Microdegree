#!/usr/bin/env python
"""
Offline file-based piano transcription with the online_amt model (Kwon/Jeong/Nam,
ISMIR 2020 "autoregressive multi-state note model"; predecessor of PARpiano).

The PARpiano GitHub repo contains NO code and NO weights (it is only a GitHub-Pages
site), so this script wraps the predecessor's released checkpoint
(online_amt/model-180000.pt, trained on MAESTRO v2).

The model is causal/autoregressive, so by default we *simulate the real-time loop*
exactly as run_on_plt.py does: feed 512-sample (32 ms @ 16 kHz) chunks one by one
into OnlineTranscriber and record the per-frame 88x5-state output.  The reported
processing speed therefore reflects what a live microphone loop would achieve.

Outputs (written to --out_dir, default piano/outputs/<stem>/):
  amt_notes.json   {"meta": {...}, "notes": [{onset, offset, pitch, velocity}, ...]}
  amt.mid          MIDI file (pretty_midi), one piano track
  amt_roll.npy     uint8 (T, 88): 1 = note active (states 2/3/4), 0 otherwise
  amt_states.npy   uint8 (T, 88): raw model state per frame
                   0=off, 1=offset, 2=sustain(on), 3=onset, 4=re-onset
  amt_timing.json  speed / latency measurements
  amt_roll.png     quick-look piano roll image
Frame rate: 16000 / 512 = 31.25 fps (32 ms hop).  In online mode, array frame k was
emitted after consuming (k+1) chunks and describes audio time (k+1)*0.032 - 0.160 s
(= k*0.032 - 0.128 s); the note JSON already applies this.  In offline mode frame k
is audio time k*0.032 s.

Usage:
  CUDA_VISIBLE_DEVICES="" python run_piano_amt.py song.wav [--out_dir DIR]
      [--ref_midi gt.mid] [--mode online|offline] [--threads N] [--max_seconds S]
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only (GPU off-limits on this server)
warnings.filterwarnings("ignore")

import numpy as np
import torch

TOOLS_DIR = Path(__file__).resolve().parent
ONLINE_AMT_DIR = TOOLS_DIR / "online_amt"
sys.path.insert(0, str(ONLINE_AMT_DIR))

from transcribe import load_model, OnlineTranscriber  # noqa: E402
from autoregressive.constants import SAMPLE_RATE, HOP_LENGTH, MIN_MIDI  # noqa: E402

FPS = SAMPLE_RATE / HOP_LENGTH  # 31.25
# Online buffer = 5120 samples = 7 STFT frames (2048 win, 512 hop, no centre padding).
# The 3 conv layers (time padding 0) reduce 7 -> 1 frame: the emitted frame is the
# CENTRE frame of the buffer, whose window centre lies 2560 samples (160 ms) before
# the newest audio sample.  That is the model's inherent algorithmic delay.
INHERENT_DELAY_SAMPLES = 2560
INHERENT_DELAY_SEC = INHERENT_DELAY_SAMPLES / SAMPLE_RATE  # 0.16 s
STATE_NAMES = {0: "off", 1: "offset", 2: "sustain", 3: "onset", 4: "reonset"}


class RawStateTranscriber(OnlineTranscriber):
    """Same as OnlineTranscriber.inference but returns the raw 88 state ids."""

    def inference(self, audio):
        with torch.no_grad():
            self.update_buffer(audio)
            self.switch_on_or_off()
            if self.num_under_thr > self.patience:  # silence gate (as in the repo)
                return np.zeros(88, dtype=np.uint8)
            self.update_mel_buffer()
            acoustic_out = self.update_acoustic_out(self.mel_buffer.transpose(-1, -2))
            language_out, self.hidden = self.model.lm_model_step(acoustic_out, self.hidden, self.prev_output)
            language_out[0, 0, :, 3:5] *= 2  # repo heuristic: boost onset / re-onset
            self.prev_output = language_out.argmax(dim=3)
            return self.prev_output[0, 0, :].numpy().astype(np.uint8)


def load_audio(path, max_seconds=None):
    import librosa
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=max_seconds)
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    return y


def run_online(model, audio, silence_threshold=0.05, silence_patience=100):
    """Simulate the real-time loop: 512-sample chunks, one model step per chunk."""
    tr = RawStateTranscriber(model, return_roll=True)
    tr.inten_threshold = silence_threshold
    tr.patience = silence_patience
    n_chunks = len(audio) // HOP_LENGTH
    states = np.zeros((n_chunks, 88), dtype=np.uint8)
    step_times = np.zeros(n_chunks, dtype=np.float64)
    t0 = time.perf_counter()
    for k in range(n_chunks):
        ts = time.perf_counter()
        states[k] = tr.inference(audio[k * HOP_LENGTH:(k + 1) * HOP_LENGTH])
        step_times[k] = time.perf_counter() - ts
    total = time.perf_counter() - t0
    return states, total, step_times


def run_offline(model, audio):
    """Whole-file forward: batched CNN over the full mel, then the sequential AR LSTM
    loop (model.forward with gt_label=False).  Faster, but no longer a causal loop."""
    model.eval()
    # OnlineTranscriber.__init__ (used for warm-up / online mode) sets conv time-padding to 0;
    # restore symmetric padding so frames stay aligned, and centre-pad the audio by n_fft/2
    # because MelSpectrogram.forward uses librosa.stft(center=False).
    for i in (0, 3, 8):
        model.acoustic_model.cnn[i].padding = (1, 1)
    with torch.no_grad():
        t0 = time.perf_counter()
        y = torch.from_numpy(np.pad(audio, 1024, mode="reflect")).unsqueeze(0)
        mel = model.melspectrogram(y).transpose(-1, -2)  # (1, T, 229), frame k centred at k*512
        out = model(mel)  # (1, T, 88) argmax states
        total = time.perf_counter() - t0
    return out[0].numpy().astype(np.uint8), total, None


def states_to_notes(states, fps, velocity_proxy=None, delay_sec=0.0):
    """Decode (T, 88) state sequence into notes.
    A note starts on state 3 (onset) or 4 (re-onset), persists through 2/3/4 and ends
    at the first frame whose state is 0 or 1 (or at a re-onset, which starts a new note).
    Online mode: the state emitted at step k was computed after consuming (k+1) chunks, and
    describes the frame centred INHERENT_DELAY_SEC before the newest sample, so its audio
    time is (k+1)*hop - delay.  Offline mode (delay=0, centre-padded STFT): time = k*hop."""
    T = states.shape[0]
    notes = []
    for p in range(88):
        col = states[:, p]
        cur_on = None
        for k in range(T):
            s = col[k]
            if s in (3, 4):
                if cur_on is not None:
                    notes.append((cur_on, k, p))
                cur_on = k
            elif s == 2:
                if cur_on is None:  # sustain without onset (model skipped onset): start it
                    cur_on = k
            else:  # 0 or 1
                if cur_on is not None:
                    notes.append((cur_on, k, p))
                    cur_on = None
        if cur_on is not None:
            notes.append((cur_on, T, p))
    notes.sort()
    out = []
    for on, off, p in notes:
        if off - on < 1:
            continue
        vel = 64 if velocity_proxy is None else int(velocity_proxy[min(on, T - 1), p])
        shift = 1 if delay_sec > 0 else 0  # online: step k == (k+1) chunks consumed
        out.append({
            "onset": round(max(0.0, (on + shift) / fps - delay_sec), 4),
            "offset": round(max(0.0, (off + shift) / fps - delay_sec), 4),
            "pitch": int(p + MIN_MIDI),
            "velocity": int(np.clip(vel, 1, 127)),
            "onset_frame": int(on),
            "offset_frame": int(off),
        })
    return out


def velocity_proxy_from_audio(audio, n_frames):
    """The model has no velocity head.  Crude loudness proxy: frame RMS (dB) around the
    onset mapped to 1..127.  Documented as a heuristic, not a model output."""
    import librosa
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=HOP_LENGTH, center=True)[0]
    rms = rms[:n_frames] if len(rms) >= n_frames else np.pad(rms, (0, n_frames - len(rms)))
    db = 20 * np.log10(np.maximum(rms, 1e-6))
    # look ahead 3 frames (~100 ms) to catch the attack peak
    db_peak = np.array([db[k:k + 4].max() for k in range(n_frames)])
    vel = np.interp(db_peak, [-50, -10], [20, 120])
    return np.repeat(vel[:, None], 88, axis=1)


def write_midi(notes, path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="online_amt")
    for n in notes:
        if n["offset"] <= n["onset"]:
            continue
        inst.notes.append(pretty_midi.Note(velocity=n["velocity"], pitch=n["pitch"],
                                           start=n["onset"], end=n["offset"]))
    pm.instruments.append(inst)
    pm.write(str(path))


def evaluate(notes, ref_midi, max_seconds=None):
    import mir_eval
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(ref_midi))
    ref = [(n.start, n.end, n.pitch) for i in pm.instruments for n in i.notes]
    if max_seconds:
        ref = [r for r in ref if r[0] < max_seconds]
    ref_i = np.array([[s, e] for s, e, _ in ref])
    ref_p = np.array([mir_eval.util.midi_to_hz(p) for _, _, p in ref])
    est_i = np.array([[n["onset"], max(n["offset"], n["onset"] + 1e-3)] for n in notes])
    est_p = np.array([mir_eval.util.midi_to_hz(n["pitch"]) for n in notes])
    if len(est_i) == 0:
        return {"onset_f1": 0.0, "note_f1": 0.0, "n_ref": len(ref), "n_est": 0}
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p, offset_ratio=None, onset_tolerance=0.05)
    p2, r2, f2, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p, offset_ratio=0.2, onset_tolerance=0.05)
    return {"onset_precision": round(p, 4), "onset_recall": round(r, 4), "onset_f1": round(f, 4),
            "note_f1_with_offset": round(f2, 4), "n_ref": len(ref), "n_est": len(notes),
            "onset_tolerance_sec": 0.05}


def plot_roll(roll, fps, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = roll.shape[0]
    fig, ax = plt.subplots(figsize=(min(30, 4 + T / fps / 4), 4))
    ax.imshow(roll.T, aspect="auto", origin="lower", cmap="Greys", interpolation="nearest",
              extent=[0, T / fps, MIN_MIDI, MIN_MIDI + 88])
    ax.set_xlabel("time (s)"); ax.set_ylabel("MIDI pitch"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--out_dir", default=None, help="default: piano/outputs/<audio stem>/")
    ap.add_argument("--model", default=str(ONLINE_AMT_DIR / "model-180000.pt"))
    ap.add_argument("--mode", choices=["online", "offline"], default="online",
                    help="online = simulate 512-sample real-time loop (default); offline = whole-file forward")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads (a live loop should not hog all cores)")
    ap.add_argument("--max_seconds", type=float, default=None)
    ap.add_argument("--ref_midi", default=None, help="ground-truth MIDI (e.g. MAESTRO) for onset-F1 evaluation")
    ap.add_argument("--no-delay-compensation", action="store_true",
                    help="keep raw emission-time stamps instead of shifting by the 160 ms inherent delay")
    ap.add_argument("--silence_threshold", type=float, default=0.05,
                    help="repo silence gate: peak-to-peak amplitude below this for --silence_patience chunks "
                         "=> model is skipped and zeros emitted (0.05 ~ -32 dBFS; lower it for quiet/pp playing)")
    ap.add_argument("--silence_patience", type=int, default=100, help="chunks (32 ms each) before the gate engages")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    audio_path = Path(args.audio)
    out_dir = Path(args.out_dir) if args.out_dir else TOOLS_DIR.parent / "outputs" / audio_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[amt] loading model {args.model} (device=cpu, threads={args.threads})")
    model = load_model(args.model)
    n_params = sum(p.numel() for p in model.parameters())

    audio = load_audio(audio_path, args.max_seconds)
    dur = len(audio) / SAMPLE_RATE
    print(f"[amt] audio {audio_path.name}: {dur:.1f}s @ {SAMPLE_RATE} Hz, mode={args.mode}")

    # warm-up (first call pays lazy init costs)
    _ = RawStateTranscriber(model).inference(np.zeros(HOP_LENGTH, dtype=np.float32))

    if args.mode == "online":
        states, total, step_times = run_online(model, audio, args.silence_threshold, args.silence_patience)
        delay = 0.0 if args.no_delay_compensation else INHERENT_DELAY_SEC
    else:
        states, total, step_times = run_offline(model, audio)
        delay = 0.0  # centre-padded STFT => frames already aligned

    T = states.shape[0]
    roll = np.isin(states, (2, 3, 4)).astype(np.uint8)
    vel_proxy = velocity_proxy_from_audio(audio, T)
    notes = states_to_notes(states, FPS, vel_proxy, delay_sec=delay)

    rtf = total / dur
    timing = {
        "mode": args.mode, "device": "cpu", "torch_threads": args.threads,
        "torch_version": torch.__version__, "model_params": n_params,
        "audio_seconds": round(dur, 3), "n_frames": int(T), "fps": FPS, "hop_sec": HOP_LENGTH / SAMPLE_RATE,
        "processing_seconds": round(total, 3), "real_time_factor": round(rtf, 4),
        "speed_x_realtime": round(1 / rtf, 2), "frames_per_second_processed": round(T / total, 1),
        "inherent_model_delay_sec": INHERENT_DELAY_SEC,
        "inherent_model_delay_note": "emitted frame = centre of 7-frame (5120-sample) buffer -> 160 ms behind newest sample; "
                                     "plus one 32 ms chunk of buffering and the per-step compute time below",
        "delay_compensation_applied_sec": delay,
        "silence_gate": {"threshold": args.silence_threshold, "patience_chunks": args.silence_patience},
    }
    if step_times is not None:
        timing.update({
            "step_ms_mean": round(step_times.mean() * 1000, 3),
            "step_ms_p50": round(np.percentile(step_times, 50) * 1000, 3),
            "step_ms_p95": round(np.percentile(step_times, 95) * 1000, 3),
            "step_ms_max": round(step_times.max() * 1000, 3),
            "chunk_ms": HOP_LENGTH / SAMPLE_RATE * 1000,
            "end_to_end_latency_estimate_ms": round((INHERENT_DELAY_SEC + HOP_LENGTH / SAMPLE_RATE) * 1000
                                                    + np.percentile(step_times, 95) * 1000, 1),
        })
    if args.ref_midi:
        timing["eval_vs_ref_midi"] = evaluate(notes, args.ref_midi, args.max_seconds)
        timing["ref_midi"] = str(args.ref_midi)

    meta = {
        "model": "online_amt AR_Transcriber (Kwon/Jeong/Nam ISMIR 2020), checkpoint model-180000.pt",
        "audio": str(audio_path), "sample_rate": SAMPLE_RATE, "hop": HOP_LENGTH, "fps": FPS,
        "pitch_range_midi": [MIN_MIDI, MIN_MIDI + 87],
        "velocity_source": "heuristic RMS-dB proxy (model has no velocity output)",
        "state_names": STATE_NAMES, "timing": timing,
    }
    (out_dir / "amt_notes.json").write_text(json.dumps({"meta": meta, "notes": notes}, indent=1))
    (out_dir / "amt_timing.json").write_text(json.dumps(timing, indent=1))
    np.save(out_dir / "amt_roll.npy", roll)
    np.save(out_dir / "amt_states.npy", states)
    write_midi(notes, out_dir / "amt.mid")
    if not args.no_plot:
        plot_roll(roll, FPS, out_dir / "amt_roll.png", f"{audio_path.stem} - online_amt ({args.mode})")

    print(f"[amt] {len(notes)} notes, {T} frames ({FPS} fps)")
    print(f"[amt] processed {dur:.1f}s audio in {total:.2f}s -> RTF {rtf:.3f} ({1/rtf:.1f}x real time), "
          f"{T/total:.0f} frames/s")
    if step_times is not None:
        print(f"[amt] per-step: mean {timing['step_ms_mean']:.2f} ms, p95 {timing['step_ms_p95']:.2f} ms, "
              f"max {timing['step_ms_max']:.2f} ms (chunk = 32 ms)")
        print(f"[amt] inherent delay 160 ms (+32 ms chunk + compute) ~= {timing['end_to_end_latency_estimate_ms']} ms end-to-end")
    if args.ref_midi:
        print(f"[amt] eval vs {Path(args.ref_midi).name}: {timing['eval_vs_ref_midi']}")
    print(f"[amt] outputs -> {out_dir}")


if __name__ == "__main__":
    main()

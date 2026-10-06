import cv2
import numpy as np
import os
import subprocess
import json
from collections import deque

# ---------------- PATHS ----------------
video_path = "D:\\JPP\\Super 50 News  आज की 50 बड़ी खबरें । SIR Row Escalates  Rahul Gandhi  Akhilesh Yadav  TMC  BJP - IndiaTV (720p, h264).mp4"    # <-- change this
output_folder = "D:\\JPP\\TSVS"
os.makedirs(output_folder, exist_ok=True)

# ---------------- VIDEO LOAD ----------------
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ---------------- HISTOGRAM FUNCTION ----------------
def compute_hist(frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hY  = cv2.calcHist([yuv], [0], None, [256], [0,256])
    hCr = cv2.calcHist([yuv], [1], None, [256], [0,256])
    hCb = cv2.calcHist([yuv], [2], None, [256], [0,256])
    return np.concatenate((hY, hCr, hCb))

# ---------------- BASELINE CALCULATION ----------------
baseline_start_sec = 19
baseline_end_sec = 40

baseline_start_frame = int(fps * baseline_start_sec)
baseline_end_frame   = int(fps * baseline_end_sec)

baseline_diffs = []

cap.set(cv2.CAP_PROP_POS_FRAMES, baseline_start_frame)
ret, prev_frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read baseline start frame")

prev_hist = compute_hist(prev_frame)
current_frame = baseline_start_frame + 1

while current_frame < baseline_end_frame:
    ret, frame = cap.read()
    if not ret:
        break

    curr_hist = compute_hist(frame)
    diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CHISQR)
    baseline_diffs.append(diff)

    prev_hist = curr_hist
    current_frame += 1

baseline_threshold = np.mean(baseline_diffs) + 5 * np.std(baseline_diffs)
print(f"Baseline threshold: {baseline_threshold:.2f}")

# ---------------- RESET VIDEO ----------------
cap.release()
cap = cv2.VideoCapture(video_path)

ret, prev_frame = cap.read()
prev_hist = compute_hist(prev_frame)

# ---------------- ADAPTIVE THRESHOLD ----------------
adaptive_window = int(fps * 2)
k_adaptive = 4
recent_diffs = deque(maxlen=adaptive_window)

# ---------------- SHOT DETECTION ----------------
shots = []
start_frame = 0
frame_idx = 1
min_frames = int(fps * 2.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_hist = compute_hist(frame)
    diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CHISQR)
    recent_diffs.append(diff)

    if len(recent_diffs) >= adaptive_window:
        adaptive_threshold = np.mean(recent_diffs) + k_adaptive * np.std(recent_diffs)
    else:
        adaptive_threshold = baseline_threshold

    final_threshold = max(baseline_threshold, adaptive_threshold)

    if diff > final_threshold:
        shot_length = frame_idx - start_frame
        if shot_length >= min_frames:
            shots.append((start_frame, frame_idx))
        start_frame = frame_idx
        recent_diffs.clear()

    prev_hist = curr_hist
    frame_idx += 1

# ---------------- HANDLE LAST SHOT ----------------
if frame_idx - start_frame >= min_frames:
    shots.append((start_frame, frame_idx))

cap.release()
print(f"Detected {len(shots)} valid shots")

# ---------------- FRAME → TIME ----------------
def frame_to_time(frame):
    return frame / fps

# ---------------- SAVE SHOT TIMESTAMPS ----------------
shot_timestamps = []

for i, (start, end) in enumerate(shots, 1):
    start_time = frame_to_time(start)
    end_time   = frame_to_time(end)

    shot_timestamps.append({
        "shot_id": i,
        "start_frame": start,
        "end_frame": end,
        "start_time": round(start_time, 3),
        "end_time": round(end_time, 3),
        "duration": round(end_time - start_time, 3)
    })

timestamp_file = os.path.join(output_folder, "shot_timestamps.json")
with open(timestamp_file, "w") as f:
    json.dump(shot_timestamps, f, indent=4)

print(f"Shot timestamps saved: {timestamp_file}")

# ---------------- EXTRACT SHOTS WITH AUDIO ----------------
for i, shot in enumerate(shot_timestamps, 1):
    output_path = os.path.join(output_folder, f"shot_{i:03d}.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{shot['start_time']}",
        "-i", video_path,
        "-t", f"{shot['duration']}",
        "-c:v", "copy",
        "-c:a", "copy",
        output_path
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Saved: {output_path}")

print("Finished saving shots + timestamps (ready for reuse)")

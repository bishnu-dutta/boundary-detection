import cv2
import numpy as np
import os
import subprocess
from collections import deque

# ---------------- PATHS ----------------
video_path = "D:\\JPPvis\\Super 50 News  आज की 50 बड़ी खबरें । SIR Row Escalates  Rahul Gandhi  Akhilesh Yadav  TMC  BJP - IndiaTV (720p, h264).mp4"
output_folder = "D:\\JPPvis\\final"
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

# ---------------- BASELINE CALCULATION (ns → ns) ----------------
baseline_start_sec = 19
baseline_end_sec = 40

baseline_start_frame = int(fps * baseline_start_sec)
baseline_end_frame   = int(fps * baseline_end_sec)

baseline_diffs = []

# Seek to baseline start
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
print(
    f"Baseline threshold (from {baseline_start_sec}s to "
    f"{baseline_end_sec}s): {baseline_threshold:.2f}"
)

# ---------------- RESET VIDEO ----------------
cap.release()
cap = cv2.VideoCapture(video_path)

ret, prev_frame = cap.read()
prev_hist = compute_hist(prev_frame)

# ---------------- ADAPTIVE THRESHOLD SETUP ----------------
adaptive_window = int(fps * 2 )   # last 5 seconds
k_adaptive = 4                    # sensitivity
recent_diffs = deque(maxlen=adaptive_window) # only given 5 seconds and window will move next

# ---------------- SHOT DETECTION ----------------
shots = []
start_frame = 0
frame_idx = 1

min_frames = int(fps * 2.0)   # 2 second minimum

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_hist = compute_hist(frame)
    diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CHISQR)

    recent_diffs.append(diff)

    if len(recent_diffs) >= adaptive_window:
        adaptive_threshold = (
            np.mean(recent_diffs) + k_adaptive * np.std(recent_diffs)
        )
    else:
        adaptive_threshold = baseline_threshold

    final_threshold = max(baseline_threshold, adaptive_threshold)

    # ---- HARD CUT DETECTED ----
    if diff > final_threshold:
        shot_length = frame_idx - start_frame

        if shot_length >= min_frames:
            shots.append((start_frame, frame_idx))
        else:
            print(f"Discarded short shot ({shot_length/fps:.2f}s)")

        start_frame = frame_idx
        recent_diffs.clear()  # reset window after cut

    prev_hist = curr_hist
    frame_idx += 1

# ---------------- HANDLE LAST SHOT ----------------
final_shot_length = frame_idx - start_frame
if final_shot_length >= min_frames:
    shots.append((start_frame, frame_idx))
else:
    print(f"Discarded last short shot ({final_shot_length/fps:.2f}s)")

cap.release()

print(f"Detected {len(shots)} valid shots")

# ---------------- FRAME → TIME ----------------
def frame_to_time(frame):
    return frame / fps

# ---------------- EXTRACT SHOTS WITH AUDIO ----------------
for i, (start, end) in enumerate(shots, 1):
    start_time = frame_to_time(start)
    duration = frame_to_time(end - start)

    output_path = os.path.join(output_folder, f"shot_{i:03d}.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_time:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "copy",
        "-c:a", "copy",
        output_path
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Saved: {output_path}")

print("Finished saving shots with audio (adaptive threshold enabled)!")

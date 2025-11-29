import numpy as np
import imageio.v2 as imageio  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

CSV_PATH = "events.csv"
TICKS_PER_FRAME = 100
NUM_FRAMES = -1   # set to -1 to use all possible frames
GRID_SIZE = 4     # grid dimension in pixels
VIDEO_PATH = "event_frames.mp4"
CSV_OUT_PATH = "bounding_box_events.csv"

# -------------------------------------------------
# Load events
# -------------------------------------------------
events = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)

x = events[:, 0].astype(int)
y = events[:, 1].astype(int)
p = events[:, 2]
t = events[:, 3]

total_ticks = t.max() - t.min()
frames_needed = int(np.ceil(total_ticks / TICKS_PER_FRAME))

# handle -1 (all frames) and clamp if user asks for more than possible
if NUM_FRAMES == -1 or NUM_FRAMES > frames_needed:
    NUM_FRAMES = frames_needed
if NUM_FRAMES < 1:
    NUM_FRAMES = 1

t0 = t.min()
t_end = t0 + TICKS_PER_FRAME * NUM_FRAMES
mask_all = (t >= t0) & (t < t_end)

x_all = x[mask_all]
y_all = y[mask_all]
t_all = t[mask_all]

if x_all.size == 0 or y_all.size == 0:
    raise ValueError("No events in selected time range")

height = int(y_all.max()) + 1
width = int(x_all.max()) + 1

# -------------------------------------------------
# Build per frame pixel counts
# -------------------------------------------------
frames_counts = np.zeros((NUM_FRAMES, height, width), dtype=np.int32)

for i in range(NUM_FRAMES):
    t_start_i = t0 + i * TICKS_PER_FRAME
    t_end_i = t_start_i + TICKS_PER_FRAME
    mask_i = (t_all >= t_start_i) & (t_all < t_end_i)
    xi = x_all[mask_i]
    yi = y_all[mask_i]
    if xi.size > 0:
        np.add.at(frames_counts[i], (yi, xi), 1)

# -------------------------------------------------
# Clustered frames (grid brightness, per frame normalized)
# -------------------------------------------------
clustered_frames = np.zeros_like(frames_counts, dtype=np.float32)

for i in range(NUM_FRAMES):
    fc = frames_counts[i]
    if fc.sum() == 0:
        continue

    h_eff = (height // GRID_SIZE) * GRID_SIZE
    w_eff = (width // GRID_SIZE) * GRID_SIZE
    if h_eff == 0 or w_eff == 0:
        continue

    cropped = fc[:h_eff, :w_eff]

    coarse = cropped.reshape(
        h_eff // GRID_SIZE, GRID_SIZE,
        w_eff // GRID_SIZE, GRID_SIZE
    ).sum(axis=(1, 3))

    coarse = coarse.astype(np.float32)
    frame_max = coarse.max()
    if frame_max > 0:
        coarse /= frame_max
    coarse = np.clip(coarse, 0.0, 1.0)

    upscaled = np.repeat(np.repeat(coarse, GRID_SIZE, axis=0), GRID_SIZE, axis=1)
    clustered_frames[i, :h_eff, :w_eff] = upscaled

# -------------------------------------------------
# Hard coded bounding box
# -------------------------------------------------
x0, y0 = 159.13419913419915, 1.7510822510822592
x1, y1 = 204.15584415584416, 255.70129870129867

x0_i = int(np.floor(x0))
y0_i = int(np.floor(y0))
x1_i = int(np.ceil(x1))
y1_i = int(np.ceil(y1))

# clamp to image bounds just in case
x0_i = max(0, min(x0_i, width - 1))
x1_i = max(0, min(x1_i, width))
y0_i = max(0, min(y0_i, height - 1))
y1_i = max(0, min(y1_i, height))

# -------------------------------------------------
# Compute per frame bounding box stats and write video
# -------------------------------------------------
rows = []  # for bounding_box_events.csv

writer = imageio.get_writer(VIDEO_PATH, fps=10)

for i in range(NUM_FRAMES):
    fc = frames_counts[i]

    in_box = int(fc[y0_i:y1_i, x0_i:x1_i].sum())
    total = int(fc.sum())
    out_box = total - in_box
    rows.append([i, out_box, in_box])

    frame_float = clustered_frames[i]
    frame_float = np.clip(frame_float, 0.0, 1.0)
    frame_gray = (frame_float * 255).astype(np.uint8)

    frame_rgb = np.stack([frame_gray, frame_gray, frame_gray], axis=-1)

    frame_rgb[y0_i, x0_i:x1_i, :] = [255, 0, 0]
    frame_rgb[y1_i - 1, x0_i:x1_i, :] = [255, 0, 0]
    frame_rgb[y0_i:y1_i, x0_i, :] = [255, 0, 0]
    frame_rgb[y0_i:y1_i, x1_i - 1, :] = [255, 0, 0]

    writer.append_data(frame_rgb)

writer.close()

# -------------------------------------------------
# Save bounding box stats csv (no header)
# -------------------------------------------------
rows_arr = np.array(rows, dtype=np.int64)
np.savetxt(CSV_OUT_PATH, rows_arr, fmt="%d", delimiter=",")

print("Saved video to:", VIDEO_PATH)
print("Saved bounding box stats to:", CSV_OUT_PATH)

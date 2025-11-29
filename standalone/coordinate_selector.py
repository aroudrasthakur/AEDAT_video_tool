import numpy as np
import matplotlib.pyplot as plt # type: ignore

CSV_PATH = "events.csv"
TICKS_WINDOW = 10000   # ticks per frame

events = np.loadtxt(CSV_PATH, delimiter=",")

x = events[:, 0].astype(int)
y = events[:, 1].astype(int)
p = events[:, 2]
t = events[:, 3]

t0 = t.min()
t_end = t.max()

total_ticks = t_end - t0
num_frames = int(np.ceil(total_ticks / TICKS_WINDOW))

print("Total frames:", num_frames)

best_frame_idx = None
best_frame_event_count = -1

frames_meta = []  # stores (start, end, event_count)

for i in range(num_frames):
    t_start = t0 + i * TICKS_WINDOW
    t_end_i = t_start + TICKS_WINDOW

    mask = (t >= t_start) & (t < t_end_i)
    event_count = np.sum(mask)

    frames_meta.append((t_start, t_end_i, event_count))

    if event_count > best_frame_event_count:
        best_frame_event_count = event_count
        best_frame_idx = i

best_start, best_end, _ = frames_meta[best_frame_idx]

print(f"\nSelected frame {best_frame_idx} with {best_frame_event_count} events")

mask = (t >= best_start) & (t < best_end)
x_sel = x[mask]
y_sel = y[mask]

width = x_sel.max() + 1
height = y_sel.max() + 1

frame = np.zeros((height, width), dtype=np.float32)
np.add.at(frame, (y_sel, x_sel), 1)

if frame.max() > 0:
    frame /= frame.max()

fig, ax = plt.subplots()
ax.set_title(f"Frame {best_frame_idx} (Most events). Click bottom-left and top-right.")
im = ax.imshow(frame, cmap="gray", origin="lower")

pts = plt.ginput(2, timeout=-1)
print("\nSelected points:", pts, "\n")

plt.show()

import dv #type: ignore
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

root = tk.Tk()
root.withdraw()

aedat_path = filedialog.askopenfilename(
    title="Select AEDAT File",
    filetypes=[("AEDAT files", "*.aedat *.aedat4 *.aedat3"), ("All files", "*.*")]
)

if not aedat_path:
    raise ValueError("No AEDAT file selected")

script_dir = os.path.dirname(os.path.abspath(__file__))

reader = dv.AedatFile(aedat_path)

events = []

if "events" in reader.names:
    for e in reader["events"]:
        xs = np.asarray(e.x)
        ys = np.asarray(e.y)
        ps = np.asarray(e.polarity, dtype=int)
        ts = np.asarray(e.timestamp)

        block = np.column_stack((xs, ys, ps, ts))
        events.append(block)
else:
    raise ValueError("Event stream not found in AEDAT file")

events = np.vstack(events)

events[:, 3] -= events[0, 3]  # zero based timestamps

csv_path = os.path.join(script_dir, "events.csv")
np.savetxt(csv_path, events, fmt="%d,%d,%d,%d")

print("Saved:", csv_path)

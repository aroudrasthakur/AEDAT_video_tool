# AEDAT Stream Viewer

A web application for side-by-side comparison of event and RGB streams from AEDAT4 files.

## Features

- **Dual Stream Visualization**: Event data (Stream 0) and RGB frames (Stream 1) displayed side by side
- **Synchronized Playback**: Both streams stay in sync during playback and scrubbing
- **Interactive Timeline**: Scrub through the recording with a slider
- **Playback Controls**: Play/pause and speed control
- **File Upload**: Load local AEDAT4 files directly in the browser
- **Generate / Focus Mode**: Backend can generate frames for the full recording or only a focused time window around the busiest event period (upload form accepts `generate_mode` = `all`|`focus` and `focus_ms` in milliseconds).
- **High-frequency Event Updates**: Frontend renders event windows at a higher frequency than RGB frames (configurable `dt_events`) so events remain low-latency while RGB updates are coarser.
- **Null RGB Placeholder**: When no RGB frames are present, the UI shows an inline placeholder icon instead of an empty image.

## Architecture

### Backend (Python + FastAPI)

- Parses AEDAT4 files using the `aedat` library
- Extracts and caches event data and RGB frames
- Serves frames and events via REST API
- Handles temporal alignment and synchronization

Notes:

- The `/upload` endpoint accepts additional form fields: `generate_mode` ("all" or "focus") and `focus_ms` (integer, milliseconds). When `generate_mode=focus`, the backend finds the most active event window and trims generated event windows and RGB frames to ±`focus_ms` around that center to speed up processing for long recordings.
- The backend groups events into fixed windows (default ~50 ms) for efficient rendering. The frontend further requests events in a small sliding `dt_events` window around the current play time for low-latency updates.
- The server will read files and serve `index.html` with UTF-8 encoding to avoid platform-specific decoding issues on Windows.

### Frontend (HTML + Vanilla JavaScript)

- Canvas-based event visualization (green=ON, blue=OFF events)
- Image-based RGB frame display
- Responsive split-pane layout
- Smooth playback with requestAnimationFrame

Notes:

- The frontend uses a single global time `t` (raw hardware ticks, µs) to synchronize events and RGB frames. Events are requested with a small `dt_events` window for frequent updates; RGB frames are updated more coarsely using a nearest/threshold strategy.
- For the most accurate synchronization, the backend exposes RGB frame timestamps in metadata (when available). If a backend build returns only `rgb_frame_count`, the frontend may synthesize uniform timestamps — for best results ensure the backend metadata includes exact frame times.

## AEDAT Format Assumptions

- **Format**: AEDAT4 (binary format with packet-based streams)
- **Stream 0**: Event data with fields `t` (timestamp), `x`, `y`, `p` (polarity)
- **Stream 1**: RGB frame data with timestamp and pixel array
- **Timestamps**: Raw hardware ticks (typically microseconds)
- **Coordinates**: 0-indexed pixel coordinates

## Requirements

```bash
pip install fastapi uvicorn python-multipart aedat numpy pillow
```

## How to Run

1. **Start the backend server**:

   ```bash
   cd aedat_viewer
   # Run using the Python interpreter that has the required packages installed
   python -m uvicorn main:app --reload --port 8000
   ```

2. **Open the web interface**:

   - Navigate to: http://localhost:8000
   - Or open `index.html` directly in a browser (static mode)

3. **Load an AEDAT file**:
   - Click "Choose AEDAT File"
   - Select your `.aedat4` file
   - Wait for parsing (progress shown in console)
   - Use playback controls to view synchronized streams

## API Endpoints

- `GET /` - Serves the main HTML page
- `POST /upload` - Upload and parse AEDAT file
- `GET /metadata` - Get file info (duration, resolution, frame counts)
- `GET /rgb_frame?t={timestamp}` - Get RGB frame closest to timestamp
- `GET /events_window?start_t={start}&end_t={end}` - Get events in time window
- `GET /reset` - Clear cached data

Additional upload/query notes:

- `POST /upload` accepts form fields `generate_mode` and `focus_ms` to control generation scope.
- `/metadata` returns cached summary info (duration, width/height, counts). When possible the backend will also include exact RGB frame timestamps so the frontend can synchronize frames precisely.

## Performance Notes

- Events are grouped into 50ms windows for efficient rendering
- RGB frames are cached as JPEG for fast delivery
- Frontend uses double buffering for smooth canvas updates
- Playback uses requestAnimationFrame for 60fps rendering

## Dummy Mode

If you want to test without a real AEDAT file, uncomment the dummy data generator in `main.py` (lines marked with `# DUMMY MODE`). This will generate synthetic event and frame data for testing the UI.

## Project Structure

```
aedat_viewer/
├── main.py          # FastAPI backend server
├── index.html       # Frontend HTML + CSS + JavaScript
└── README.md        # This file
```

## Troubleshooting

**"No events found"**: Check that your AEDAT file has Stream 0 with event data.

**"No RGB frames found"**: Check that your AEDAT file has Stream 1 with frame data.

**Slow loading**: Large files may take time to parse. Check console for progress.

**Out of sync**: Click "Reset" and reload the file to recalculate synchronization. If timing still looks off, ensure the backend returned exact RGB timestamps in `/metadata` and that you are using the same raw tick units (µs) in the frontend configuration.

**Windows UTF-8 server error**: If you previously saw a UnicodeDecodeError while serving `index.html` on Windows, ensure you start the server from inside the `aedat_viewer` folder and use the provided `python -m uvicorn ...` command. The server reads `index.html` using UTF-8 to avoid cp1252 decoding failures.

"""
AEDAT Stream Viewer - Backend Server

This FastAPI server handles:
1. AEDAT4 file parsing and caching
2. Event data extraction and windowing
3. RGB frame extraction and JPEG encoding
4. Temporal synchronization between streams

Dependencies:
    pip install fastapi uvicorn python-multipart aedat numpy pillow

Run:
    uvicorn main:app --reload --port 8000

Then open: http://localhost:8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import aedat
import numpy as np
from io import BytesIO
from PIL import Image
import tempfile
import os
from typing import Dict, List, Optional
from pathlib import Path

app = FastAPI(title="AEDAT Stream Viewer")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for parsed AEDAT data
cache: Dict = {
    "metadata": None,
    "event_windows": [],  # List of event windows with timestamps
    "rgb_frames": [],     # List of (timestamp, image_data)
    "temp_file": None,
}


def extract_aps_rgb(packet, width=None, height=None):
    """Extract RGB image from APS packet."""
    frame = packet.get("frame", packet)
    
    if isinstance(frame, dict):
        for key in ("image", "pixels", "data"):
            if key in frame:
                arr = np.asarray(frame[key])
                if arr.ndim == 1 and width and height:
                    arr = arr.reshape(height, width)
                if arr.ndim == 2:
                    # Grayscale - convert to RGB
                    arr = np.stack([arr, arr, arr], axis=2)
                if arr.ndim == 3:
                    if arr.shape[2] == 1:
                        arr = np.repeat(arr, 3, axis=2)
                    elif arr.shape[2] == 4:
                        arr = arr[:, :, :3]
                    # Ensure uint8
                    if arr.dtype != np.uint8:
                        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
                    return arr
    
    if isinstance(frame, np.ndarray):
        arr = frame
        if arr.ndim == 1 and width and height:
            arr = arr.reshape(height, width)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=2)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.dtype != np.uint8:
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        return arr
    
    return None


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    generate_mode: str = Form("all"),  # 'all' or 'focus'
    focus_ms: int = Form(100)           # when generate_mode=='focus', window half-width in ms
):
    """
    Upload and parse AEDAT4 file.
    
    Extracts:
    - Stream 0: Event data (x, y, polarity, timestamp)
    - Stream 1: RGB frames with timestamps
    
    Returns metadata about the parsed file.
    """
    global cache
    
    # Clear previous cache
    if cache["temp_file"] and os.path.exists(cache["temp_file"]):
        os.remove(cache["temp_file"])
    cache = {
        "metadata": None,
        "event_windows": [],
        "rgb_frames": [],
        "temp_file": None,
    }
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".aedat4") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    cache["temp_file"] = tmp_path
    
    try:
        # Parse AEDAT file
        decoder = aedat.Decoder(tmp_path)
        streams = decoder.id_to_stream()
        
        # Get stream metadata
        event_meta = streams.get(0, {})
        rgb_meta = streams.get(1, {})
        
        width = event_meta.get("width", 640)
        height = event_meta.get("height", 480)
        
        # Extract events (Stream 0) - group into 50ms windows
        print("Extracting events from Stream 0...")
        WINDOW_SIZE = 50000  # 50ms in microseconds
        current_window = []
        window_start_t = None
        event_count = 0
        
        for packet in decoder:
            if packet.get("stream_id") == 0 and "events" in packet:
                events = packet["events"]
                if events.size > 0:
                    event_count += len(events)
                    
                    for i in range(len(events)):
                        t = int(events['t'][i])
                        x = int(events['x'][i])
                        y = int(events['y'][i])
                        p = bool(events['p'][i])
                        
                        if window_start_t is None:
                            window_start_t = t
                        
                        # Check if we need to start a new window
                        if t >= window_start_t + WINDOW_SIZE:
                            if current_window:
                                cache["event_windows"].append({
                                    "start_t": window_start_t,
                                    "end_t": window_start_t + WINDOW_SIZE,
                                    "events": current_window.copy()
                                })
                            current_window = []
                            window_start_t = (t // WINDOW_SIZE) * WINDOW_SIZE
                        
                        current_window.append({"x": x, "y": y, "p": p, "t": t})
        
        # Add final window
        if current_window:
            cache["event_windows"].append({
                "start_t": window_start_t,
                "end_t": window_start_t + WINDOW_SIZE,
                "events": current_window
            })
        
        print(f"Extracted {event_count} events in {len(cache['event_windows'])} windows")

        # If generate_mode == 'focus', find the event window with maximum events
        # and restrict subsequent frame extraction (and optionally events) to
        # the time range [center - focus_ms, center + focus_ms].
        focus_range_us = int(focus_ms) * 1000
        if generate_mode == 'focus' and cache['event_windows']:
            max_window = max(cache['event_windows'], key=lambda w: len(w['events']))
            # center of the max window
            center_t = (max_window['start_t'] + max_window['end_t']) // 2
            focus_start = max(0, center_t - focus_range_us)
            focus_end = center_t + focus_range_us
            print(f"Focus mode: selecting events between {focus_start} and {focus_end} (±{focus_ms}ms around center {center_t})")
            # Optionally trim event_windows to only those overlapping focus range
            filtered_windows = []
            for w in cache['event_windows']:
                if w['end_t'] >= focus_start and w['start_t'] <= focus_end:
                    filtered_windows.append(w)
            cache['event_windows'] = filtered_windows
        else:
            focus_start = None
            focus_end = None

        # Extract RGB frames (Stream 1)
        print("Extracting RGB frames from Stream 1...")
        decoder = aedat.Decoder(tmp_path)  # Re-create decoder
        frame_count = 0
        
        for packet in decoder:
            if packet.get("stream_id") == 1:
                # Try to get timestamp
                t_val = None
                for key in ("t", "timestamp", "ts", "time"):
                    if key in packet:
                        t_val = packet[key]
                        break
                if t_val is None and "frame" in packet and isinstance(packet["frame"], dict):
                    for key in ("t", "timestamp", "ts", "time"):
                        if key in packet["frame"]:
                            t_val = packet["frame"][key]
                            break
                
                if t_val is None:
                    continue
                
                # Extract RGB image
                rgb_img = extract_aps_rgb(packet, width, height)
                if rgb_img is not None:
                    # Convert to JPEG bytes
                    img_pil = Image.fromarray(rgb_img)
                    buf = BytesIO()
                    img_pil.save(buf, format="JPEG", quality=85)
                    buf.seek(0)
                    # If focus mode is enabled, only store frames within the focus range
                    if generate_mode == 'focus' and focus_start is not None and focus_end is not None:
                        if not (focus_start <= int(t_val) <= focus_end):
                            continue

                    cache["rgb_frames"].append({
                        "timestamp": int(t_val),
                        "data": buf.getvalue()
                    })
                    frame_count += 1
        
        print(f"Extracted {frame_count} RGB frames")
        
        # Calculate metadata
        if cache["event_windows"] and cache["rgb_frames"]:
            min_t = min(cache["event_windows"][0]["start_t"], cache["rgb_frames"][0]["timestamp"])
            max_t = max(cache["event_windows"][-1]["end_t"], cache["rgb_frames"][-1]["timestamp"])
        elif cache["event_windows"]:
            min_t = cache["event_windows"][0]["start_t"]
            max_t = cache["event_windows"][-1]["end_t"]
        elif cache["rgb_frames"]:
            min_t = cache["rgb_frames"][0]["timestamp"]
            max_t = cache["rgb_frames"][-1]["timestamp"]
        else:
            raise HTTPException(status_code=400, detail="No event or RGB data found in file")
        
        cache["metadata"] = {
            "duration_ticks": max_t - min_t,
            "start_time": min_t,
            "end_time": max_t,
            "width": width,
            "height": height,
            "event_window_count": len(cache["event_windows"]),
            "rgb_frame_count": len(cache["rgb_frames"]),
            "total_events": event_count,
        }
        
        return JSONResponse(content={
            "success": True,
            "metadata": cache["metadata"]
        })
    
    except Exception as e:
        import traceback
        print(f"Error parsing file: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error parsing AEDAT file: {str(e)}")


@app.get("/metadata")
async def get_metadata():
    """Get metadata about the loaded AEDAT file."""
    if cache["metadata"] is None:
        raise HTTPException(status_code=400, detail="No file loaded")
    return cache["metadata"]


@app.get("/rgb_frame")
async def get_rgb_frame(t: int):
    """
    Get RGB frame closest to timestamp t.
    
    Args:
        t: Timestamp in ticks
    
    Returns:
        JPEG image
    """
    if not cache["rgb_frames"]:
        raise HTTPException(status_code=400, detail="No RGB frames available")
    
    # Find closest frame by timestamp
    closest_idx = min(
        range(len(cache["rgb_frames"])),
        key=lambda i: abs(cache["rgb_frames"][i]["timestamp"] - t)
    )
    
    frame_data = cache["rgb_frames"][closest_idx]["data"]
    
    return StreamingResponse(
        BytesIO(frame_data),
        media_type="image/jpeg"
    )


@app.get("/events_window")
async def get_events_window(start_t: int, end_t: int):
    """
    Get events in time window [start_t, end_t).
    
    Args:
        start_t: Start timestamp in ticks
        end_t: End timestamp in ticks
    
    Returns:
        JSON array of events: [{x, y, p, t}, ...]
    """
    if not cache["event_windows"]:
        raise HTTPException(status_code=400, detail="No event data available")
    
    # Find overlapping windows
    result_events = []
    for window in cache["event_windows"]:
        # Check if window overlaps with requested range
        if window["end_t"] >= start_t and window["start_t"] <= end_t:
            # Add events from this window that fall in range
            for event in window["events"]:
                if start_t <= event["t"] < end_t:
                    result_events.append(event)
    
    return {"events": result_events, "count": len(result_events)}


@app.get("/reset")
async def reset_cache():
    """Clear cached data."""
    global cache
    if cache["temp_file"] and os.path.exists(cache["temp_file"]):
        os.remove(cache["temp_file"])
    cache = {
        "metadata": None,
        "event_windows": [],
        "rgb_frames": [],
        "temp_file": None,
    }
    return {"success": True}


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main HTML page."""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding='utf-8')
    else:
        return """
        <html>
        <head><title>AEDAT Viewer</title></head>
        <body>
            <h1>AEDAT Viewer</h1>
            <p>index.html not found. Please create it in the same directory as main.py</p>
        </body>
        </html>
        """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

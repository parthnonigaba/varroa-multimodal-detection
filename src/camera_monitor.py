"""
Camera Monitor - IMX708 Bee & Varroa Detection
Updated: Periodic capture mode (no live stream) for smooth, lag-free operation
         Added --vflip --hflip for upside-down mounted camera

Workflow:
- Continuously capture frames and run ML detection
- Save snapshot + 30s video + trigger audio every 10 minutes
- Immediately save on varroa detection
- No live streaming (reduces CPU load, enables smooth video with autofocus)
"""

import os
import time
import cv2
import threading
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Tuple

from ultralytics import YOLO
import numpy as np


class IMX708BeeMonitor:
    def __init__(self, config: Dict[str, Any]) -> None:
        c = config.get("camera", {})
        self.device = c.get("device", "/dev/video0")
        self.width, self.height = c.get("resolution", [2304, 1296])
        self.fps = int(c.get("framerate", 30))

        self.bee_model_path = c.get("bee_model_path")
        self.varroa_model_path = c.get("varroa_model_path")
        self.bee_conf = float(c.get("bee_confidence", 0.6))
        self.varroa_conf = float(c.get("varroa_confidence", 0.7))
        self.roi_min = int(c.get("roi_min_size", 32))
        self.stride = int(c.get("detection_stride", 4))
        self.burst_stride = int(c.get("burst_stride", 1))
        self.burst_seconds = int(c.get("burst_seconds", 10))
        self.clip_seconds = int(c.get("clip_seconds", 30))
        
        # Camera flip settings (for upside-down mounted camera)
        self.vflip = c.get("vflip", True)  # Vertical flip
        self.hflip = c.get("hflip", True)  # Horizontal flip
        
        # Intervals
        self.routine_interval = 600  # 10 minutes for routine saves
        self.detection_interval = 5  # Seconds between detection frames (lower = more CPU)

        s = config.get("storage", {})
        self.captures_dir = s.get("captures_dir", "data/captures")
        self.varroa_dir = s.get("varroa_dir", "data/varroa_detections")
        self.clips_dir = s.get("clips_dir", "data/clips")
        os.makedirs(self.captures_dir, exist_ok=True)
        os.makedirs(self.varroa_dir, exist_ok=True)
        os.makedirs(self.clips_dir, exist_ok=True)

        # Load ML models
        self._bee_model = None
        self._varroa_model = None
        self._load_models()

        self._burst_until: Optional[float] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        
        # Track last routine save time
        self._last_routine_save = 0
        
        # Track current clip path (don't delete this one!)
        self._current_clip_path: Optional[str] = None

    def _get_flip_args(self) -> List[str]:
        """Get rpicam flip arguments based on config"""
        args = []
        if self.vflip:
            args.append("--vflip")
        if self.hflip:
            args.append("--hflip")
        return args

    def _load_models(self) -> None:
        """Load YOLO models for bee and varroa detection"""
        try:
            if self.bee_model_path and os.path.exists(self.bee_model_path):
                self._bee_model = YOLO(self.bee_model_path)
                print(f"✓ Loaded bee detector: {self.bee_model_path}")
            else:
                print(f"⚠ Bee model not found: {self.bee_model_path}")
        except Exception as e:
            print(f"✗ Failed to load bee model: {e}")
            
        try:
            if self.varroa_model_path and os.path.exists(self.varroa_model_path):
                self._varroa_model = YOLO(self.varroa_model_path)
                print(f"✓ Loaded varroa detector: {self.varroa_model_path}")
            else:
                print(f"⚠ Varroa model not found: {self.varroa_model_path}")
        except Exception as e:
            print(f"✗ Failed to load varroa model: {e}")

    def stop(self) -> None:
        """Signal the monitor to stop"""
        self._stop.set()

    def _current_stride(self) -> int:
        """Get current frame stride (lower during burst mode for more detection)"""
        with self._lock:
            if self._burst_until and time.time() < self._burst_until:
                return self.burst_stride
        return self.stride

    def _trigger_burst(self) -> None:
        """Enter burst mode for more frequent detection after varroa found"""
        with self._lock:
            self._burst_until = time.time() + self.burst_seconds

    def _take_snapshot(self) -> Optional[Tuple[str, np.ndarray]]:
        """
        Take a high-quality snapshot using rpicam-still with autofocus
        Returns: (path, image_array) or None on failure
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_path = os.path.join(self.captures_dir, f"snapshot_{timestamp}.jpg")
        
        cmd = [
            "rpicam-still",
            "--timeout", "2000",  # 2 seconds for autofocus to settle
            "--autofocus-mode", "auto",  # Single AF for sharp still
            "--width", str(self.width),
            "--height", str(self.height),
            "-o", snap_path,
            "--nopreview"
        ] + self._get_flip_args()
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode == 0 and os.path.exists(snap_path):
                img = cv2.imread(snap_path)
                if img is not None:
                    print(f"✓ Snapshot: {snap_path}")
                    return snap_path, img
            else:
                stderr = result.stderr.decode() if result.stderr else ""
                print(f"✗ Snapshot failed: {stderr[:100]}")
        except Exception as e:
            print(f"✗ Snapshot error: {e}")
        
        return None

    def _record_video_clip(self, duration_sec: int = 30, event_triggered: bool = False, with_audio: bool = True) -> Optional[str]:
        """
        Record a smooth video clip with audio using libav codec with continuous autofocus
        
        Args:
            duration_sec: Length of clip in seconds
            event_triggered: If True, save to varroa_dir, else to clips_dir
            with_audio: If True, record audio simultaneously and merge
        
        Returns: Path to recorded clip or None on failure
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if event_triggered:
            clip_path = os.path.join(self.varroa_dir, f"varroa_clip_{timestamp}.mp4")
        else:
            clip_path = os.path.join(self.clips_dir, f"clip_{timestamp}.mp4")
        
        # Temp files for video and audio
        temp_video = f"/tmp/temp_video_{timestamp}.mp4"
        temp_audio = f"/tmp/temp_audio_{timestamp}.wav"
        
        # Small delay to ensure camera is released from any previous operation
        time.sleep(0.5)
        
        video_cmd = [
            "rpicam-vid",
            "--timeout", str(duration_sec * 1000),  # milliseconds
            "--width", "1920",
            "--height", "1080",
            "--framerate", "30",
            "--autofocus-mode", "continuous",  # KEY: keeps moving bees sharp
            "--codec", "libav",                 # KEY: smooth MP4 directly
            "--libav-format", "mp4",
            "--bitrate", "8000000",             # 8 Mbps for good quality
            "-o", temp_video if with_audio else clip_path,
            "--nopreview"
        ] + self._get_flip_args()
        
        audio_cmd = [
            "arecord",
            "-D", "hw:2,0",        # Audio device
            "-f", "S32_LE",        # Format
            "-r", "48000",         # Sample rate
            "-c", "2",             # Stereo
            "-d", str(duration_sec),  # Duration
            temp_audio
        ]
        
        try:
            print(f"📹 Recording {duration_sec}s clip with continuous autofocus" + (" + audio..." if with_audio else "..."))
            
            if with_audio:
                # Start video recording in background
                video_proc = subprocess.Popen(
                    video_cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                
                # Start audio recording (blocks until done)
                audio_result = subprocess.run(audio_cmd, capture_output=True, timeout=duration_sec + 10)
                
                # Wait for video to finish with timeout
                try:
                    video_stdout, video_stderr = video_proc.communicate(timeout=15)
                    video_returncode = video_proc.returncode
                except subprocess.TimeoutExpired:
                    video_proc.kill()
                    video_stdout, video_stderr = video_proc.communicate()
                    video_returncode = -1
                    print(f"✗ Video recording timed out")
                
                # Debug output
                if video_returncode != 0:
                    print(f"   Video return code: {video_returncode}")
                    if video_stderr:
                        print(f"   Video stderr: {video_stderr.decode()[:200]}")
                
                # Check video succeeded
                if video_returncode != 0 or not os.path.exists(temp_video):
                    print(f"✗ Video recording failed (code={video_returncode}, exists={os.path.exists(temp_video)})")
                    return None
                
                # Check audio
                audio_ok = audio_result.returncode == 0 and os.path.exists(temp_audio)
                if not audio_ok:
                    print(f"⚠ Audio recording failed, using video only")
                    os.rename(temp_video, clip_path)
                else:
                    # Merge video and audio with ffmpeg (50x volume boost)
                    # Use -itsoffset to sync audio (negative = audio starts earlier)
                    merge_cmd = [
                        "ffmpeg",
                        "-i", temp_video,
                        "-itsoffset", "-0.5",    # Shift audio 0.5s earlier to sync
                        "-i", temp_audio,
                        "-c:v", "copy",          # Don't re-encode video
                        "-af", "volume=60.0",    # Boost audio 60x
                        "-c:a", "aac",           # Encode audio as AAC
                        "-shortest",             # Match shortest stream
                        "-y",                    # Overwrite output
                        clip_path
                    ]
                    
                    merge_result = subprocess.run(merge_cmd, capture_output=True, timeout=30)
                    
                    if merge_result.returncode != 0:
                        print(f"⚠ Merge failed, using video only")
                        if merge_result.stderr:
                            print(f"   Merge stderr: {merge_result.stderr.decode()[:200]}")
                        os.rename(temp_video, clip_path)
                    else:
                        print(f"   ✓ Audio merged successfully")
                        # Cleanup temp files
                        try:
                            os.remove(temp_video)
                            os.remove(temp_audio)
                        except:
                            pass
            else:
                # Video only (original behavior)
                result = subprocess.run(video_cmd, capture_output=True, timeout=duration_sec + 15)
                if result.returncode != 0:
                    stderr = result.stderr.decode() if result.stderr else ""
                    print(f"✗ Video recording failed: {stderr[:100]}")
                    return None
            
            if os.path.exists(clip_path):
                size_mb = os.path.getsize(clip_path) / (1024 * 1024)
                print(f"✓ Video clip: {clip_path} ({size_mb:.1f} MB)")
                
                # Update current clip path and cleanup old ones AFTER successful recording
                if not event_triggered:
                    old_clip = self._current_clip_path
                    self._current_clip_path = clip_path
                    self._cleanup_old_clips(keep_clip=clip_path)
                
                return clip_path
            else:
                print(f"✗ Clip file not created")
                return None
                
        except subprocess.TimeoutExpired:
            print("✗ Recording timed out")
        except Exception as e:
            print(f"✗ Video recording error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup temp files on error
            for f in [temp_video, temp_audio]:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except:
                    pass
        
        return None

    def _cleanup_old_clips(self, keep_clip: str = None) -> None:
        """
        Remove old routine clips to save space
        - Never delete clips less than 15 minutes old
        - Always keep the most recent clip
        - Keep up to 6 clips total (1 hour worth)
        """
        try:
            clips = []
            now = time.time()
            
            for f in os.listdir(self.clips_dir):
                if f.startswith("clip_") and f.endswith(".mp4"):
                    path = os.path.join(self.clips_dir, f)
                    
                    # Never delete the clip we just recorded or the current active clip
                    if keep_clip and os.path.abspath(path) == os.path.abspath(keep_clip):
                        continue
                    if self._current_clip_path and os.path.abspath(path) == os.path.abspath(self._current_clip_path):
                        continue
                    
                    age_minutes = (now - os.path.getmtime(path)) / 60
                    
                    # Never delete clips less than 15 minutes old
                    if age_minutes < 15:
                        continue
                    
                    clips.append((os.path.getmtime(path), path))
            
            # Sort by modification time (newest first)
            clips.sort(reverse=True)
            
            # Keep 6 clips (1 hour worth at 10-min intervals), delete older ones
            for _, path in clips[6:]:
                try:
                    os.remove(path)
                    print(f"   🗑️ Cleaned old clip: {os.path.basename(path)}")
                except Exception as e:
                    print(f"   ⚠️ Failed to delete {path}: {e}")
                    
        except Exception as e:
            print(f"Cleanup error: {e}")

    def _run_detection(self, img: np.ndarray) -> Tuple[int, int, Optional[str]]:
        """
        Run bee and varroa detection on an image
        
        Returns: (bee_count, varroa_count, annotated_image_path or None)
        """
        if self._bee_model is None:
            return 0, 0, None

        bees_count = 0
        varroa_count = 0
        annotated = img.copy()
        varroa_detected = False

        try:
            # Stage 1: Detect bees
            res = self._bee_model.predict(source=img, conf=self.bee_conf, verbose=False)[0]
            crops: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
            
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                    w, h = x2 - x1, y2 - y1
                    
                    if w < self.roi_min or h < self.roi_min:
                        continue
                    
                    bees_count += 1
                    crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    
                    if crop.size > 0:
                        crops.append((crop, (x1, y1, x2, y2)))
                    
                    # Draw green box for bee
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Stage 2: Check each bee crop for varroa
                if crops and self._varroa_model is not None:
                    for crop, (x1, y1, x2, y2) in crops:
                        vres = self._varroa_model.predict(source=crop, conf=self.varroa_conf, verbose=False)[0]
                        
                        if vres.boxes is not None and len(vres.boxes) > 0:
                            varroa_count += 1
                            varroa_detected = True
                            # Draw red box for varroa-infected bee
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(annotated, "VARROA", (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        except Exception as e:
            print(f"Detection error: {e}")
            return 0, 0, None

        # Save annotated image if detections found
        annotated_path = None
        if bees_count > 0 or varroa_detected:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            if varroa_detected:
                annotated_path = os.path.join(self.varroa_dir, f"varroa_{timestamp}.jpg")
            else:
                annotated_path = os.path.join(self.captures_dir, f"annotated_{timestamp}.jpg")
            
            try:
                cv2.imwrite(annotated_path, annotated)
            except Exception as e:
                print(f"Failed to save annotated image: {e}")
                annotated_path = None

        return bees_count, varroa_count, annotated_path

    def _capture_frame_for_detection(self) -> Optional[np.ndarray]:
        """
        Capture a single frame for ML detection using rpicam-still
        Fast capture without full autofocus delay
        """
        temp_path = "/tmp/detection_frame.jpg"
        
        cmd = [
            "rpicam-still",
            "--timeout", "500",  # Quick capture
            "--width", str(self.width),
            "--height", str(self.height),
            "-o", temp_path,
            "--nopreview",
            "--immediate"  # Don't wait for AE/AWB to settle
        ] + self._get_flip_args()
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            if result.returncode == 0 and os.path.exists(temp_path):
                img = cv2.imread(temp_path)
                try:
                    os.remove(temp_path)
                except:
                    pass
                return img
        except Exception:
            pass
        
        return None

    def run(
        self,
        on_detection_cb: Callable,
        on_snapshot_cb: Optional[Callable] = None,
        on_frame_cb: Optional[Callable] = None,  # Not used in new design
        on_clip_cb: Optional[Callable] = None,
        on_routine_cb: Optional[Callable] = None,  # NEW: called every 10 min to trigger audio
    ) -> None:
        """
        Main monitoring loop
        
        Continuously monitors via ML detection, saves periodically and on events.
        
        Callbacks:
            on_detection_cb(bees, varroa, annotated_path, clip_path): Called on each detection
            on_snapshot_cb(path): Called when snapshot is saved
            on_clip_cb(path): Called when video clip is saved
            on_routine_cb(): Called every 10 min to trigger audio recording
        """
        print("=" * 50)
        print("🐝 Camera Monitor Started (Periodic Mode)")
        print(f"   Detection: every {self.detection_interval} seconds")
        print(f"   Routine saves: every {self.routine_interval // 60} minutes")
        print(f"   Event saves: on varroa detection")
        print(f"   Camera flip: vflip={self.vflip}, hflip={self.hflip}")
        print("=" * 50)
        
        self._last_routine_save = time.time() - self.routine_interval  # Trigger first save immediately
        
        last_detection_time = 0
        
        while not self._stop.is_set():
            current_time = time.time()
            
            # === ROUTINE SAVE (every 10 minutes) ===
            if (current_time - self._last_routine_save) >= self.routine_interval:
                self._last_routine_save = current_time
                print(f"\n⏰ Routine save triggered at {datetime.now().strftime('%H:%M:%S')}")
                
                # 1. Take snapshot
                snapshot_result = self._take_snapshot()
                if snapshot_result:
                    snap_path, snap_img = snapshot_result
                    if on_snapshot_cb:
                        try:
                            on_snapshot_cb(snap_path)
                        except Exception as e:
                            print(f"Snapshot callback error: {e}")
                    
                    # Run detection on snapshot
                    bees, varroa, ann_path = self._run_detection(snap_img)
                    print(f"   Detection: {bees} bees, {varroa} varroa")
                    on_detection_cb(bees, varroa, ann_path, None)
                
                # 2. Record video clip
                clip_path = self._record_video_clip(self.clip_seconds, event_triggered=False)
                if clip_path and on_clip_cb:
                    try:
                        on_clip_cb(clip_path)
                    except Exception as e:
                        print(f"Clip callback error: {e}")
                
                # 3. Trigger audio recording
                if on_routine_cb:
                    try:
                        on_routine_cb()
                    except Exception as e:
                        print(f"Routine callback error: {e}")
                
                print(f"   Routine save complete\n")
                continue  # Skip detection this iteration since we just did it
            
            # === CONTINUOUS DETECTION (every 5 seconds) ===
            if (current_time - last_detection_time) >= self.detection_interval:
                last_detection_time = current_time
                
                # Capture frame for detection
                img = self._capture_frame_for_detection()
                if img is None:
                    time.sleep(0.5)
                    continue
                
                # Run detection
                bees, varroa, ann_path = self._run_detection(img)
                
                # Report detection (only if something found to reduce noise)
                if bees > 0 or varroa > 0:
                    on_detection_cb(bees, varroa, ann_path, None)
                
                # === EVENT-TRIGGERED SAVE (varroa detected) ===
                if varroa > 0:
                    print(f"\n🚨 VARROA DETECTED! Recording event clip...")
                    self._trigger_burst()
                    
                    # Record event video
                    event_clip = self._record_video_clip(self.clip_seconds, event_triggered=True)
                    if event_clip:
                        on_detection_cb(bees, varroa, ann_path, event_clip)
                        if on_clip_cb:
                            try:
                                on_clip_cb(event_clip)
                            except Exception as e:
                                print(f"Event clip callback error: {e}")
                    
                    print(f"   Event save complete\n")
            
            # Sleep between checks (not spinning CPU)
            time.sleep(1)
        
        print("Camera monitor stopped")


# For backwards compatibility
def create_monitor(config: Dict[str, Any]) -> IMX708BeeMonitor:
    return IMX708BeeMonitor(config)
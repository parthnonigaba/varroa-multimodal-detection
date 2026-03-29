"""
Data Manager - Database and file storage management
Handles SQLite database, file retention, and latest asset tracking
"""

import os
import json
import time
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, Optional


class DataManager:
    def __init__(self, config: Dict[str, Any]) -> None:
        storage = config.get("storage", {})
        self.base_dir = os.path.abspath(storage.get("base_dir", "data"))
        self.captures_dir = os.path.abspath(storage.get("captures_dir", os.path.join(self.base_dir, "captures")))
        self.varroa_dir = os.path.abspath(storage.get("varroa_dir", os.path.join(self.base_dir, "varroa_detections")))
        self.audio_dir = os.path.abspath(storage.get("audio_dir", os.path.join(self.base_dir, "audio")))
        self.reports_dir = os.path.abspath(storage.get("reports_dir", os.path.join(self.base_dir, "reports")))
        self.clips_dir = os.path.abspath(storage.get("clips_dir", os.path.join(self.base_dir, "clips")))
        self.db_path = os.path.abspath(storage.get("database_path", os.path.join(self.base_dir, "bee_monitor.db")))
        self.max_storage_gb = float(storage.get("max_storage_gb", 10))
        self.routine_days = int(storage.get("routine_days", 1))
        self.event_days = int(storage.get("event_days", 30))

        # Create all directories
        for d in [self.base_dir, self.captures_dir, self.varroa_dir, self.audio_dir, self.reports_dir, self.clips_dir]:
            os.makedirs(d, exist_ok=True)

        self._init_db()
        
        # Latest asset tracking
        self._latest_audio_path: Optional[str] = None
        self._latest_unhealthy_audio_path: Optional[str] = None
        self._latest_varroa_clip_path: Optional[str] = None
        self._latest_annotated_img_path: Optional[str] = None
        self._latest_snapshot_path: Optional[str] = None
        self._latest_clip_path: Optional[str] = None
        self._lock = threading.Lock()

    def _init_db(self) -> None:
        """Initialize SQLite database with all required tables"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            
            # Sensor readings
            c.execute("""CREATE TABLE IF NOT EXISTS readings(
                ts TEXT PRIMARY KEY,
                temperature REAL,
                humidity REAL,
                co2 REAL,
                risk TEXT
            )""")
            
            # Audio analysis
            c.execute("""CREATE TABLE IF NOT EXISTS audio(
                ts TEXT PRIMARY KEY,
                path TEXT,
                label TEXT,
                confidence REAL,
                unhealthy INTEGER
            )""")
            
            # Camera detections
            c.execute("""CREATE TABLE IF NOT EXISTS detections(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                bees INTEGER,
                varroa INTEGER,
                img_path TEXT
            )""")
            
            # Events (varroa, unhealthy audio, etc.)
            c.execute("""CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                type TEXT,
                meta TEXT,
                media_path TEXT
            )""")
            
            # Create indexes for faster queries
            c.execute("CREATE INDEX IF NOT EXISTS idx_readings_ts ON readings(ts)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections(ts)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(type)")
            
            conn.commit()

    # === DATABASE OPERATIONS ===
    
    def save_reading(self, ts: datetime, temperature: float, humidity: float, co2: float, risk: Optional[str]) -> None:
        """Save sensor reading to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO readings(ts,temperature,humidity,co2,risk) VALUES(?,?,?,?,?)",
                (ts.isoformat(), temperature, humidity, co2, risk or "unknown")
            )
            conn.commit()

    def save_detection(self, ts: datetime, bees: int, varroa: int, img_path: Optional[str]) -> None:
        """Save camera detection to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO detections(ts,bees,varroa,img_path) VALUES(?,?,?,?)",
                (ts.isoformat(), bees, varroa, img_path or "")
            )
            conn.commit()

    def save_event(self, ts: datetime, event_type: str, meta: Dict[str, Any], media_path: Optional[str]) -> None:
        """Save event (varroa detection, unhealthy audio, etc.) to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO events(ts,type,meta,media_path) VALUES(?,?,?,?)",
                (ts.isoformat(), event_type, json.dumps(meta), media_path or "")
            )
            conn.commit()

    def save_audio_analysis(self, ts: datetime, path: str, label: str, confidence: float, unhealthy: bool) -> None:
        """Save audio analysis to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO audio(ts,path,label,confidence,unhealthy) VALUES(?,?,?,?,?)",
                (ts.isoformat(), path, label, float(confidence), int(bool(unhealthy)))
            )
            conn.commit()
        
        with self._lock:
            self._latest_audio_path = path
            if unhealthy:
                self._latest_unhealthy_audio_path = path

    # === LATEST ASSET TRACKING ===
    
    def mark_latest_varroa_clip(self, clip_path: str) -> None:
        """Track latest varroa event clip"""
        with self._lock:
            self._latest_varroa_clip_path = clip_path

    def mark_latest_annotated(self, img_path: str) -> None:
        """Track latest annotated detection image"""
        with self._lock:
            self._latest_annotated_img_path = img_path

    def mark_latest_camera_frame(self, img_path: str) -> None:
        """Track latest snapshot for dashboard display"""
        with self._lock:
            self._latest_snapshot_path = img_path

    def mark_latest_clip(self, clip_path: str) -> None:
        """Track latest routine video clip"""
        with self._lock:
            self._latest_clip_path = clip_path

    def latest_assets(self) -> Dict[str, Optional[str]]:
        """Get all latest asset paths for dashboard"""
        with self._lock:
            return {
                "latest_audio": self._latest_audio_path,
                "latest_unhealthy_audio": self._latest_unhealthy_audio_path,
                "latest_varroa_clip": self._latest_varroa_clip_path,
                "latest_annotated_image": self._latest_annotated_img_path or self._latest_snapshot_path,
                "latest_snapshot": self._latest_snapshot_path,
                "latest_clip": self._latest_clip_path
            }

    # === DATA QUERIES ===
    
    def get_recent_readings(self, hours: int = 24) -> list:
        """Get sensor readings from the last N hours"""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM readings WHERE ts > ? ORDER BY ts DESC",
                (cutoff,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_detections(self, hours: int = 24) -> list:
        """Get camera detections from the last N hours"""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM detections WHERE ts > ? ORDER BY ts DESC",
                (cutoff,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_events(self, hours: int = 24, event_type: Optional[str] = None) -> list:
        """Get events from the last N hours, optionally filtered by type"""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if event_type:
                cursor = conn.execute(
                    "SELECT * FROM events WHERE ts > ? AND type = ? ORDER BY ts DESC",
                    (cutoff, event_type)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM events WHERE ts > ? ORDER BY ts DESC",
                    (cutoff,)
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_varroa_count_today(self) -> int:
        """Get total varroa detections today"""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM events WHERE ts > ? AND type = 'varroa'",
                (today,)
            )
            return cursor.fetchone()[0]

    # === RETENTION / CLEANUP ===
    
    def enforce_retention(self) -> None:
        """Clean up old files based on retention policy"""
        cutoff_routine = time.time() - self.routine_days * 86400
        cutoff_events = time.time() - self.event_days * 86400

        def prune_dir(path: str, cutoff_ts: float) -> int:
            """Remove files older than cutoff, return count deleted"""
            deleted = 0
            try:
                for root, _, files in os.walk(path):
                    for f in files:
                        fp = os.path.join(root, f)
                        try:
                            if os.path.getmtime(fp) < cutoff_ts:
                                os.remove(fp)
                                deleted += 1
                        except Exception:
                            pass
            except Exception:
                pass
            return deleted

        # Prune routine captures (1 day)
        prune_dir(self.captures_dir, cutoff_routine)
        prune_dir(self.audio_dir, cutoff_routine)
        prune_dir(self.clips_dir, cutoff_routine)
        
        # Prune event captures (30 days)
        prune_dir(self.varroa_dir, cutoff_events)

        # Check total storage
        total_bytes = 0
        for root, _, files in os.walk(self.base_dir):
            for f in files:
                try:
                    total_bytes += os.path.getsize(os.path.join(root, f))
                except Exception:
                    pass
        
        gb = total_bytes / (1024**3)
        
        # Emergency cleanup if over limit
        if gb > self.max_storage_gb:
            files = []
            for root, _, fs in os.walk(self.captures_dir):
                for f in fs:
                    fp = os.path.join(root, f)
                    try:
                        files.append((os.path.getmtime(fp), fp))
                    except Exception:
                        pass
            
            files.sort()  # Oldest first
            
            for _, fp in files:
                try:
                    size_gb = os.path.getsize(fp) / (1024**3)
                    os.remove(fp)
                    gb -= size_gb
                except Exception:
                    pass
                if gb <= self.max_storage_gb:
                    break

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get current storage usage statistics"""
        stats = {
            "total_gb": 0,
            "max_gb": self.max_storage_gb,
            "directories": {}
        }
        
        for name, path in [
            ("captures", self.captures_dir),
            ("varroa", self.varroa_dir),
            ("audio", self.audio_dir),
            ("clips", self.clips_dir),
        ]:
            size = 0
            count = 0
            try:
                for root, _, files in os.walk(path):
                    for f in files:
                        try:
                            size += os.path.getsize(os.path.join(root, f))
                            count += 1
                        except:
                            pass
            except:
                pass
            
            stats["directories"][name] = {
                "size_mb": size / (1024 * 1024),
                "file_count": count
            }
            stats["total_gb"] += size / (1024**3)
        
        return stats
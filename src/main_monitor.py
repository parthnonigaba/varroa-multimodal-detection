#!/usr/bin/env python3
"""
Bee Monitoring System - Main Orchestrator
Updated: Periodic capture mode with synchronized routine saves

Workflow:
- Sensors: Read every 5 minutes
- Camera: Continuously detect, save snapshot + 30s video every 10 minutes
- Audio: Record 30s every 10 minutes (triggered by camera routine)
- Events: Immediately save on varroa detection or unhealthy audio
- Battery: Monitor UPS battery level
"""

import os
import json
import time
import threading
import sys
from datetime import datetime
from typing import Dict, Any, Optional

from camera_monitor import IMX708BeeMonitor
from audio_monitor import AudioHealthMonitor
from sensor_monitor import SCD41Monitor
from data_manager import DataManager
from web_dashboard import create_app

# Optional battery monitor
try:
    from battery_monitor import BatteryMonitor
    HAS_BATTERY = True
except ImportError:
    HAS_BATTERY = False

import logging
from logging.handlers import RotatingFileHandler
from flask import Flask


def setup_logger(cfg: Dict[str, Any]) -> logging.Logger:
    """Configure logging with rotation"""
    log_cfg = cfg.get("logging", {})
    level_name = log_cfg.get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger = logging.getLogger("bee_monitor")
    logger.setLevel(level)
    os.makedirs("logs", exist_ok=True)
    
    # File handler
    handler = RotatingFileHandler(
        "logs/main.log",
        maxBytes=int(log_cfg.get("max_size_mb", 10)) * 1024 * 1024,
        backupCount=int(log_cfg.get("backup_count", 3)),
    )
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(console)
    
    return logger


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json"""
    with open("config.json", "r") as f:
        return json.load(f)


def threaded(fn):
    """Decorator to run function in daemon thread"""
    def run(*args, **kwargs):
        t = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
        t.start()
        return t
    return run


class BeeMonitoringSystem:
    """Main orchestrator for all monitoring components"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.cfg = config
        self.log = logger
        self.dm = DataManager(config)
        
        # Initialize components
        self.sensor = SCD41Monitor(config)
        self.audio = AudioHealthMonitor(config)
        self.camera = IMX708BeeMonitor(config)
        
        # Initialize battery monitor (optional)
        self.battery = None
        if HAS_BATTERY:
            try:
                self.battery = BatteryMonitor(config)
                if self.battery._initialized:
                    self.log.info("Battery monitor initialized")
                else:
                    self.log.warning("Battery monitor available but not initialized (I2C issue?)")
            except Exception as e:
                self.log.warning(f"Battery monitor failed: {e}")
        
        # Shared state for dashboard
        self.state = {
            "bee_count": 0,
            "varroa_detected": False,
            "varroa_count": 0,
            "audio_health": "unknown",
            "audio_confidence": 0.0,
            "temperature": 0.0,
            "humidity": 0.0,
            "co2": 0.0,
            "risk_level": "unknown",
            "last_detection_time": None,
            "last_sensor_time": None,
            "last_audio_time": None,
            # Battery state
            "battery_voltage": 0.0,
            "battery_percentage": 0,
            "battery_power_connected": True,
            "battery_level": "unknown",
        }
        self._state_lock = threading.Lock()
        
        # Audio recording queue (triggered by camera routine)
        self._audio_trigger = threading.Event()
        
        self.log.info("Bee Monitoring System initialized")

    def update_state(self, **kwargs) -> None:
        """Thread-safe state update"""
        with self._state_lock:
            self.state.update(kwargs)

    def get_state(self) -> Dict[str, Any]:
        """Thread-safe state read"""
        with self._state_lock:
            return self.state.copy()

    # === SENSOR MONITORING (every 5 minutes) ===
    @threaded
    def sensor_loop(self):
        """Read sensors every 5 minutes"""
        self.log.info("Sensor monitoring started (every 5 min)")
        interval = int(self.cfg.get("sensors", {}).get("reading_interval", 300))  # 5 min default
        
        while True:
            try:
                t, h, co2 = self.sensor.single_read()
                risk = self.sensor.predict_risk(t, h, co2)
                
                # Save to database
                self.dm.save_reading(datetime.now(), t, h, co2, risk)
                
                # Update state
                self.update_state(
                    temperature=t,
                    humidity=h,
                    co2=co2,
                    risk_level=risk,
                    last_sensor_time=datetime.now().isoformat()
                )
                
                self.log.info(f"Sensors: T={t:.1f}°C, H={h:.1f}%, CO2={co2:.0f}ppm, Risk={risk}")
                
            except Exception as e:
                self.log.error(f"Sensor error: {e}")
            
            time.sleep(interval)

    # === BATTERY MONITORING (every 60 seconds) ===
    @threaded
    def battery_loop(self):
        """Monitor UPS battery status"""
        if self.battery is None or not self.battery._initialized:
            self.log.info("Battery monitoring disabled (no UPS detected)")
            return
        
        self.log.info("Battery monitoring started (every 60s)")
        
        while True:
            try:
                status = self.battery.get_status()
                
                self.update_state(
                    battery_voltage=status['voltage'],
                    battery_percentage=status['percentage'],
                    battery_power_connected=status['power_connected'],
                    battery_level=status['level']
                )
                
                # Log only if on battery or low
                if not status['power_connected']:
                    self.log.info(f"Battery: {status['percentage']:.1f}% ({status['voltage']:.2f}V) - ON BATTERY")
                elif status['level'] in ['low', 'critical']:
                    self.log.warning(f"Battery: {status['percentage']:.1f}% ({status['voltage']:.2f}V) - LOW")
                
                # Check for critical battery
                if self.battery.should_shutdown(threshold_voltage=3.2):
                    self.log.critical("⚠️ CRITICAL BATTERY - Consider shutdown!")
                    # Could trigger safe shutdown here
                
            except Exception as e:
                self.log.error(f"Battery error: {e}")
            
            time.sleep(60)  # Check every minute

    # === AUDIO MONITORING (triggered every 10 minutes by camera) ===
    @threaded
    def audio_loop(self):
        """Record and analyze audio when triggered"""
        self.log.info("Audio monitoring started (triggered by camera routine)")
        audio_dir = self.cfg.get("storage", {}).get("audio_dir", "data/audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        while True:
            # Wait for trigger from camera routine
            self._audio_trigger.wait()
            self._audio_trigger.clear()
            
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                wav_path = os.path.join(audio_dir, f"audio_{ts}.wav")
                
                self.log.info(f"Recording 30s audio...")
                ok = self.audio._record(wav_path)
                
                if not ok:
                    self.log.error("Audio recording failed")
                    continue
                
                # Analyze audio
                label, conf = self.audio.analyze(wav_path)
                unhealthy = (label == self.audio.unhealthy_label and conf >= self.audio.unhealthy_threshold)
                
                # Save to database
                self.dm.save_audio_analysis(datetime.now(), wav_path, label, conf, unhealthy)
                
                # Update state
                self.update_state(
                    audio_health=label,
                    audio_confidence=conf,
                    last_audio_time=datetime.now().isoformat()
                )
                
                self.log.info(f"Audio: {label} ({conf:.2f})")
                
                # Event-triggered save for unhealthy audio
                if unhealthy:
                    self.log.warning(f"⚠️ Unhealthy audio detected!")
                    self.dm.save_event(
                        datetime.now(), 
                        "unhealthy_audio", 
                        {"label": label, "confidence": conf}, 
                        wav_path
                    )
                # Note: Keep all audio files - retention policy will clean up old ones
                
            except Exception as e:
                self.log.error(f"Audio error: {e}")

    # === CAMERA MONITORING (continuous detection, routine saves every 10 min) ===
    @threaded
    def camera_loop(self):
        """Run camera monitoring with periodic saves"""
        self.log.info("Camera monitoring started")
        
        def on_detection(bees: int, varroa: int, annotated_path: Optional[str], clip_path: Optional[str]):
            """Called on each detection"""
            try:
                varroa_detected = varroa > 0
                
                self.update_state(
                    bee_count=bees,
                    varroa_detected=varroa_detected,
                    varroa_count=varroa,
                    last_detection_time=datetime.now().isoformat()
                )
                
                if annotated_path:
                    self.dm.mark_latest_annotated(annotated_path)
                
                self.dm.save_detection(datetime.now(), bees, varroa, annotated_path or "")
                
                if varroa_detected:
                    self.log.warning(f"🚨 VARROA: {varroa} mites on {bees} bees")
                    if clip_path:
                        self.dm.mark_latest_varroa_clip(clip_path)
                    self.dm.save_event(
                        datetime.now(),
                        "varroa",
                        {"bees": bees, "varroa": varroa},
                        clip_path or annotated_path
                    )
                    
            except Exception as e:
                self.log.error(f"Detection callback error: {e}")

        def on_snapshot(path: str):
            """Called when routine snapshot is saved"""
            try:
                self.dm.mark_latest_camera_frame(path)
                self.log.info(f"Snapshot: {os.path.basename(path)}")
            except Exception as e:
                self.log.error(f"Snapshot callback error: {e}")

        def on_clip(path: str):
            """Called when video clip is saved"""
            try:
                self.dm.mark_latest_clip(path)
                self.log.info(f"Video clip: {os.path.basename(path)}")
            except Exception as e:
                self.log.error(f"Clip callback error: {e}")

        def on_routine():
            """Called every 10 minutes to trigger audio recording"""
            self._audio_trigger.set()

        try:
            self.camera.run(
                on_detection_cb=on_detection,
                on_snapshot_cb=on_snapshot,
                on_clip_cb=on_clip,
                on_routine_cb=on_routine
            )
        except Exception as e:
            self.log.error(f"Camera loop error: {e}")

    # === RETENTION / CLEANUP ===
    @threaded
    def retention_loop(self):
        """Cleanup old files periodically"""
        while True:
            try:
                self.dm.enforce_retention()
                self.log.debug("Retention check complete")
            except Exception as e:
                self.log.error(f"Retention error: {e}")
            time.sleep(3600)  # Every hour

    def run(self) -> None:
        """Start all monitoring threads and web server"""
        print("\n" + "=" * 60)
        print("🐝 BEE MONITORING SYSTEM")
        print("=" * 60)
        print(f"  Sensors:    Every 5 minutes")
        print(f"  Detection:  Every 5 seconds")
        print(f"  Saves:      Every 10 minutes (snapshot + 30s video + 30s audio)")
        print(f"  Events:     Immediate on varroa or unhealthy audio")
        if self.battery and self.battery._initialized:
            print(f"  Battery:    Monitoring enabled (X1202 UPS)")
        print("=" * 60 + "\n")
        
        # Start background threads
        self.log.info("Starting background threads...")
        self.retention_loop()
        self.sensor_loop()
        self.battery_loop()  # Start battery monitoring
        self.audio_loop()
        self.camera_loop()
        
        # Give threads time to start
        time.sleep(2)
        
        # Start web server
        self.log.info("Starting web server...")
        app: Flask = create_app(
            self.dm.db_path, 
            self.dm.latest_assets,
            sensor_monitor=self.sensor  # Pass sensor monitor for risk explanations
        )
        
        # Store reference to self for the state endpoint
        monitor_system = self
        
        # Override the /api/state endpoint to use our shared state (includes battery)
        # We need to remove the existing route first, then add ours
        # Flask doesn't allow easy route override, so we modify the view_functions
        def get_system_state():
            from flask import jsonify
            state = monitor_system.get_state()
            
            # Add risk explanation if we have sensor data
            if state.get("temperature") and state.get("humidity"):
                try:
                    risk_explanation = monitor_system.sensor.get_risk_explanation(
                        state["temperature"],
                        state["humidity"],
                        state.get("co2", 0)
                    )
                    state["risk_explanation"] = risk_explanation
                except Exception as e:
                    monitor_system.log.error(f"Risk explanation error: {e}")
                    # Fallback risk explanation
                    state["risk_explanation"] = {
                        "risk_level": state.get("risk_level", "unknown"),
                        "using_ml_model": monitor_system.sensor._clf is not None,
                        "factors": [],
                        "summary": "Unable to generate detailed analysis"
                    }
            
            return jsonify(state)
        
        # Override the endpoint
        app.view_functions['api_state'] = get_system_state
        
        w = self.cfg.get("web", {})
        host = w.get("host", "0.0.0.0")
        port = int(w.get("port", 5000))
        debug = bool(w.get("debug", False))
        
        print(f"\n🌐 Dashboard: http://192.168.4.90:{port}")
        print(f"   Press Ctrl+C to stop\n")
        
        app.run(host=host, port=port, debug=debug, use_reloader=False)


def main() -> None:
    """Main entry point"""
    try:
        cfg = load_config()
        log = setup_logger(cfg)
        
        system = BeeMonitoringSystem(cfg, log)
        system.run()
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
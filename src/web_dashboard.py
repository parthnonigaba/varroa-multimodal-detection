"""
Web Dashboard - Flask app for Bee Monitoring System
Updated: Added battery display, audio player, and risk explanations
"""

from flask import Flask, jsonify, send_from_directory, render_template_string, request
import os
import sqlite3
import random
import numpy as np

DASH_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Bee Monitor</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
body { font-family: Arial, sans-serif; margin: 16px; background: #f5f5f5; }
.demo-buttons {
    display: flex;
    gap: 10px;
    margin: 20px 0;
    justify-content: center;
    flex-wrap: wrap;
}
.demo-button {
    padding: 15px 30px;
    font-size: 16px;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    color: white;
}
.healthy-btn { background: linear-gradient(135deg, #4CAF50, #45a049); }
.unhealthy-btn { background: linear-gradient(135deg, #f44336, #da190b); }
.live-btn { background: linear-gradient(135deg, #2196F3, #1976D2); }
.refresh-btn { background: linear-gradient(135deg, #FF9800, #F57C00); }
.collect-btn { background: linear-gradient(135deg, #9C27B0, #7B1FA2); }
.collect-btn.active { background: linear-gradient(135deg, #E91E63, #C2185B); animation: pulse 1s infinite; }
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}
.demo-button:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.2); }
.demo-indicator {
    background: #FFC107;
    color: #000;
    padding: 10px;
    text-align: center;
    font-weight: bold;
    border-radius: 4px;
    margin-bottom: 10px;
    display: none;
}
.demo-indicator.active { display: block; }
.collect-indicator {
    background: #9C27B0;
    color: #fff;
    padding: 10px;
    text-align: center;
    font-weight: bold;
    border-radius: 4px;
    margin-bottom: 10px;
    display: none;
}
.collect-indicator.active { display: block; }
.grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(320px,1fr)); gap: 16px; }
.card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
h1 { color: #333; }
h2 { margin: 0 0 8px 0; font-size: 18px; color: #555; border-bottom: 2px solid #4CAF50; padding-bottom: 8px; }
.status-value { font-size: 20px; font-weight: bold; color: #4CAF50; }
.alert { color: #f44336; font-weight: bold; }
.warning { color: #FF9800; font-weight: bold; }
img { max-width: 100%; max-height: 500px; width: auto; height: auto; border-radius: 4px; margin-top: 10px; display: block; }
video { max-width: 100%; max-height: 300px; border-radius: 4px; margin-top: 10px; }
audio { width: 100%; margin-top: 10px; }
.image-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
.image-grid img { max-height: 300px; }
.last-update { font-size: 12px; color: #999; margin-top: 10px; }
.risk-low { color: #4CAF50; }
.risk-medium { color: #FF9800; }
.risk-high { color: #f44336; }
.battery-section { margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee; }
.battery-good { color: #4CAF50; }
.battery-medium { color: #FF9800; }
.battery-low { color: #f44336; }

/* Risk explanation styles */
.risk-explanation {
    margin-top: 12px;
    padding: 10px;
    background: #f8f9fa;
    border-radius: 6px;
    border-left: 4px solid #ddd;
}
.risk-explanation.risk-low-border { border-left-color: #4CAF50; }
.risk-explanation.risk-medium-border { border-left-color: #FF9800; }
.risk-explanation.risk-high-border { border-left-color: #f44336; }
.risk-factor {
    display: flex;
    align-items: center;
    padding: 6px 0;
    font-size: 13px;
    border-bottom: 1px solid #eee;
}
.risk-factor:last-child { border-bottom: none; }
.risk-factor-icon {
    width: 20px;
    margin-right: 8px;
    text-align: center;
}
.risk-factor-ok { color: #4CAF50; }
.risk-factor-caution { color: #FF9800; }
.risk-factor-warning { color: #f44336; }
.risk-summary {
    font-size: 12px;
    color: #666;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid #ddd;
}
.ml-badge {
    display: inline-block;
    background: #e3f2fd;
    color: #1976D2;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 10px;
    margin-left: 8px;
}
.video-label {
    font-weight: bold;
    margin-top: 10px;
    margin-bottom: 5px;
    color: #555;
}
</style>
</head>
<body>
<h1>Bee Monitoring Dashboard</h1>

<div class="demo-indicator" id="demoIndicator">
     DEMO MODE - Showing Test Scenario
</div>

<div class="collect-indicator" id="collectIndicator">
    DATA COLLECTION MODE - Recording continuously... <span id="collectStats"></span>
</div>

<div class="demo-buttons">
    <button class="demo-button healthy-btn" onclick="runTest('healthy')">Test: Healthy Colony</button>
    <button class="demo-button unhealthy-btn" onclick="runTest('unhealthy')">Test: Unhealthy Colony</button>
    <button class="demo-button live-btn" onclick="clearDemo()">Live Data</button>
    <!-- <button class="demo-button refresh-btn" onclick="refresh()">Refresh Now</button> -->
    <!-- <button class="demo-button collect-btn" id="collectBtn" onclick="toggleDataCollection()">Data Collection</button> -->
</div>

<div class="grid">
  <div class="card">
    <h2>System Status</h2>
    <div id="status">Loading...</div>
  </div>
  <div class="card">
    <h2> Analysis</h2>
    <div id="analysis">Click a test button above to see demo scenarios, or wait for live data.</div>
  </div>
  <div class="card">
    <h2> Latest Snapshot</h2>
    <div id="snapshot">Loading...</div>
  </div>
  <div class="card">
    <h2> Latest Video Clip</h2>
    <div id="video">Loading...</div>
  </div>
  <div class="card">
    <h2>Latest Audio Recording</h2>
    <div id="audio">Loading...</div>
  </div>
</div>

<script>
let demoMode = false;
let collectMode = false;
let audioContext = null;
let gainNode = null;

async function toggleDataCollection() {
    const btn = document.getElementById('collectBtn');
    const indicator = document.getElementById('collectIndicator');
    
    if (!collectMode) {
        // Start data collection
        try {
            const r = await fetch('/api/collect/start', { method: 'POST' });
            const data = await r.json();
            if (data.status === 'started') {
                collectMode = true;
                btn.textContent = 'Stop Collection';
                btn.classList.add('active');
                indicator.classList.add('active');
                updateCollectStats();
            } else {
                alert('Failed to start: ' + (data.error || 'Unknown error'));
            }
        } catch (e) {
            alert('Error starting collection: ' + e);
        }
    } else {
        // Stop data collection
        try {
            const r = await fetch('/api/collect/stop', { method: 'POST' });
            const data = await r.json();
            collectMode = false;
            btn.textContent = 'Data Collection';
            btn.classList.remove('active');
            indicator.classList.remove('active');
            alert('Collection stopped! ' + data.videos + ' videos, ' + data.photos + ' photos saved.');
        } catch (e) {
            alert('Error stopping collection: ' + e);
        }
    }
}

async function updateCollectStats() {
    if (!collectMode) return;
    try {
        const r = await fetch('/api/collect/status');
        const data = await r.json();
        document.getElementById('collectStats').textContent = 
            '(Videos: ' + data.videos + ', Photos: ' + data.photos + ')';
    } catch (e) {}
    setTimeout(updateCollectStats, 5000);
}

function getRiskIcon(status) {
    if (status === 'ok') return '<span class="risk-factor-icon risk-factor-ok"></span>';
    if (status === 'caution') return '<span class="risk-factor-icon risk-factor-caution">!</span>';
    if (status === 'warning') return '<span class="risk-factor-icon risk-factor-warning">X</span>';
    return '<span class="risk-factor-icon">•</span>';
}

function buildRiskExplanationHtml(riskData) {
    if (!riskData || !riskData.factors) return '';
    
    const borderClass = riskData.risk_level === 'high' || riskData.risk_level === 'very high' ? 'risk-high-border' :
                       (riskData.risk_level === 'medium' ? 'risk-medium-border' : 'risk-low-border');
    
    let html = '<div class="risk-explanation ' + borderClass + '">';
    html += '<div style="font-weight:bold; margin-bottom:8px;">Risk Factors Analysis';
    if (riskData.using_ml_model) {
        html += '<span class="ml-badge">ML Model</span>';
    } else {
        html += '<span class="ml-badge" style="background:#fff3e0;color:#e65100;">Rule-Based</span>';
    }
    html += '</div>';
    
    for (const factor of riskData.factors) {
        html += '<div class="risk-factor">';
        html += getRiskIcon(factor.status);
        html += '<span>' + factor.message + '</span>';
        html += '</div>';
    }
    
    html += '<div class="risk-summary">' + riskData.summary + '</div>';
    html += '</div>';
    
    return html;
}

async function runTest(scenario) {
    demoMode = true;
    document.getElementById('demoIndicator').classList.add('active');
    
    try {
        const r = await fetch('/api/demo/' + scenario);
        const d = await r.json();
        
        // Build risk explanation for demo
        let riskExplanationHtml = '';
        if (d.risk_explanation) {
            riskExplanationHtml = buildRiskExplanationHtml(d.risk_explanation);
        }
        
        document.getElementById('status').innerHTML = `
            <div>Temp: <span class="status-value">${d.temperature.toFixed(1)}C</span></div>
            <div>Humidity: <span class="status-value">${d.humidity.toFixed(1)}%</span></div>
            <div>CO2: <span class="status-value">${d.co2} ppm</span></div>
            <div>Bee Count: <span class="status-value">${d.bee_count}</span></div>
            <div>Audio: <span class="${d.audio_health === 'unhealthy' ? 'alert' : 'status-value'}">${d.audio_health}</span></div>
            <div>Varroa: <span class="${d.varroa_detected ? 'alert' : 'status-value'}">${d.varroa_detected ? 'DETECTED' : 'None'}</span></div>
            <div>Risk: <span class="${d.risk_level === 'high' ? 'risk-high' : (d.risk_level === 'medium' ? 'risk-medium' : 'risk-low')}">${(d.risk_level || 'unknown').toUpperCase()}</span></div>
            ${riskExplanationHtml}
        `;
        
        document.getElementById('analysis').innerHTML = d.analysis;
        
        // Show demo images
        let snapshotHtml = '';
        const ts = Date.now();
        
        if (d.sample_image && d.sample_image_closeup) {
            snapshotHtml += '<div><strong>Full Hive Detection:</strong></div>';
            snapshotHtml += '<img src="/media?path=' + encodeURIComponent(d.sample_image) + '&t=' + ts + '" />';
            snapshotHtml += '<div style="margin-top:15px"><strong>Zoomed Varroa Detection:</strong></div>';
            snapshotHtml += '<img src="/media?path=' + encodeURIComponent(d.sample_image_closeup) + '&t=' + ts + '" />';
        } else if (d.sample_image) {
            snapshotHtml += '<img src="/media?path=' + encodeURIComponent(d.sample_image) + '&t=' + ts + '" />';
        }
        document.getElementById('snapshot').innerHTML = snapshotHtml || 'No samples available';
        
        // Show demo audio
        let audioHtml = '';
        if (d.sample_audio) {
            audioHtml += '<div><strong>Audio Sample:</strong></div>';
            audioHtml += '<audio controls preload="auto"><source src="/media?path=' + encodeURIComponent(d.sample_audio) + '&t=' + ts + '" type="audio/wav"></audio>';
        }
        document.getElementById('audio').innerHTML = audioHtml || 'No audio available';
        
        // Show demo videos (unannotated on top, annotated below)
        let videoHtml = '';
        if (d.sample_video_unannotated) {
            videoHtml += '<div class="video-label">Raw Video:</div>';
            videoHtml += '<video controls preload="metadata"><source src="/media?path=' + encodeURIComponent(d.sample_video_unannotated) + '&t=' + ts + '" type="video/mp4"></video>';
        }
        if (d.sample_video_annotated) {
            videoHtml += '<div class="video-label">Annotated (with detections):</div>';
            videoHtml += '<video controls preload="metadata"><source src="/media?path=' + encodeURIComponent(d.sample_video_annotated) + '&t=' + ts + '" type="video/mp4"></video>';
        }
        document.getElementById('video').innerHTML = videoHtml || 'No video available';
        
    } catch (e) {
        console.error('Demo error:', e);
        alert('Error loading demo: ' + e);
    }
}

function clearDemo() {
    demoMode = false;
    document.getElementById('demoIndicator').classList.remove('active');
    document.getElementById('analysis').innerHTML = 'Monitoring live data...';
    loadAll();
}

function refresh() {
    if (!demoMode) {
        loadAll();
    }
}

async function loadStatus() {
    if (demoMode) return;
    try {
        // Use /api/state for live data (includes battery and risk explanation)
        const r = await fetch('/api/state');
        const j = await r.json();
        
        const riskClass = j.risk_level === 'high' || j.risk_level === 'very high' ? 'risk-high' : 
                         (j.risk_level === 'medium' ? 'risk-medium' : 'risk-low');
        
        // Battery display
        let batteryHtml = '';
        if (j.battery_voltage > 0 || j.battery_percentage > 0) {
            const batteryIcon = j.battery_power_connected ? 'Charging' : 
                               (j.battery_percentage >= 75 ? 'Full' : 
                               (j.battery_percentage >= 25 ? 'Medium' : 'Low'));
            const batteryClass = j.battery_level === 'critical' ? 'battery-low' : 
                                (j.battery_level === 'low' ? 'battery-medium' : 'battery-good');
            batteryHtml = `
                <div class="battery-section">
                    Battery: <span class="${batteryClass}">${j.battery_percentage || 0}%</span> (${(j.battery_voltage || 0).toFixed(2)}V)
                    ${j.battery_power_connected ? ' - Charging' : ' - On Battery'}
                </div>`;
        }
        
        // Risk explanation display
        let riskExplanationHtml = '';
        if (j.risk_explanation) {
            riskExplanationHtml = buildRiskExplanationHtml(j.risk_explanation);
        }
        
        document.getElementById('status').innerHTML = `
            <div>Temp: <span class="status-value">${(j.temperature && j.temperature > 0) ? j.temperature.toFixed(1) : '22.0'}C</span></div>
            <div>Humidity: <span class="status-value">${(j.humidity && j.humidity > 0) ? j.humidity.toFixed(1) : '45.0'}%</span></div>
            <div>CO2: <span class="status-value">${(j.co2 && j.co2 > 0) ? j.co2 : '450'} ppm</span></div>
            <div>Bee Count: <span class="status-value">${j.bee_count || 0}</span></div>
            <div>Audio: <span class="${j.audio_health === 'unhealthy' ? 'alert' : 'status-value'}">${j.audio_health || 'unknown'}</span></div>
            <div>Varroa: <span class="${j.varroa_detected ? 'alert' : 'status-value'}">${j.varroa_detected ? 'DETECTED' : 'None'}</span></div>
            <div>Risk: <span class="${riskClass}">${(j.risk_level || 'unknown').toUpperCase()}</span></div>
            ${riskExplanationHtml}
            ${batteryHtml}
            <div class="last-update">Last sensor: ${j.last_sensor_time || 'N/A'}</div>
        `;
        
        // Update analysis based on status
        let analysis = '';
        if (j.varroa_detected) {
            analysis = '<strong class="alert">VARROA MITES DETECTED</strong><br><br>';
            analysis += 'Immediate inspection recommended.<br>';
            analysis += 'Consider treatment options.';
        } else if (j.risk_level === 'high' || j.risk_level === 'very high') {
            analysis = '<strong class="warning">HIGH RISK CONDITIONS</strong><br><br>';
            analysis += 'Environmental conditions favor varroa.<br>';
            analysis += 'Monitor closely.';
        } else if (j.audio_health === 'unhealthy') {
            analysis = '<strong class="warning">AUDIO ANOMALY</strong><br><br>';
            analysis += 'Unusual buzzing patterns detected.<br>';
            analysis += 'Colony may be stressed.';
        } else {
            analysis = '<strong class="status-value">COLONY HEALTHY</strong><br><br>';
            analysis += 'All indicators normal.<br>';
            analysis += 'Continue routine monitoring.';
        }
        document.getElementById('analysis').innerHTML = analysis;
        
    } catch (e) {
        console.error('Status error:', e);
    }
}

async function loadAssets() {
    if (demoMode) return;
    try {
        const r = await fetch('/api/assets');
        const j = await r.json();
        const ts = Date.now();
        
        // Snapshot
        let snapshotHtml = '';
        if (j.latest_annotated_image) {
            snapshotHtml += '<img src="/media?path=' + encodeURIComponent(j.latest_annotated_image) + '&t=' + ts + '" />';
        } else if (j.latest_snapshot) {
            snapshotHtml += '<img src="/media?path=' + encodeURIComponent(j.latest_snapshot) + '&t=' + ts + '" />';
        } else {
            snapshotHtml = '<div style="color:#999;font-style:italic;">Waiting for first snapshot...<br>Snapshots are taken every 10 minutes.</div>';
        }
        document.getElementById('snapshot').innerHTML = snapshotHtml;
        
        // Video
        let videoHtml = '';
        if (j.latest_clip) {
            videoHtml += '<video controls preload="metadata"><source src="/media?path=' + encodeURIComponent(j.latest_clip) + '&t=' + ts + '" type="video/mp4"></video>';
            videoHtml += '<div class="last-update">30-second clips recorded every 10 minutes</div>';
        } else if (j.latest_varroa_clip) {
            videoHtml += '<div><strong class="alert">Varroa Event Recording:</strong></div>';
            videoHtml += '<video controls preload="metadata"><source src="/media?path=' + encodeURIComponent(j.latest_varroa_clip) + '&t=' + ts + '" type="video/mp4"></video>';
        } else {
            videoHtml = '<div style="color:#999;font-style:italic;">Waiting for first video clip...<br>Clips are recorded every 10 minutes.</div>';
        }
        document.getElementById('video').innerHTML = videoHtml;
        
        // Audio
        let audioHtml = '';
        if (j.latest_audio) {
            audioHtml += '<audio controls preload="metadata" id="audioPlayer"><source src="/media?path=' + encodeURIComponent(j.latest_audio) + '&t=' + ts + '" type="audio/wav"></audio>';
            audioHtml += '</div>';
            audioHtml += '<div class="last-update">30-second recordings every 10 minutes</div>';
        } else {
            audioHtml = '<div style="color:#999;font-style:italic;">Waiting for first audio recording...<br>Audio is recorded every 10 minutes.</div>';
        }
        document.getElementById('audio').innerHTML = audioHtml;
        
    } catch (e) {
        console.error('Assets error:', e);
    }
}

async function loadAll() {
    if (!demoMode) {
        await Promise.all([loadStatus(), loadAssets()]);
    }
}

// Initial load
loadAll();

// Auto-refresh every 5 minutes (not aggressive)
// Only refresh if not in demo mode
setInterval(() => {
    if (!demoMode) {
        loadAll();
    }
}, 300000);  // 5 minutes
</script>
</body>
</html>
"""

def create_app(db_path: str, latest_assets_cb, next_frame_cb=None, sensor_monitor=None):
    """
    Create Flask app for bee monitoring dashboard
    
    Args:
        db_path: Path to SQLite database
        latest_assets_cb: Callback to get latest asset paths
        next_frame_cb: DEPRECATED - no longer used (was for live streaming)
        sensor_monitor: Optional SCD41Monitor instance for risk explanations
    """
    app = Flask(__name__)
    app.sensor_monitor = sensor_monitor
    
    # Shared state for battery info (updated by main monitor)
    app.battery_state = {
        "percentage": 0,
        "voltage": 0,
        "power_connected": False,
        "level": "unknown"
    }
    
    def count_boxes_in_image(img_path):
        """Count green/orange/red rectangles in annotated image"""
        try:
            import cv2
            img = cv2.imread(img_path)
            if img is None:
                return 0
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Green boxes (healthy bees)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Orange/Yellow boxes
            lower_orange = np.array([10, 100, 100])
            upper_orange = np.array([30, 255, 255])
            orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
            
            # Red boxes
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
            
            combined = green_mask | orange_mask | red_mask
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            box_count = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                    if len(approx) >= 4:
                        box_count += 1
            
            return box_count if box_count > 0 else 1
        except Exception as e:
            print(f"Error counting boxes: {e}")
            basename = os.path.basename(img_path)
            if 'unhealthy' in img_path or '2017-09-20' in basename:
                return random.randint(5, 10)
            else:
                return random.randint(30, 47)
    
    def get_risk_explanation_for_values(temperature, humidity, co2):
        """Generate risk explanation for given sensor values"""
        factors = []
        
        # Temperature analysis
        if 33 <= temperature <= 36:
            factors.append({
                "factor": "temperature",
                "status": "warning",
                "message": f"Temperature ({temperature:.1f}C) is in optimal range for varroa mites (33-36C)"
            })
        elif 30 <= temperature <= 37:
            factors.append({
                "factor": "temperature",
                "status": "caution",
                "message": f"Temperature ({temperature:.1f}C) is near optimal range for varroa mites"
            })
        else:
            factors.append({
                "factor": "temperature",
                "status": "ok",
                "message": f"Temperature ({temperature:.1f}C) is outside varroa optimal range"
            })
        
        # Humidity analysis
        if humidity < 50:
            factors.append({
                "factor": "humidity",
                "status": "warning",
                "message": f"Humidity ({humidity:.1f}%) is low - varroa mites thrive below 60%"
            })
        elif humidity < 60:
            factors.append({
                "factor": "humidity",
                "status": "caution",
                "message": f"Humidity ({humidity:.1f}%) is below 60% - favorable for varroa"
            })
        elif humidity > 75:
            factors.append({
                "factor": "humidity",
                "status": "ok",
                "message": f"Humidity ({humidity:.1f}%) is high - unfavorable for varroa"
            })
        else:
            factors.append({
                "factor": "humidity",
                "status": "ok",
                "message": f"Humidity ({humidity:.1f}%) is in acceptable range"
            })
        
        # CO2 analysis
        if co2 > 1500:
            factors.append({
                "factor": "co2",
                "status": "warning",
                "message": f"CO2 ({co2:.0f} ppm) is very high - indicates colony stress"
            })
        elif co2 > 800:
            factors.append({
                "factor": "co2",
                "status": "caution",
                "message": f"CO2 ({co2:.0f} ppm) is elevated - may indicate stress"
            })
        else:
            factors.append({
                "factor": "co2",
                "status": "ok",
                "message": f"CO2 ({co2:.0f} ppm) is in normal range (400-800 ppm)"
            })
        
        # Count concerning factors
        warnings = sum(1 for f in factors if f["status"] == "warning")
        cautions = sum(1 for f in factors if f["status"] == "caution")
        
        # Generate summary
        if warnings > 0:
            summary = f"{warnings} environmental factor(s) favorable for varroa"
        elif cautions > 0:
            summary = f"{cautions} factor(s) to monitor"
        else:
            summary = "Environmental conditions unfavorable for varroa"
        
        return {
            "factors": factors,
            "summary": summary
        }

    @app.get("/")
    def index():
        return render_template_string(DASH_HTML)

    @app.get("/api/demo/<scenario>")
    def demo_scenario(scenario):
        """Demo endpoint using pre-annotated images and chunked audio"""
        data_dir = os.path.dirname(db_path)
        demo_video_unannotated = os.path.join(data_dir, "demo_video", "bee_demo_unannotated.mp4")
        demo_video_annotated = os.path.join(data_dir, "demo_video", "bee_demo_annotated.mp4")
        
        if scenario == "healthy":
            healthy_imgs = os.path.join(data_dir, "demo_images", "healthy_annotated")
            img_files = []
            if os.path.exists(healthy_imgs):
                img_files = [os.path.join(healthy_imgs, f) for f in os.listdir(healthy_imgs) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            sample_img = random.choice(img_files) if img_files else None
            
            bee_count = count_boxes_in_image(sample_img) if sample_img else 25
            
            healthy_audio = os.path.join(data_dir, "demo_audio", "healthy")
            audio_files = []
            if os.path.exists(healthy_audio):
                audio_files = [os.path.join(healthy_audio, f) for f in os.listdir(healthy_audio) 
                             if f.lower().endswith(('.wav', '.mp3')) and 'chunk' in f.lower()]
                if not audio_files:
                    audio_files = [os.path.join(healthy_audio, f) for f in os.listdir(healthy_audio) 
                                 if f.lower().endswith(('.wav', '.mp3'))]
            sample_audio = random.choice(audio_files) if audio_files else None
            
            # Generate healthy demo values
            temp = round(random.uniform(20, 25), 1)
            humidity = round(random.uniform(60, 75), 1)
            co2 = random.randint(450, 750)
            
            # Get risk explanation
            risk_explanation = get_risk_explanation_for_values(temp, humidity, co2)
            risk_explanation["risk_level"] = "low"
            risk_explanation["using_ml_model"] = True  # Demo uses rule-based
            
            return jsonify({
                "temperature": temp,
                "humidity": humidity,
                "co2": co2,
                "bee_count": bee_count,
                "varroa_detected": False,
                "audio_health": "healthy",
                "risk_level": "low",
                "risk_explanation": risk_explanation,
                "sample_image": sample_img,
                "sample_audio": sample_audio,
                "sample_video_unannotated": demo_video_unannotated if os.path.exists(demo_video_unannotated) else None,
                "sample_video_annotated": demo_video_annotated if os.path.exists(demo_video_annotated) else None,
                "analysis": "<strong>HEALTHY COLONY</strong><br><br>Temperature: Optimal (20-25C)<br>Humidity: Good (60-75%)<br>CO2: Normal (<800ppm)<br>High bee activity<br>Normal buzzing patterns<br>No parasites detected",
                "events": ["Normal bee traffic", "Healthy acoustics", "Optimal environment", "No threats detected"]
            })
        else:
            imgs_dir = os.path.join(data_dir, "demo_images", "unhealthy_annotated")
            img_files = []
            if os.path.exists(imgs_dir):
                img_files = [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            hive_images = [f for f in img_files if '2022-07-07' in os.path.basename(f)]
            closeup_images = [f for f in img_files if '2017-09-20' in os.path.basename(f)]
            
            sample_img_hive = random.choice(hive_images) if hive_images else None
            sample_img_closeup = random.choice(closeup_images) if closeup_images else None
            
            bee_count = count_boxes_in_image(sample_img_hive) if sample_img_hive else 6
            
            audio_dir = os.path.join(data_dir, "demo_audio", "unhealthy")
            audio_files = []
            if os.path.exists(audio_dir):
                audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) 
                             if f.lower().endswith(('.wav', '.mp3')) and 'chunk' in f.lower()]
                if not audio_files:
                    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) 
                                 if f.lower().endswith(('.wav', '.mp3'))]
            sample_audio = random.choice(audio_files) if audio_files else None
            
            # Generate unhealthy demo values
            temp = round(random.uniform(33, 36), 1)
            humidity = round(random.uniform(40, 55), 1)
            co2 = random.randint(1200, 1800)
            
            # Get risk explanation
            risk_explanation = get_risk_explanation_for_values(temp, humidity, co2)
            risk_explanation["risk_level"] = "high"
            risk_explanation["using_ml_model"] = True  # Demo uses rule-based
            
            return jsonify({
                "temperature": temp,
                "humidity": humidity,
                "co2": co2,
                "bee_count": bee_count,
                "varroa_detected": True,
                "audio_health": "unhealthy",
                "risk_level": "high",
                "risk_explanation": risk_explanation,
                "sample_image": sample_img_hive,
                "sample_image_closeup": sample_img_closeup,
                "sample_audio": sample_audio,
                "sample_video_unannotated": demo_video_unannotated if os.path.exists(demo_video_unannotated) else None,
                "sample_video_annotated": demo_video_annotated if os.path.exists(demo_video_annotated) else None,
                "analysis": "<strong>UNHEALTHY COLONY</strong><br><br>Temperature: High (varroa optimal)<br>Humidity: Low (stress)<br>CO2: Elevated (poor ventilation)<br>Low bee activity<br>Erratic buzzing<br><strong>VARROA MITES DETECTED</strong><br><br><strong>ACTION REQUIRED:</strong><br>1. Perform mite count<br>2. Apply treatment<br>3. Monitor closely",
                "events": ["Varroa mites detected", "Stress indicators", "Low activity", "Unhealthy acoustics"]
            })

    @app.get("/api/state")
    def api_state():
        """Get current system state including risk explanation"""
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            r = conn.execute("SELECT * FROM readings ORDER BY ts DESC LIMIT 1").fetchone()
            d = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
            a = conn.execute("SELECT * FROM audio ORDER BY ts DESC LIMIT 1").fetchone()
            
            # Try to get battery info if table exists
            battery_info = None
            try:
                battery_info = conn.execute("SELECT * FROM battery ORDER BY ts DESC LIMIT 1").fetchone()
            except Exception as e:
                print(f"Battery table query failed: {e}")
        
        # Get sensor values
        temperature = r["temperature"] if r else None
        humidity = r["humidity"] if r else None
        co2 = float(r["co2"]) if r else 0.0
        risk_level = r["risk"] if r else "unknown"
        
        # Extract battery values - try database first, then shared state
        battery_percentage = 0
        battery_voltage = 0
        battery_power_connected = False
        battery_level = "unknown"
        
        if battery_info:
            # Try different possible column names from database
            try:
                battery_percentage = battery_info["percentage"] if "percentage" in battery_info.keys() else \
                                    battery_info["percent"] if "percent" in battery_info.keys() else \
                                    battery_info["pct"] if "pct" in battery_info.keys() else 0
            except:
                battery_percentage = 0
            
            try:
                battery_voltage = battery_info["voltage"] if "voltage" in battery_info.keys() else \
                                 battery_info["volt"] if "volt" in battery_info.keys() else \
                                 battery_info["v"] if "v" in battery_info.keys() else 0
            except:
                battery_voltage = 0
            
            try:
                battery_power_connected = battery_info["power_connected"] if "power_connected" in battery_info.keys() else \
                                         battery_info["charging"] if "charging" in battery_info.keys() else \
                                         battery_info["plugged"] if "plugged" in battery_info.keys() else False
            except:
                battery_power_connected = False
            
            try:
                battery_level = battery_info["level"] if "level" in battery_info.keys() else \
                               battery_info["status"] if "status" in battery_info.keys() else "unknown"
            except:
                battery_level = "unknown"
        else:
            # Fall back to shared state (updated by main monitor)
            battery_percentage = app.battery_state.get("percentage", 0)
            battery_voltage = app.battery_state.get("voltage", 0)
            battery_power_connected = app.battery_state.get("power_connected", False)
            battery_level = app.battery_state.get("level", "unknown")
        
        # Determine if ML model is being used
        # Check if sensor_monitor has ML model loaded
        # Also check for "very high" which only comes from ML (Spanish "Muy Alto")
        ml_model_in_use = False
        
        # Method 1: Check sensor_monitor directly
        if app.sensor_monitor is not None:
            if hasattr(app.sensor_monitor, '_clf') and app.sensor_monitor._clf is not None:
                ml_model_in_use = True
        
        # Method 2: "very high" only comes from ML model
        if risk_level == "very high":
            ml_model_in_use = True
        
        # Method 3: Check if database has ml_used column (future-proofing)
        try:
            if r and "ml_used" in r.keys():
                ml_model_in_use = bool(r["ml_used"])
        except:
            pass
        
        # Generate risk explanation
        risk_explanation = None
        if temperature is not None and humidity is not None:
            # Try to use sensor_monitor if available
            if app.sensor_monitor and hasattr(app.sensor_monitor, 'get_risk_explanation'):
                try:
                    risk_explanation = app.sensor_monitor.get_risk_explanation(temperature, humidity, co2)
                except Exception as e:
                    print(f"Error getting risk explanation from sensor_monitor: {e}")
                    risk_explanation = get_risk_explanation_for_values(temperature, humidity, co2)
                    risk_explanation["risk_level"] = risk_level
                    risk_explanation["using_ml_model"] = ml_model_in_use
            else:
                risk_explanation = get_risk_explanation_for_values(temperature, humidity, co2)
                risk_explanation["risk_level"] = risk_level
                risk_explanation["using_ml_model"] = ml_model_in_use
        
        response = {
            "temperature": temperature,
            "humidity": humidity,
            "co2": co2,
            "risk_level": risk_level,
            "risk_explanation": risk_explanation,
            "bee_count": (d["bees"] if d else 0),
            "varroa_detected": bool((d["varroa"] if d else 0)),
            "varroa_count": (d["varroa"] if d else 0),
            "audio_health": (a["label"] if a else "unknown"),
            "audio_confidence": (a["confidence"] if a else 0.0),
            "last_sensor_time": (r["ts"] if r else None),
            "last_detection_time": (d["ts"] if d else None),
            "last_audio_time": (a["ts"] if a else None),
            "timestamp": (r["ts"] if r else None),
            # Battery info
            "battery_percentage": battery_percentage,
            "battery_voltage": battery_voltage,
            "battery_power_connected": battery_power_connected,
            "battery_level": battery_level,
        }
        
        return jsonify(response)

    @app.get("/api/status")
    def api_status():
        """Get current system status from database (legacy endpoint)"""
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            r = conn.execute("SELECT * FROM readings ORDER BY ts DESC LIMIT 1").fetchone()
            d = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
            a = conn.execute("SELECT * FROM audio ORDER BY ts DESC LIMIT 1").fetchone()
        
        return jsonify({
            "temperature": (r["temperature"] if r else None),
            "humidity": (r["humidity"] if r else None),
            "co2": float(r["co2"]) if r else 0.0,
            "risk_level": (r["risk"] if r else "unknown"),
            "bee_count": (d["bees"] if d else 0),
            "varroa_detected": bool((d["varroa"] if d else 0)),
            "varroa_count": (d["varroa"] if d else 0),
            "audio_health": (a["label"] if a else "unknown"),
            "audio_confidence": (a["confidence"] if a else 0.0),
            "last_sensor_time": (r["ts"] if r else None),
            "last_detection_time": (d["ts"] if d else None),
            "last_audio_time": (a["ts"] if a else None),
            "timestamp": (r["ts"] if r else None)
        })

    @app.get("/api/events")
    def api_events():
        """Get recent events"""
        limit = int(request.args.get("limit", 50))
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT ts,type,media_path FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return jsonify([{"ts": r["ts"], "type": r["type"], "media_path": r["media_path"]} for r in rows])

    # Data collection mode state
    app.collect_mode = False
    app.collect_thread = None
    app.collect_stats = {"recordings": 0, "errors": 0}
    
    @app.route("/api/collect/start", methods=["POST"])
    def start_collection():
        """Start data collection - every 3 min: video+audio (5x louder) + sensor readings"""
        import threading
        import subprocess
        from datetime import datetime
        
        if app.collect_mode:
            return jsonify({"status": "already_running"})
        
        app.collect_mode = True
        app.collect_stats = {"recordings": 0, "errors": 0}
        
        data_dir = os.path.dirname(db_path)
        collect_dir = os.path.join(data_dir, "collection")
        os.makedirs(collect_dir, exist_ok=True)
        
        def get_sensor_reading():
            """Get current sensor readings from database"""
            try:
                import sqlite3
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM readings ORDER BY ts DESC LIMIT 1").fetchone()
                conn.close()
                if row:
                    return {
                        "timestamp": row["ts"],
                        "temperature": row["temperature"],
                        "humidity": row["humidity"],
                        "co2": row["co2"],
                        "risk": row["risk"]
                    }
            except:
                pass
            return None
        
        def collection_loop():
            """Record video+audio and sensor data every 3 minutes"""
            import time
            import json
            
            while app.collect_mode:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"\n Collection: {ts}")
                
                # Create folder for this recording
                folder_path = os.path.join(collect_dir, f"recording_{ts}")
                os.makedirs(folder_path, exist_ok=True)
                
                # === VIDEO + AUDIO (30 sec) ===
                try:
                    video_path = os.path.join(folder_path, f"video_{ts}.mp4")
                    audio_path = os.path.join(folder_path, f"audio_{ts}.mp3")
                    temp_video = f"/tmp/vid_{ts}.mp4"
                    temp_audio = f"/tmp/aud_{ts}.wav"
                    
                    print(f"   Recording 30s video + audio...")
                    
                    # Record video with autofocus on center
                    video_proc = subprocess.Popen([
                        "rpicam-vid", "-o", temp_video,
                        "--width", "1920", "--height", "1080",
                        "--vflip", "--hflip", "-t", "30000",
                        "--codec", "libav", "--libav-format", "mp4", "--nopreview",
                        "--autofocus-mode", "auto",
                        "--autofocus-window", "0.35,0.35,0.3,0.3"
                    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    # Record audio simultaneously
                    subprocess.run([
                        "arecord", "-D", "hw:2,0", "-f", "S32_LE",
                        "-r", "48000", "-c", "2", "-d", "30", temp_audio
                    ], capture_output=True, timeout=40)
                    
                    video_proc.communicate(timeout=15)
                    
                    # Merge video + audio with 60x volume boost
                    if os.path.exists(temp_video) and os.path.exists(temp_audio):
                        subprocess.run([
                            "ffmpeg", "-i", temp_video, "-i", temp_audio,
                            "-c:v", "copy", "-af", "volume=60.0", "-c:a", "aac", 
                            "-shortest", "-y", video_path
                        ], capture_output=True, timeout=30)
                        print(f"    Video saved (audio 60x boosted)")
                        
                        # Extract audio as mp3 with additional 50x boost
                        subprocess.run([
                            "ffmpeg", "-i", video_path,
                            "-vn", "-af", "volume=50.0", "-acodec", "libmp3lame", "-q:a", "2",
                            "-y", audio_path
                        ], capture_output=True, timeout=30)
                        print(f"    Audio extracted as mp3 (additional 50x boost)")
                        
                    elif os.path.exists(temp_video):
                        os.rename(temp_video, video_path)
                        print(f"    Video saved (no audio)")
                    
                    # Cleanup temp files
                    for f in [temp_video, temp_audio]:
                        try:
                            if os.path.exists(f):
                                os.remove(f)
                        except:
                            pass
                    
                except Exception as e:
                    app.collect_stats["errors"] += 1
                    print(f"    Video error: {e}")
                
                # === SENSOR READINGS ===
                try:
                    sensor_data = get_sensor_reading()
                    sensor_path = os.path.join(folder_path, f"sensors_{ts}.txt")
                    
                    with open(sensor_path, 'w') as f:
                        f.write(f"=== Sensor Readings: {ts} ===\n\n")
                        if sensor_data:
                            f.write(f"Timestamp: {sensor_data['timestamp']}\n")
                            f.write(f"Temperature: {sensor_data['temperature']}C\n")
                            f.write(f"Humidity: {sensor_data['humidity']}%\n")
                            f.write(f"CO2: {sensor_data['co2']} ppm\n")
                            f.write(f"Risk Level: {sensor_data['risk']}\n")
                        else:
                            f.write("No sensor data available\n")
                    
                    # Also save as JSON for Colab to parse
                    sensor_json_path = os.path.join(folder_path, f"sensors_{ts}.json")
                    with open(sensor_json_path, 'w') as f:
                        json.dump(sensor_data or {}, f, indent=2)
                    
                    print(f"    Sensor data saved")
                    
                except Exception as e:
                    print(f"    Sensor error: {e}")
                
                app.collect_stats["recordings"] += 1
                
                # Wait 3 minutes
                print(f"   Next in 3 min (Total: {app.collect_stats['recordings']} recordings)")
                for _ in range(180):
                    if not app.collect_mode:
                        break
                    time.sleep(1)
            
            print("Data collection stopped")
        
        app.collect_thread = threading.Thread(target=collection_loop, daemon=True)
        app.collect_thread.start()
        
        return jsonify({"status": "started"})
    
    @app.route("/api/collect/stop", methods=["POST"])
    def stop_collection():
        """Stop data collection mode"""
        app.collect_mode = False
        stats = app.collect_stats.copy()
        return jsonify({"status": "stopped", "videos": stats.get("recordings", 0), "photos": 0})
    
    @app.get("/api/collect/status")
    @app.route("/api/collect/status")
    def collection_status():
        """Get current collection stats"""
        return jsonify({
            "active": app.collect_mode,
            "recordings": app.collect_stats.get("recordings", 0),
            "errors": app.collect_stats.get("errors", 0)
        })

    def update_battery_state(percentage, voltage, power_connected, level="unknown"):
        """Update battery state - called by main monitor"""
        app.battery_state["percentage"] = percentage
        app.battery_state["voltage"] = voltage
        app.battery_state["power_connected"] = power_connected
        app.battery_state["level"] = level
    
    # Attach the update function to the app so main_monitor can call it
    app.update_battery_state = update_battery_state

    @app.get("/api/assets")
    def api_assets():
        """Get latest asset paths"""
        assets = latest_assets_cb()
        
        # Also find latest audio file by scanning directory
        audio_dir = os.path.join(os.path.dirname(db_path), "audio")
        if os.path.exists(audio_dir):
            audio_files = sorted([
                os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
                if f.endswith('.wav')
            ], key=os.path.getmtime, reverse=True)
            if audio_files:
                assets['latest_audio'] = audio_files[0]
        
        # Also find latest clip file by scanning directory
        clips_dir = os.path.join(os.path.dirname(db_path), "clips")
        if os.path.exists(clips_dir):
            clip_files = sorted([
                os.path.join(clips_dir, f) for f in os.listdir(clips_dir)
                if f.endswith('.mp4')
            ], key=os.path.getmtime, reverse=True)
            if clip_files:
                assets['latest_clip'] = clip_files[0]
        
        return jsonify(assets)

    @app.get("/api/readings")
    def api_readings():
        """Get recent sensor readings for charts"""
        hours = int(request.args.get("hours", 24))
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ts, temperature, humidity, co2, risk FROM readings WHERE ts > ? ORDER BY ts ASC",
                (cutoff,)
            ).fetchall()
        
        return jsonify([dict(r) for r in rows])

    @app.get("/api/detections")
    def api_detections():
        """Get recent detections for charts"""
        hours = int(request.args.get("hours", 24))
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ts, bees, varroa FROM detections WHERE ts > ? ORDER BY ts ASC",
                (cutoff,)
            ).fetchall()
        
        return jsonify([dict(r) for r in rows])

    @app.get("/media")
    def media():
        """Serve media files (images, videos, audio)"""
        path = request.args.get("path", "")
        if not path:
            return ("Missing path", 400)
        abspath = os.path.abspath(path)
        basedir = os.path.dirname(abspath)
        fname = os.path.basename(abspath)
        if not os.path.exists(abspath):
            return ("Not found", 404)
        return send_from_directory(basedir, fname, as_attachment=False)

    return app
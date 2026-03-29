"""
Battery Monitor for X1202 UPS Board
Uses same method as official x120x/bat.py
"""

import struct
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BatteryMonitor:
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize battery monitor. Config parameter accepted for compatibility."""
        self._initialized = False
        self.available = False
        self.bus = None
        self.address = 0x36
        
        try:
            import smbus2
            self.bus = smbus2.SMBus(1)
            # Test read to verify chip responds
            self.bus.read_word_data(self.address, 2)
            self.available = True
            self._initialized = True
            logger.info("Battery monitor initialized (MAX17048 at 0x36)")
        except Exception as e:
            logger.warning(f"Battery monitor not available: {e}")
            self.available = False
            self._initialized = False
    
    def _read_voltage(self) -> float:
        """Read voltage using official method"""
        read = self.bus.read_word_data(self.address, 2)
        swapped = struct.unpack("<H", struct.pack(">H", read))[0]
        voltage = swapped * 1.25 / 1000 / 16
        return voltage
    
    def _read_capacity(self) -> float:
        """Read capacity using official method"""
        read = self.bus.read_word_data(self.address, 4)
        swapped = struct.unpack("<H", struct.pack(">H", read))[0]
        capacity = swapped / 256
        return capacity
    
    def should_shutdown(self, threshold_voltage: float = 3.2) -> bool:
        """Check if battery is critically low"""
        if not self.available:
            return False
        try:
            voltage = self._read_voltage()
            return voltage < threshold_voltage
        except:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get battery status"""
        if not self.available:
            return {
                "available": False,
                "voltage": 0.0,
                "percentage": 0,
                "power_connected": True,
                "level": "unknown",
                "indicator": "❓"
            }
        
        try:
            voltage = self._read_voltage()
            percentage = min(100, max(0, self._read_capacity()))
            
            # Determine if charging (voltage > 4.0V typically means charger connected)
            power_connected = voltage > 4.0
            
            # Determine level
            if percentage >= 75:
                level = "high"
                indicator = "🔋"
            elif percentage >= 40:
                level = "medium" 
                indicator = "🔋"
            elif percentage >= 20:
                level = "low"
                indicator = "🪫"
            else:
                level = "critical"
                indicator = "⚠️"
            
            # Add charging indicator
            if power_connected:
                indicator = "🔌" + indicator
            
            return {
                "available": True,
                "voltage": round(voltage, 2),
                "percentage": int(percentage),
                "power_connected": power_connected,
                "level": level,
                "indicator": indicator
            }
            
        except Exception as e:
            logger.error(f"Battery read error: {e}")
            return {
                "available": False,
                "voltage": 0.0,
                "percentage": 0,
                "power_connected": True,
                "level": "error",
                "indicator": "❌"
            }


if __name__ == "__main__":
    import time
    
    print("🔋 Battery Monitor Test")
    print("=" * 40)
    
    monitor = BatteryMonitor()
    
    if not monitor.available:
        print("❌ Battery chip not detected!")
        print("   Check I2C connection and run: sudo i2cdetect -y 1")
        exit(1)
    
    print("✅ Battery monitor initialized\n")
    
    for i in range(5):
        status = monitor.get_status()
        print(f"Reading {i+1}:")
        print(f"   Voltage: {status['voltage']}V")
        print(f"   Battery: {status['percentage']}%")
        print(f"   Level: {status['level']}")
        print(f"   Power: {'Connected' if status['power_connected'] else 'Battery'}")
        print(f"   {status['indicator']}")
        print()
        time.sleep(2)
    
    print("✅ Test complete!")
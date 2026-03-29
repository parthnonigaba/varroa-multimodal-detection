"""
Sensor Monitor - SCD41 CO2, Temperature, and Humidity Sensor
Updated to use Teyleten Robot SCD41 NDIR sensor via I2C

Hardware: SCD41 on I2C Bus 1, Address 0x62
Features: CO2 (400-5000ppm), Temperature, Humidity
"""

import time
import logging
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
import pickle
import numpy as np

try:
    import board
    import adafruit_scd4x
except ImportError as e:
    logging.error(f"Failed to import sensor libraries: {e}")
    board = None
    adafruit_scd4x = None


class SCD41Monitor:
    """
    Monitor for SCD41 environmental sensor
    Provides CO2, temperature, and humidity readings
    Includes ML model for varroa risk prediction
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize SCD41 sensor and load ML models
        
        Args:
            config: Configuration dictionary from config.json
        """
        self.logger = logging.getLogger(__name__)
        
        # Get sensor configuration
        s = config.get("sensors", {})
        self.interval = int(s.get("reading_interval", 60))
        
        # Load ML model paths
        self._sensor_model_path = s.get("sensor_model_path")
        self._sensor_label_enc = s.get("label_encoder")
        
        # ML models
        self._clf = None  # Random Forest classifier
        self._le = None   # Label encoder
        
        # Sensor hardware
        self.sensor = None
        self._initialized = False
        
        # Initialize sensor and models
        self._init_sensor()
        self._load_models()
        
        self.logger.info("SCD41Monitor initialized")
    
    def _init_sensor(self) -> None:
        """Initialize SCD41 sensor - simple approach that works"""
        if board is None or adafruit_scd4x is None:
            self.logger.error("Sensor libraries not available")
            return
        
        try:
            # Create I2C and sensor object (just like test_minimal.py)
            i2c = board.I2C()
            self.sensor = adafruit_scd4x.SCD4X(i2c)
            
            self.logger.info(
                f"SCD41 connected. Serial: {[hex(i) for i in self.sensor.serial_number]}"
            )
            
            # Start measurements (sensor handles the rest)
            self.sensor.start_periodic_measurement()
            
            self.logger.info("SCD41 measurement started")
            
            # Wait for first measurement
            time.sleep(5)
            self._initialized = True
            
        except Exception as e:
            self.logger.error(f"Sensor initialization failed: {e}")
            self._initialized = False
    
    def _load_models(self) -> None:
        """Load ML models for varroa risk prediction"""
        import os
        
        # Log what paths we're trying to load
        self.logger.info(f"Attempting to load sensor ML models...")
        self.logger.info(f"  Model path from config: {self._sensor_model_path}")
        self.logger.info(f"  Label encoder path from config: {self._sensor_label_enc}")
        self.logger.info(f"  Current working directory: {os.getcwd()}")
        
        # Load classifier model
        if self._sensor_model_path:
            model_path = self._sensor_model_path
            
            # Check if file exists
            if not os.path.exists(model_path):
                self.logger.warning(f"Sensor model file not found at: {os.path.abspath(model_path)}")
            else:
                try:
                    with open(model_path, "rb") as f:
                        self._clf = pickle.load(f)
                    self.logger.info(f"✅ Loaded sensor ML model from {os.path.abspath(model_path)}")
                except Exception as e:
                    self.logger.error(f"Failed to load sensor model: {e}")
                    self._clf = None
        else:
            self.logger.warning("No sensor_model_path configured - will use rule-based assessment")
        
        # Load label encoder
        if self._sensor_label_enc:
            enc_path = self._sensor_label_enc
            
            # Check if file exists
            if not os.path.exists(enc_path):
                self.logger.warning(f"Label encoder file not found at: {os.path.abspath(enc_path)}")
            else:
                try:
                    with open(enc_path, "rb") as f:
                        self._le = pickle.load(f)
                    self.logger.info(f"✅ Loaded label encoder from {os.path.abspath(enc_path)}")
                    # Log the classes
                    if hasattr(self._le, 'classes_'):
                        self.logger.info(f"   Label classes: {list(self._le.classes_)}")
                except Exception as e:
                    self.logger.error(f"Failed to load label encoder: {e}")
                    self._le = None
        else:
            self.logger.warning("No label_encoder path configured - will use rule-based assessment")
        
        # Final status
        if self._clf is not None and self._le is not None:
            self.logger.info("✅ Sensor ML model ready - will use ML-based risk prediction")
        else:
            self.logger.warning("⚠️ Sensor ML model not loaded - will use RULE-BASED risk prediction")
    
    def single_read(self) -> Tuple[float, float, float]:
        """
        Perform a single sensor reading
        
        Returns:
            Tuple of (temperature_C, humidity_percent, co2_ppm)
        
        Raises:
            RuntimeError: If sensor not initialized or reading fails
        """
        if not self._initialized or self.sensor is None:
            raise RuntimeError("Sensor not initialized")
        
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                if self.sensor.data_ready:
                    temperature = self.sensor.temperature
                    humidity = self.sensor.relative_humidity
                    co2 = self.sensor.CO2
                    
                    # Validate readings
                    if self._validate_reading(temperature, humidity, co2):
                        return temperature, humidity, float(co2)
                    else:
                        self.logger.warning(
                            f"Invalid reading: T={temperature}, H={humidity}, CO2={co2}"
                        )
                
            except Exception as e:
                self.logger.error(f"Reading error on attempt {attempt + 1}: {e}")
            
            time.sleep(1)  # Wait for next reading
        
        raise RuntimeError("Failed to get valid sensor reading after all attempts")
    
    def _validate_reading(self, temp: float, humidity: float, co2: float) -> bool:
        """
        Validate sensor readings are within reasonable ranges
        
        Args:
            temp: Temperature in Celsius
            humidity: Relative humidity in percent
            co2: CO2 concentration in ppm
        
        Returns:
            True if readings are valid
        """
        # SCD41 specifications
        if co2 < 400 or co2 > 5000:
            return False
        if temp < -10 or temp > 60:
            return False
        if humidity < 0 or humidity > 100:
            return False
        return True
    
    def predict_risk(
        self, 
        temperature: float, 
        humidity: float, 
        co2: float
    ) -> Optional[str]:
        """
        Predict varroa mite risk using ML model
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity in percent
            co2: CO2 concentration in ppm
        
        Returns:
            Risk level string ('low', 'medium', 'high', 'very high') or None if model unavailable
        """
        if self._clf is None or self._le is None:
            self.logger.info("Using SIMPLE RULES for risk assessment (ML model not loaded)")
            return self._simple_risk_assessment(temperature, humidity, co2)
        
        try:
            # Feature engineering (matches training data)
            features = self._extract_features(temperature, humidity, co2)
            
            # Predict using Random Forest model
            prediction = self._clf.predict([features])[0]
            
            # Decode label
            risk = self._le.inverse_transform([prediction])[0]
            
            # Translate Spanish labels to English
            translation = {
                'Alto': 'high',
                'Bajo': 'low',
                'Medio': 'medium',
                'Muy Alto': 'very high'
            }
            risk = translation.get(risk, risk.lower())
            
            self.logger.info(f"Using ML MODEL for risk assessment: {risk} (T={temperature:.1f}°C, H={humidity:.1f}%, CO2={co2:.0f}ppm)")
            
            return risk
            
        except Exception as e:
            self.logger.error(f"ML prediction failed: {e}")
            self.logger.info("Falling back to SIMPLE RULES for risk assessment")
            return self._simple_risk_assessment(temperature, humidity, co2)
    
    def get_risk_explanation(
        self,
        temperature: float,
        humidity: float,
        co2: float
    ) -> Dict[str, Any]:
        """
        Get detailed risk assessment with explanations for each factor
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity in percent
            co2: CO2 concentration in ppm
        
        Returns:
            Dictionary with risk level and explanations for each factor
        """
        factors = []
        
        # Temperature analysis
        # Varroa mites thrive at 33-36°C
        if 33 <= temperature <= 36:
            factors.append({
                "factor": "temperature",
                "status": "warning",
                "message": f"Temperature ({temperature:.1f}°C) is in optimal range for varroa mites (33-36°C)"
            })
        elif 30 <= temperature <= 37:
            factors.append({
                "factor": "temperature",
                "status": "caution",
                "message": f"Temperature ({temperature:.1f}°C) is near optimal range for varroa mites"
            })
        else:
            factors.append({
                "factor": "temperature",
                "status": "ok",
                "message": f"Temperature ({temperature:.1f}°C) is outside varroa optimal range"
            })
        
        # Humidity analysis
        # Varroa prefer low humidity (<60% RH)
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
        # Elevated CO2 (800-1500 ppm) indicates colony stress
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
        
        # Get overall risk
        risk = self.predict_risk(temperature, humidity, co2)
        
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
            "risk_level": risk,
            "summary": summary,
            "factors": factors,
            "using_ml_model": self._clf is not None
        }
    
    def _extract_features(
        self, 
        temperature: float, 
        humidity: float, 
        co2: float
    ) -> list:
        """
        Extract features for ML model - matches training data features
        
        The model expects exactly 14 features in this order:
        ['temperature', 'humidity', 'co2', 'tvoc', 'total', 
         'temp_stress', 'temp_optimal_deviation', 'humidity_stress', 'humidity_optimal_deviation',
         'co2_stress', 'co2_excess', 'heat_index', 'temp_humidity_ratio', 'total_stress']
        
        We only have 3 sensor readings (temp, humidity, co2), so we:
        - Set tvoc and total to neutral/average values (we don't have these sensors)
        - Compute all derived features from the 3 values we have
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity in percent
            co2: CO2 concentration in ppm
        
        Returns:
            Feature vector with 14 features for model
        """
        # Optimal ranges for bees (from research)
        OPTIMAL_TEMP_MIN, OPTIMAL_TEMP_MAX = 32.0, 36.0  # Brood nest temperature
        OPTIMAL_HUMIDITY_MIN, OPTIMAL_HUMIDITY_MAX = 50.0, 70.0
        OPTIMAL_CO2_MAX = 800.0  # Normal indoor baseline
        
        # 1. temperature - raw value
        # 2. humidity - raw value  
        # 3. co2 - raw value
        
        # 4. tvoc - Total Volatile Organic Compounds (we don't have this sensor)
        #    Set to a neutral/average value (typical indoor: 100-300 ppb)
        tvoc = 200.0  # neutral default
        
        # 5. total - unclear what this represents in original dataset
        #    Could be a combined air quality index, set to neutral
        total = 500.0  # neutral default
        
        # 6. temp_stress - how far from optimal temperature range
        if temperature < OPTIMAL_TEMP_MIN:
            temp_stress = OPTIMAL_TEMP_MIN - temperature
        elif temperature > OPTIMAL_TEMP_MAX:
            temp_stress = temperature - OPTIMAL_TEMP_MAX
        else:
            temp_stress = 0.0
        
        # 7. temp_optimal_deviation - deviation from center of optimal range
        optimal_temp_center = (OPTIMAL_TEMP_MIN + OPTIMAL_TEMP_MAX) / 2  # 34°C
        temp_optimal_deviation = abs(temperature - optimal_temp_center)
        
        # 8. humidity_stress - how far from optimal humidity range
        if humidity < OPTIMAL_HUMIDITY_MIN:
            humidity_stress = OPTIMAL_HUMIDITY_MIN - humidity
        elif humidity > OPTIMAL_HUMIDITY_MAX:
            humidity_stress = humidity - OPTIMAL_HUMIDITY_MAX
        else:
            humidity_stress = 0.0
        
        # 9. humidity_optimal_deviation - deviation from center of optimal range
        optimal_humidity_center = (OPTIMAL_HUMIDITY_MIN + OPTIMAL_HUMIDITY_MAX) / 2  # 60%
        humidity_optimal_deviation = abs(humidity - optimal_humidity_center)
        
        # 10. co2_stress - elevated CO2 indicates poor ventilation/stress
        if co2 > OPTIMAL_CO2_MAX:
            co2_stress = (co2 - OPTIMAL_CO2_MAX) / 100.0  # Scaled
        else:
            co2_stress = 0.0
        
        # 11. co2_excess - binary or scaled indicator of high CO2
        co2_excess = max(0.0, co2 - OPTIMAL_CO2_MAX)
        
        # 12. heat_index - combines temperature and humidity
        #     Using simplified heat index formula
        heat_index = temperature + (0.05 * humidity)
        
        # 13. temp_humidity_ratio
        if humidity > 0:
            temp_humidity_ratio = temperature / humidity
        else:
            temp_humidity_ratio = temperature  # Avoid division by zero
        
        # 14. total_stress - combined stress indicator
        total_stress = temp_stress + humidity_stress + co2_stress
        
        features = [
            temperature,                # 1
            humidity,                   # 2
            co2,                        # 3
            tvoc,                       # 4 (default - no sensor)
            total,                      # 5 (default - no sensor)
            temp_stress,                # 6
            temp_optimal_deviation,     # 7
            humidity_stress,            # 8
            humidity_optimal_deviation, # 9
            co2_stress,                 # 10
            co2_excess,                 # 11
            heat_index,                 # 12
            temp_humidity_ratio,        # 13
            total_stress                # 14
        ]
        
        self.logger.debug(f"Extracted 14 features: {features}")
        return features
    
    def _simple_risk_assessment(
        self, 
        temperature: float, 
        humidity: float, 
        co2: float
    ) -> str:
        """
        Simple rule-based risk assessment when ML model unavailable
        
        Based on varroa mite research:
        - Varroa thrive at 33-36°C, <60% RH, elevated CO2
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity in percent
            co2: CO2 concentration in ppm
        
        Returns:
            Risk level: 'low', 'medium', or 'high'
        """
        risk_factors = 0
        
        # Temperature: Varroa prefer warm (33-36°C)
        if 33 <= temperature <= 36:
            risk_factors += 2
        elif 30 <= temperature <= 37:
            risk_factors += 1
        
        # Humidity: Varroa prefer dry (<60% RH)
        if humidity < 60:
            risk_factors += 2
        elif humidity < 70:
            risk_factors += 1
        
        # CO2: Elevated indicates stress
        if 800 <= co2 <= 1500:
            risk_factors += 1
        elif co2 > 1500:
            risk_factors += 2
        
        # Classify risk
        if risk_factors >= 4:
            return "high"
        elif risk_factors >= 2:
            return "medium"
        else:
            return "low"
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get sensor health status
        
        Returns:
            Dictionary with sensor status information
        """
        return {
            "initialized": self._initialized,
            "sensor_type": "SCD41",
            "has_ml_model": self._clf is not None,
            "serial_number": (
                [hex(i) for i in self.sensor.serial_number] 
                if self.sensor else None
            ),
        }
    
    def close(self) -> None:
        """Clean shutdown of sensor"""
        if self.sensor:
            try:
                self.sensor.stop_periodic_measurement()
                self.logger.info("SCD41 sensor stopped")
            except Exception as e:
                self.logger.error(f"Error stopping sensor: {e}")


# Maintain backward compatibility with old class name
ENS160AHT21Monitor = SCD41Monitor


# Test script
if __name__ == "__main__":
    import json
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # Initialize sensor
    monitor = SCD41Monitor(config)
    
    if not monitor._initialized:
        print("\n❌ Sensor failed to initialize!")
        exit(1)
    
    # Test readings
    print("\n✅ SCD41 sensor initialized successfully!")
    print("=" * 50)
    
    for i in range(5):
        try:
            temp, humidity, co2 = monitor.single_read()
            risk = monitor.predict_risk(temp, humidity, co2)
            
            print(f"\nReading {i+1}:")
            print(f"  Temperature: {temp:.1f}°C")
            print(f"  Humidity:    {humidity:.1f}%")
            print(f"  CO2:         {co2:.0f} ppm")
            print(f"  Risk Level:  {risk}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        if i < 4:
            time.sleep(5)
    
    # Clean up
    monitor.close()
    print("\n✅ Test complete!")
    # Maintain backward compatibility
ENS160AHT21Monitor = SCD41Monitor
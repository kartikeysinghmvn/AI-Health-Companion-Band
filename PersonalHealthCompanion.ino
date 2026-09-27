/*
=============================================================================
 SIH26181 — AI-Powered Personal Health & Environmental Disaster Companion
 Target: ESP32-S3 (Dual-Core, Wi-Fi, native ADC1 on GPIO1-10)

 *** MODIFIED VERSION ***
 TensorFlow Lite Micro (neural network) REMOVED and replaced with 3
 classical Random Forest models (HR, RESP, SpO2), exported via `emlearn`
 from Python/scikit-learn, running as plain C code (no ML runtime, no
 tensor arena). A 4th output, Stress, is computed directly from ECG-derived
 HRV (RMSSD) using a fixed formula — not machine-learned.

 ARCHITECTURE
 ------------
 - Core 0 : "SensorTask" -> polls all sensors, buffers 8s of raw PPG+ECG,
                             runs the 3 RF models + stress formula once per
                             window, evaluates alert rules, drives
                             buzzer/vibrator.
 - Core 1 : Arduino loop() -> services the Wi-Fi web server and the
                               push-button-driven multi-page SSD1306 OLED UI.
 - A FreeRTOS mutex (dataMutex) guards the single shared `Telemetry` struct.

 REQUIRED LIBRARIES (install via Arduino Library Manager / PlatformIO)
 -----------------------------------------------------------------------
 1.  WiFi.h                          - bundled with ESP32 Arduino core
 2.  WebServer.h                     - bundled with ESP32 Arduino core
 3.  ArduinoJson                     - by Benoit Blanchon
 4.  Adafruit_GFX                    - by Adafruit
 5.  Adafruit_SSD1306                - by Adafruit (128x64 I2C OLED)
 6.  DHT sensor library              - by Adafruit (DHT22)
 7.  Adafruit_Unified_Sensor         - by Adafruit (DHT dependency)
 8.  OneWire                         - by Jim Studt / PJRC
 9.  DallasTemperature              - by Miles Burton (DS18B20)
 10. Adafruit_BMP280_Library         - by Adafruit
 11. SparkFun MAX3010x Pulse and Proximity Sensor Library
                                     - by SparkFun (MAX30105, provides
                                       particleSensor.getIR())
 12. arduinoFFT                     - by Enrique Condes (used ONLY to
                                       compute PPG dominant frequency / PSD
                                       features for the RF models — this
                                       replaces TensorFlowLite_ESP32, which
                                       has been REMOVED)

 NOTE: TensorFlowLite_ESP32 and health_risk_model.h are NO LONGER NEEDED
 and have been removed from this sketch.

 PIN MAP (unchanged from original)
 -----------------------------------------------------------------------
 I2C (OLED, MAX30102, MPU6050, BMP280) : SDA = GPIO8 , SCL = GPIO9
 DS18B20 (Body Temp, OneWire)          : GPIO4   (+4.7k pull-up to 3V3)
 DHT22 (Ambient Temp/Humidity)         : GPIO5
 AD8232 ECG  OUTPUT                    : GPIO1 (ADC1_CH0)
 AD8232 ECG  LO+                       : GPIO2
 AD8232 ECG  LO-                       : GPIO3
 MQ135 (Air Quality)                   : GPIO6 (ADC1_CH5)
 UV Sensor                             : GPIO7 (ADC1_CH6)
 GSR Sensor                            : GPIO10 (ADC1_CH9)
 Push Button                           : GPIO11 (INPUT_PULLUP, active LOW)
 Buzzer                                : GPIO12 (digital/PWM)
 Vibrator Motor (via N-MOSFET)         : GPIO13 (digital gate drive)
=============================================================================
*/
#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <DHT.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Adafruit_BMP280.h>
#include <MAX30105.h>          // SparkFun MAX3010x library (class name: MAX30105)
#include <arduinoFFT.h>        // used for PPG dominant-frequency / PSD features

// ---- Random Forest models (emlearn-exported C headers) --------------------
// These 3 files must sit in the SAME folder as this .ino
#include "model_hr_final.h"
#include "model_resp_final.h"
#include "model_spo2_final.h"

// =============================================================================
// PIN DEFINITIONS
// =============================================================================
#define PIN_SDA            8
#define PIN_SCL            9
#define PIN_DS18B20         4
#define PIN_DHT22           5
#define PIN_ECG_OUTPUT      1
#define PIN_ECG_LOPLUS      2
#define PIN_ECG_LOMINUS     3
#define PIN_MQ135           6
#define PIN_UV              7
#define PIN_GSR            10
#define PIN_BUTTON         11
#define PIN_BUZZER         12
#define PIN_VIBRATOR       13

// =============================================================================
// CONSTANTS / CONFIG
// =============================================================================
#define OLED_WIDTH        128
#define OLED_HEIGHT        64
#define OLED_ADDR        0x3C
#define MPU6050_ADDR     0x68
#define BMP280_ADDR      0x76
#define DHTTYPE          DHT22

const char* AP_SSID     = "HealthCompanion_AI";
const char* AP_PASSWORD = "health1234";

// Debounce / UI timing
const unsigned long DEBOUNCE_MS      = 50;
const unsigned long OLED_REFRESH_MS  = 500;

// Sensor cadences (all non-blocking, millis()-based)
const unsigned long T_DHT_MS       = 2500;
const unsigned long T_BMP_MS       = 2000;
const unsigned long T_DS18B20_MS   = 5000;
const unsigned long T_ANALOG_MS    = 500;    // MQ135 / UV / GSR
const unsigned long T_ACCEL_MS     = 200;    // MPU6050

// Physiological / environmental safety thresholds (tune to real-world calibration)
const float HR_MIN_SAFE       = 40.0f;
const float HR_MAX_SAFE       = 140.0f;
const float SPO2_MIN_SAFE     = 90.0f;
const float BODY_TEMP_MAX     = 39.5f;
const float BODY_TEMP_MIN     = 35.0f;
const float AMBIENT_TEMP_HEAT = 45.0f;
const float HUMIDITY_HEAT     = 60.0f;
const int   MQ135_RAW_DANGER  = 3200;
const int   GSR_STRESS_RAW    = 3000;
const float UV_INDEX_HIGH     = 7.0f;

// ---- Model / windowing config (MUST match the Python training pipeline) ----
#define SAMPLE_RATE_HZ     125
#define WINDOW_SEC           8
#define WINDOW_LEN         (SAMPLE_RATE_HZ * WINDOW_SEC)   // 1000 samples
#define FFT_SIZE          1024   // next power of 2 >= WINDOW_LEN, zero-padded
#define NUM_FEATURES        16

// Feature order + integer scale factors — EXACT match to the Python
// `scale_factors` dict used when training the exported RF models.
// [0]mean [1]std [2]min [3]max [4]ptp [5]skew [6]kurtosis [7]rms [8]zcr
// [9]dom_freq [10]psd_mean [11]psd_std [12]ppg_hr_est [13]ibi_std
// [14]sdnn [15]rmssd
const float FEATURE_SCALE[NUM_FEATURES] = {
  1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f,      // mean,std,min,max,ptp
  1000.0f, 1000.0f, 1000.0f,                        // skew,kurtosis,rms
  1.0f,                                             // zcr
  1000.0f,                                          // dom_freq
  1000000.0f,                                       // psd_mean
  100000.0f,                                        // psd_std
  10.0f,                                            // ppg_hr_est
  1000.0f,                                          // ibi_std
  10.0f,                                            // sdnn
  10.0f                                             // rmssd
};

// =============================================================================
// GLOBAL OBJECTS
// =============================================================================
Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire, -1);
DHT dht(PIN_DHT22, DHTTYPE);
OneWire oneWire(PIN_DS18B20);
DallasTemperature ds18b20(&oneWire);
Adafruit_BMP280 bmp;
MAX30105 particleSensor;
WebServer server(80);
SemaphoreHandle_t dataMutex;
TaskHandle_t sensorTaskHandle;

// ---- Raw signal buffers for our own windowed feature extraction -----------
static float ppgBuffer[WINDOW_LEN];
static float ecgBuffer[WINDOW_LEN];
static uint16_t bufIdx = 0;
static volatile bool bufferFull = false;

// ---- FFT working arrays (reused every window, kept off the stack) ---------
static double fftReal[FFT_SIZE];
static double fftImag[FFT_SIZE];
ArduinoFFT<double> FFT = ArduinoFFT<double>(fftReal, fftImag, FFT_SIZE, (double)SAMPLE_RATE_HZ);

// =============================================================================
// SHARED TELEMETRY STRUCT  (protected by dataMutex)
// =============================================================================
enum RiskLevel : uint8_t { RISK_NORMAL = 0, RISK_CAUTION = 1, RISK_CRITICAL = 2 };

struct Telemetry {
  // Vitals (heartRateBPM / spo2Pct are now RF-model outputs, not SparkFun's algorithm)
  float heartRateBPM   = 0;
  float spo2Pct        = 0;
  float respRate       = 0;     // NEW: RF-model output
  float stressIndex    = 0;     // NEW: RMSSD-formula output (0-100)
  bool  hrValid        = false; // true once at least one window has been scored
  float bodyTempC      = 0;
  bool  ecgLeadsOff    = true;
  int   ecgRaw         = 0;

  // Environment
  float ambientTempC   = 0;
  float humidityPct    = 0;
  float pressureHPa    = 0;
  int   mq135Raw       = 0;
  int   uvRaw          = 0;
  float uvIndexApprox  = 0;

  // Stress / motion
  int   gsrRaw         = 0;
  float accelMagG      = 1.0f;
  const char* activityState = "Stationary";

  // AI / system
  RiskLevel aiRisk      = RISK_NORMAL;
  bool  ruleAnomaly      = false;
  bool  alertActive      = false;
  char  suggestion[80]  = "Initializing sensors...";
  unsigned long lastUpdateMs = 0;
};
Telemetry telemetry;

// =============================================================================
// SMALL HELPERS
// =============================================================================
static inline float clamp01(float v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }

static inline int16_t clampInt16(float v) {
  if (v > 32767.0f) return 32767;
  if (v < -32768.0f) return -32768;
  return (int16_t)lroundf(v);
}

Telemetry getTelemetrySnapshot() {
  Telemetry copy;
  xSemaphoreTake(dataMutex, portMAX_DELAY);
  copy = telemetry;
  xSemaphoreGive(dataMutex);
  return copy;
}

// =============================================================================
// MPU6050 — MINIMAL RAW I2C DRIVER (unchanged from original)
// =============================================================================
bool mpu6050WriteReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}
bool mpu6050ReadRegs(uint8_t reg, uint8_t* buf, uint8_t len) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  Wire.requestFrom((int)MPU6050_ADDR, (int)len);
  for (uint8_t i = 0; i < len && Wire.available(); i++) buf[i] = Wire.read();
  return true;
}
bool mpu6050Init() {
  mpu6050WriteReg(0x6B, 0x00);
  delay(50);
  mpu6050WriteReg(0x1C, 0x00);
  delay(10);
  uint8_t whoAmI = 0;
  mpu6050ReadRegs(0x75, &whoAmI, 1);
  return (whoAmI == 0x68);
}
bool mpu6050ReadAccel(float &gx, float &gy, float &gz) {
  uint8_t raw[6];
  if (!mpu6050ReadRegs(0x3B, raw, 6)) return false;
  int16_t x = (int16_t)(raw[0] << 8 | raw[1]);
  int16_t y = (int16_t)(raw[2] << 8 | raw[3]);
  int16_t z = (int16_t)(raw[4] << 8 | raw[5]);
  const float sensitivity = 16384.0f;
  gx = x / sensitivity;
  gy = y / sensitivity;
  gz = z / sensitivity;
  return true;
}

// =============================================================================
// SIGNAL PROCESSING — feature extraction, mirrors the Python pipeline exactly
// =============================================================================

// Per-window min-max normalization to 0..1 — REQUIRED because the training
// data (BIDMC PLETH channel) was pre-scaled to roughly 0..1. Raw MAX30102 IR
// counts are in the tens of thousands, so feeding them in unnormalized would
// give the model completely out-of-distribution feature magnitudes. ECG does
// NOT need this: only its R-peak *timing* is used (sdnn/rmssd/ibi), which is
// scale-invariant.
void normalizeWindow(float* buf, int n) {
  float mn = buf[0], mx = buf[0];
  for (int i = 1; i < n; i++) {
    if (buf[i] < mn) mn = buf[i];
    if (buf[i] > mx) mx = buf[i];
  }
  float range = mx - mn;
  if (range < 1e-6f) range = 1e-6f;
  for (int i = 0; i < n; i++) {
    buf[i] = (buf[i] - mn) / range;
  }
}

// Basic time-domain statistics (mean, std, min, max, ptp, skew, kurtosis, rms, zcr)
void computeBasicStats(const float* buf, int n,
                        float &mean, float &stdv, float &minV, float &maxV,
                        float &ptp, float &skewV, float &kurtV, float &rms, float &zcr) {
  mean = 0; minV = buf[0]; maxV = buf[0];
  for (int i = 0; i < n; i++) {
    mean += buf[i];
    if (buf[i] < minV) minV = buf[i];
    if (buf[i] > maxV) maxV = buf[i];
  }
  mean /= n;
  ptp = maxV - minV;

  float sumSq = 0, sumCube = 0, sumQuad = 0, sumSqRaw = 0;
  for (int i = 0; i < n; i++) {
    float d = buf[i] - mean;
    sumSq   += d * d;
    sumCube += d * d * d;
    sumQuad += d * d * d * d;
    sumSqRaw += buf[i] * buf[i];
  }
  float variance = sumSq / n;
  stdv = sqrtf(variance);
  rms  = sqrtf(sumSqRaw / n);

  if (stdv > 1e-9f) {
    skewV = (sumCube / n) / (stdv * stdv * stdv);
    kurtV = (sumQuad / n) / (variance * variance) - 3.0f; // excess kurtosis (matches scipy default)
  } else {
    skewV = 0; kurtV = 0;
  }

  // zero-crossing rate around the mean
  int crossings = 0;
  for (int i = 1; i < n; i++) {
    bool prevPos = (buf[i - 1] - mean) >= 0;
    bool curPos  = (buf[i] - mean) >= 0;
    if (prevPos != curPos) crossings++;
  }
  zcr = (float)crossings;
}

// Simple peak finder: local maxima above `heightThresh`, at least
// `minDistance` samples apart — mirrors scipy.signal.find_peaks(distance=, height=)
int findPeaks(const float* buf, int n, float heightThresh, int minDistance, int* peakIdxOut, int maxPeaks) {
  int count = 0;
  int lastPeak = -minDistance;
  for (int i = 1; i < n - 1; i++) {
    if (buf[i] > buf[i - 1] && buf[i] >= buf[i + 1] && buf[i] > heightThresh) {
      if (i - lastPeak >= minDistance) {
        if (count < maxPeaks) {
          peakIdxOut[count++] = i;
          lastPeak = i;
        }
      }
    }
  }
  return count;
}

// PPG-only features: dominant frequency + PSD stats via FFT, plus
// peak-interval based HR estimate and its variability (ibi_std)
void computePPGFeatures(const float* normPpg, int n,
                         float &domFreq, float &psdMean, float &psdStd,
                         float &ppgHrEst, float &ibiStd) {
  // ---- FFT (zero-padded to FFT_SIZE) ----
  for (int i = 0; i < FFT_SIZE; i++) {
    fftReal[i] = (i < n) ? (double)normPpg[i] : 0.0;
    fftImag[i] = 0.0;
  }
  FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);
  FFT.compute(FFTDirection::Forward);
  FFT.complexToMagnitude();

  // Restrict search to a physiological band (~0.5-4 Hz) to avoid DC/noise bins
  int binLo = (int)(0.5 * FFT_SIZE / SAMPLE_RATE_HZ);
  int binHi = (int)(4.0 * FFT_SIZE / SAMPLE_RATE_HZ);
  if (binLo < 1) binLo = 1;
  if (binHi >= FFT_SIZE / 2) binHi = FFT_SIZE / 2 - 1;

  int bestBin = binLo;
  double bestMag = fftReal[binLo];
  double sumMag = 0, sumMagSq = 0;
  int cnt = 0;
  for (int b = binLo; b <= binHi; b++) {
    double mag = fftReal[b];
    if (mag > bestMag) { bestMag = mag; bestBin = b; }
    sumMag += mag;
    sumMagSq += mag * mag;
    cnt++;
  }
  domFreq = (float)bestBin * (float)SAMPLE_RATE_HZ / (float)FFT_SIZE;
  psdMean = (cnt > 0) ? (float)(sumMag / cnt) : 0.0f;
  psdStd  = (cnt > 0) ? (float)sqrt(fmax(0.0, (sumMagSq / cnt) - (sumMag / cnt) * (sumMag / cnt))) : 0.0f;

  // ---- Peak-interval based HR (on normalized PPG, threshold = its own mean) ----
  float ppgMean = 0;
  for (int i = 0; i < n; i++) ppgMean += normPpg[i];
  ppgMean /= n;

  static int peakIdx[64];
  int minDist = (int)(SAMPLE_RATE_HZ * 0.4f); // matches Python distance=fs*0.4
  int nPeaks = findPeaks(normPpg, n, ppgMean, minDist, peakIdx, 64);

  if (nPeaks > 1) {
    float ibiSum = 0, ibiSumSq = 0;
    int nIbi = nPeaks - 1;
    float ibis[64];
    for (int i = 0; i < nIbi; i++) {
      ibis[i] = (float)(peakIdx[i + 1] - peakIdx[i]) / (float)SAMPLE_RATE_HZ; // seconds
      ibiSum += ibis[i];
    }
    float ibiMean = ibiSum / nIbi;
    ppgHrEst = 60.0f / ibiMean;
    for (int i = 0; i < nIbi; i++) {
      float d = ibis[i] - ibiMean;
      ibiSumSq += d * d;
    }
    ibiStd = sqrtf(ibiSumSq / nIbi);
  } else {
    ppgHrEst = 0;
    ibiStd = 0;
  }
}

// ECG-only features: SDNN / RMSSD from R-R intervals (raw ECG, NOT
// normalized — only peak timing matters, amplitude scale is irrelevant)
void computeECGFeatures(const float* ecg, int n, float &sdnn, float &rmssd) {
  float ecgMean = 0, ecgStd = 0;
  for (int i = 0; i < n; i++) ecgMean += ecg[i];
  ecgMean /= n;
  for (int i = 0; i < n; i++) {
    float d = ecg[i] - ecgMean;
    ecgStd += d * d;
  }
  ecgStd = sqrtf(ecgStd / n);

  static int rPeaks[64];
  int minDist = (int)(SAMPLE_RATE_HZ * 0.4f);
  float heightThresh = ecgMean + 0.5f * ecgStd;
  int nPeaks = findPeaks(ecg, n, heightThresh, minDist, rPeaks, 64);

  if (nPeaks > 2) {
    int nRR = nPeaks - 1;
    float rr[64];
    for (int i = 0; i < nRR; i++) {
      rr[i] = (float)(rPeaks[i + 1] - rPeaks[i]) / (float)SAMPLE_RATE_HZ * 1000.0f; // ms
    }
    float rrMean = 0;
    for (int i = 0; i < nRR; i++) rrMean += rr[i];
    rrMean /= nRR;
    float sdSum = 0;
    for (int i = 0; i < nRR; i++) {
      float d = rr[i] - rrMean;
      sdSum += d * d;
    }
    sdnn = sqrtf(sdSum / nRR);

    float diffSqSum = 0;
    int nDiff = nRR - 1;
    if (nDiff > 0) {
      for (int i = 0; i < nDiff; i++) {
        float diff = rr[i + 1] - rr[i];
        diffSqSum += diff * diff;
      }
      rmssd = sqrtf(diffSqSum / nDiff);
    } else {
      rmssd = 0;
    }
  } else {
    sdnn = 0;
    rmssd = 0;
  }
}

// Stress proxy — same formula as Step 14 (Python): low RMSSD -> high stress
float rmssdStressProxy(float rmssdMs) {
  if (rmssdMs <= 0.0f) return 0.0f;
  return 100.0f / (1.0f + (rmssdMs / 30.0f));
}

// =============================================================================
// MODEL INFERENCE — builds the 16-feature int16 vector and calls the 3
// exported Random Forest models (replaces the old TFLite runInference())
// =============================================================================
void runModelInference(Telemetry &out) {
  // Work on local copies — normalization is destructive for PPG
  static float ppgNorm[WINDOW_LEN];
  memcpy(ppgNorm, ppgBuffer, sizeof(ppgNorm));
  normalizeWindow(ppgNorm, WINDOW_LEN);

  float mean, stdv, minV, maxV, ptp, skewV, kurtV, rms, zcr;
  computeBasicStats(ppgNorm, WINDOW_LEN, mean, stdv, minV, maxV, ptp, skewV, kurtV, rms, zcr);

  float domFreq, psdMean, psdStd, ppgHrEst, ibiStd;
  computePPGFeatures(ppgNorm, WINDOW_LEN, domFreq, psdMean, psdStd, ppgHrEst, ibiStd);

  float sdnn, rmssd;
  computeECGFeatures(ecgBuffer, WINDOW_LEN, sdnn, rmssd);

  float rawFeatures[NUM_FEATURES] = {
    mean, stdv, minV, maxV, ptp, skewV, kurtV, rms, zcr,
    domFreq, psdMean, psdStd, ppgHrEst, ibiStd, sdnn, rmssd
  };

  int16_t features[NUM_FEATURES];
  for (int i = 0; i < NUM_FEATURES; i++) {
    features[i] = clampInt16(rawFeatures[i] * FEATURE_SCALE[i]);
  }

  float hrPred   = model_hr_predict(features, NUM_FEATURES);
  float respPred = model_resp_predict(features, NUM_FEATURES);
  float spo2Pred = model_spo2_predict(features, NUM_FEATURES);
  float stress   = rmssdStressProxy(rmssd);

  out.heartRateBPM = hrPred;
  out.respRate     = respPred;
  out.spo2Pct      = spo2Pred;
  out.stressIndex  = stress;
  out.hrValid      = true;
}

// =============================================================================
// ALERTS — non-blocking buzzer + vibrator pulse pattern (unchanged)
// =============================================================================
void updateAlertActuators(bool shouldAlert) {
  static bool pulseOn = false;
  static unsigned long lastToggle = 0;
  const unsigned long PULSE_MS = 300;
  if (!shouldAlert) {
    digitalWrite(PIN_BUZZER, LOW);
    digitalWrite(PIN_VIBRATOR, LOW);
    pulseOn = false;
    return;
  }
  unsigned long now = millis();
  if (now - lastToggle >= PULSE_MS) {
    lastToggle = now;
    pulseOn = !pulseOn;
    digitalWrite(PIN_BUZZER, pulseOn ? HIGH : LOW);
    digitalWrite(PIN_VIBRATOR, pulseOn ? HIGH : LOW);
  }
}

bool evaluateRuleBasedAnomaly(const Telemetry &t) {
  bool hrBad     = t.hrValid && (t.heartRateBPM < HR_MIN_SAFE || t.heartRateBPM > HR_MAX_SAFE);
  bool spo2Bad   = t.hrValid && (t.spo2Pct < SPO2_MIN_SAFE);
  bool tempBad   = (t.bodyTempC > BODY_TEMP_MAX || t.bodyTempC < BODY_TEMP_MIN);
  bool heatRisk  = (t.ambientTempC > AMBIENT_TEMP_HEAT && t.humidityPct > HUMIDITY_HEAT);
  bool airBad    = (t.mq135Raw > MQ135_RAW_DANGER);
  return hrBad || spo2Bad || tempBad || heatRisk || airBad;
}

// Derives the coarse RiskLevel (used by the OLED/dashboard) directly from
// our RF model outputs + existing env rules — replaces the old TFLite
// classifier's role, but via thresholds instead of a learned classifier.
RiskLevel computeAIRisk(const Telemetry &t) {
  if (!t.hrValid) return RISK_NORMAL;

  bool critical =
      t.spo2Pct < 88.0f ||
      t.heartRateBPM > 150.0f || t.heartRateBPM < 35.0f ||
      t.respRate > 30.0f || t.respRate < 8.0f ||
      t.bodyTempC > 40.0f;

  bool caution =
      t.spo2Pct < 92.0f ||
      t.heartRateBPM > 100.0f || t.heartRateBPM < 50.0f ||
      t.respRate > 20.0f || t.respRate < 12.0f ||
      t.stressIndex > 60.0f;

  if (critical) return RISK_CRITICAL;
  if (caution)  return RISK_CAUTION;
  return RISK_NORMAL;
}

// =============================================================================
// SUGGESTION ENGINE — merges the original environmental/heat/UV rules with
// our model's HR/RESP/SpO2/Stress thresholds (Step 20 in the Python work),
// kept SHORT for the small AMOLED screen. Priority order = most urgent first,
// first match wins.
// =============================================================================
void generateSuggestion(const Telemetry &t, char* outBuf, size_t outLen) {
  // 1) Critical vitals (from our RF models) or extreme body temp -> most urgent
  if (t.hrValid && t.spo2Pct < 88.0f) {
    snprintf(outBuf, outLen, "SpO2 very low. Seek help now."); return;
  }
  if (t.hrValid && (t.heartRateBPM > 150.0f || t.heartRateBPM < 35.0f)) {
    snprintf(outBuf, outLen, "HR critical. Seek help now."); return;
  }
  if (t.hrValid && (t.respRate > 30.0f || t.respRate < 8.0f)) {
    snprintf(outBuf, outLen, "Breathing abnormal. Seek help."); return;
  }
  if (t.bodyTempC > 40.0f) {
    snprintf(outBuf, outLen, "URGENT: High body temp. See a doctor."); return;
  }

  // 2) Heat-stress combination
  if ((t.ambientTempC > AMBIENT_TEMP_HEAT && t.humidityPct > HUMIDITY_HEAT) ||
      (t.bodyTempC > BODY_TEMP_MAX)) {
    snprintf(outBuf, outLen, "Heat stress risk: Drink water, rest."); return;
  }

  // 3) Low body temperature
  if (t.bodyTempC < BODY_TEMP_MIN && t.bodyTempC > 0.0f) {
    snprintf(outBuf, outLen, "Low body temp. Warm up, stay dry."); return;
  }

  // 4) Moderate SpO2 low
  if (t.hrValid && t.spo2Pct >= 88.0f && t.spo2Pct < SPO2_MIN_SAFE) {
    snprintf(outBuf, outLen, "SpO2 below normal. See a doctor."); return;
  }
  if (t.hrValid && t.spo2Pct >= SPO2_MIN_SAFE && t.spo2Pct < 95.0f) {
    snprintf(outBuf, outLen, "SpO2 slightly low. Rest & monitor."); return;
  }

  // 5) Abnormal heart rate (moderate)
  if (t.hrValid && (t.heartRateBPM > HR_MAX_SAFE || t.heartRateBPM < HR_MIN_SAFE)) {
    snprintf(outBuf, outLen, "HR abnormal. Rest & monitor."); return;
  }
  if (t.hrValid && t.heartRateBPM > 100.0f) {
    snprintf(outBuf, outLen, "HR high. Rest & hydrate."); return;
  }

  // 6) Abnormal respiration (moderate)
  if (t.hrValid && t.respRate > 24.0f) {
    snprintf(outBuf, outLen, "Fast breathing. See a doctor."); return;
  }
  if (t.hrValid && (t.respRate > 20.0f || t.respRate < 12.0f)) {
    snprintf(outBuf, outLen, "Breathing slightly off. Monitor."); return;
  }

  // 7) Poor ambient air quality
  if (t.mq135Raw > MQ135_RAW_DANGER) {
    snprintf(outBuf, outLen, "Poor air quality. Move indoors."); return;
  }

  // 8) High UV exposure
  if (t.uvIndexApprox > UV_INDEX_HIGH) {
    snprintf(outBuf, outLen, "High UV. Use sunscreen, seek shade."); return;
  }

  // 9) Elevated stress — either skin conductance (GSR) or HRV-based stressIndex
  if (t.hrValid && t.stressIndex > 85.0f) {
    snprintf(outBuf, outLen, "High stress. Take a break."); return;
  }
  if (t.gsrRaw > GSR_STRESS_RAW || (t.hrValid && t.stressIndex > 60.0f)) {
    snprintf(outBuf, outLen, "Stress elevated. Rest a bit."); return;
  }

  // 10) ECG electrodes not making contact
  if (t.ecgLeadsOff) {
    snprintf(outBuf, outLen, "ECG leads off. Reposition them."); return;
  }

  // 11) Nothing flagged
  snprintf(outBuf, outLen, "Vitals normal. Keep it up.");
}

// =============================================================================
// SENSOR TASK  (runs on Core 0)
// =============================================================================

// Continuously samples raw PPG (MAX30102 IR) + raw ECG (AD8232) into the
// window buffers. Called once per task iteration; the task loop is timed to
// ~125Hz (see vTaskDelay at the bottom of sensorTask) to approximate the
// dataset's sampling rate.
void sampleWaveforms() {
  particleSensor.check();
  uint32_t ir = particleSensor.getIR();
  particleSensor.nextSample();
  int ecgRawVal = analogRead(PIN_ECG_OUTPUT);
  bool leadsOff = (digitalRead(PIN_ECG_LOPLUS) == HIGH) || (digitalRead(PIN_ECG_LOMINUS) == HIGH);

  if (bufIdx < WINDOW_LEN) {
    ppgBuffer[bufIdx] = (float)ir;
    ecgBuffer[bufIdx] = (float)ecgRawVal;
    bufIdx++;
  }
  if (bufIdx >= WINDOW_LEN) {
    bufferFull = true; // consumed + reset by sensorTask below
  }

  xSemaphoreTake(dataMutex, portMAX_DELAY);
  telemetry.ecgRaw = ecgRawVal;
  telemetry.ecgLeadsOff = leadsOff;
  xSemaphoreGive(dataMutex);
}

void pollDS18B20() {
  static uint8_t state = 0;
  static unsigned long convStart = 0;
  if (state == 0) {
    ds18b20.requestTemperatures();
    convStart = millis();
    state = 1;
  } else if (millis() - convStart >= 750) {
    float tempC = ds18b20.getTempCByIndex(0);
    if (tempC > -50 && tempC < 100) {
      xSemaphoreTake(dataMutex, portMAX_DELAY);
      telemetry.bodyTempC = tempC;
      xSemaphoreGive(dataMutex);
    }
    state = 0;
  }
}

void pollDHT22() {
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  if (!isnan(h) && !isnan(t)) {
    xSemaphoreTake(dataMutex, portMAX_DELAY);
    telemetry.humidityPct  = h;
    telemetry.ambientTempC = t;
    xSemaphoreGive(dataMutex);
  }
}

void pollBMP280() {
  float p = bmp.readPressure() / 100.0F;
  xSemaphoreTake(dataMutex, portMAX_DELAY);
  telemetry.pressureHPa = p;
  xSemaphoreGive(dataMutex);
}

void pollAnalogSensors() {
  int mq135 = analogRead(PIN_MQ135);
  int uv    = analogRead(PIN_UV);
  int gsr   = analogRead(PIN_GSR);

  float voltage = uv * (3.3f / 4095.0f);
  float uvIndex = voltage / 0.1f; // placeholder — calibrate against your UV breakout

  xSemaphoreTake(dataMutex, portMAX_DELAY);
  telemetry.mq135Raw      = mq135;
  telemetry.uvRaw         = uv;
  telemetry.uvIndexApprox = uvIndex;
  telemetry.gsrRaw        = gsr;
  xSemaphoreGive(dataMutex);
}

void pollAccelerometer() {
  float gx, gy, gz;
  if (mpu6050ReadAccel(gx, gy, gz)) {
    float mag = sqrtf(gx * gx + gy * gy + gz * gz);
    const char* state;
    if (fabsf(mag - 1.0f) < 0.05f)      state = "Stationary";
    else if (fabsf(mag - 1.0f) < 0.3f)  state = "Moving";
    else                                 state = "Active/Impact";

    xSemaphoreTake(dataMutex, portMAX_DELAY);
    telemetry.accelMagG    = mag;
    telemetry.activityState = state;
    xSemaphoreGive(dataMutex);
  }
}

void sensorTask(void* pvParameters) {
  unsigned long tDht = 0, tBmp = 0, tAnalog = 0, tAccel = 0;

  for (;;) {
    unsigned long now = millis();

    sampleWaveforms();  // ~125Hz raw PPG+ECG capture into the 8s window buffers
    pollDS18B20();

    if (now - tDht >= T_DHT_MS)       { tDht = now;    pollDHT22(); }
    if (now - tBmp >= T_BMP_MS)       { tBmp = now;    pollBMP280(); }
    if (now - tAnalog >= T_ANALOG_MS) { tAnalog = now; pollAnalogSensors(); }
    if (now - tAccel >= T_ACCEL_MS)   { tAccel = now;  pollAccelerometer(); }

    if (bufferFull) {
      Telemetry snap = getTelemetrySnapshot();
      runModelInference(snap);                 // fills HR/RESP/SpO2/Stress
      RiskLevel risk = computeAIRisk(snap);
      bool anomaly = evaluateRuleBasedAnomaly(snap);

      snap.aiRisk = risk;
      snap.ruleAnomaly = anomaly;
      char suggBuf[80];
      generateSuggestion(snap, suggBuf, sizeof(suggBuf));

      xSemaphoreTake(dataMutex, portMAX_DELAY);
      telemetry.heartRateBPM = snap.heartRateBPM;
      telemetry.respRate     = snap.respRate;
      telemetry.spo2Pct      = snap.spo2Pct;
      telemetry.stressIndex  = snap.stressIndex;
      telemetry.hrValid      = snap.hrValid;
      telemetry.aiRisk       = risk;
      telemetry.ruleAnomaly  = anomaly;
      telemetry.alertActive  = anomaly || (risk == RISK_CRITICAL);
      strncpy(telemetry.suggestion, suggBuf, sizeof(telemetry.suggestion) - 1);
      telemetry.suggestion[sizeof(telemetry.suggestion) - 1] = '\0';
      telemetry.lastUpdateMs = now;
      xSemaphoreGive(dataMutex);

      // Reset window for the next 8s (non-overlapping windows on-device,
      // simpler than the Python training's 2s-step sliding window)
      bufIdx = 0;
      bufferFull = false;
    }

    bool alertNow;
    xSemaphoreTake(dataMutex, portMAX_DELAY);
    alertNow = telemetry.alertActive;
    xSemaphoreGive(dataMutex);
    updateAlertActuators(alertNow);

    vTaskDelay(pdMS_TO_TICKS(8)); // ~125Hz task cadence, matches SAMPLE_RATE_HZ
  }
}

// =============================================================================
// WEB SERVER — DASHBOARD (Core 1 / main loop)
// =============================================================================
const char INDEX_HTML[] PROGMEM = R"HTML(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Health Companion Dashboard</title>
<style>
:root{--ok:#2ecc71;--warn:#f39c12;--bad:#e74c3c;--bg:#0b2545;}
*{box-sizing:border-box;}
body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#0e1a2b;color:#eaf2f8;margin:0;padding:16px;}
h1{font-size:1.2rem;margin:0 0 4px;}
.sub{color:#9fb3c8;font-size:0.8rem;margin-bottom:14px;}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;}
.card{background:#132b45;border-radius:10px;padding:12px;border:1px solid #1f4066;}
.label{font-size:0.72rem;color:#9fb3c8;text-transform:uppercase;letter-spacing:.04em;}
.value{font-size:1.4rem;font-weight:600;margin-top:4px;}
.unit{font-size:0.8rem;color:#9fb3c8;margin-left:4px;}
#riskBanner{margin-top:14px;padding:12px;border-radius:10px;font-weight:700;text-align:center;background:var(--ok);color:#08210f;}
.risk-1{background:var(--warn)!important;color:#3a2600!important;}
.risk-2{background:var(--bad)!important;color:#fff!important;}
#suggestionBox{margin-top:10px;padding:14px;border-radius:10px;background:#132b45;border:1px solid #1f4066;text-align:center;font-size:1rem;}
#suggestionBox .label{margin-bottom:6px;}
#suggestionBox.risk-1{background:#3a2600;border-color:var(--warn);color:#ffd58a;}
#suggestionBox.risk-2{background:#3a0d0d;border-color:var(--bad);color:#ffb3ab;}
footer{margin-top:16px;color:#69819a;font-size:0.72rem;text-align:center;}
</style>
</head>
<body>
<h1>AI Personal Health Companion</h1>
<div class="sub">Local telemetry &middot; auto-refreshing every 2s</div>
<div class="grid">
<div class="card"><div class="label">Heart Rate</div><div class="value" id="hr">--<span class="unit">bpm</span></div></div>
<div class="card"><div class="label">SpO2</div><div class="value" id="spo2">--<span class="unit">%</span></div></div>
<div class="card"><div class="label">Resp Rate</div><div class="value" id="resp">--<span class="unit">/min</span></div></div>
<div class="card"><div class="label">Stress</div><div class="value" id="stress">--</div></div>
<div class="card"><div class="label">Body Temp</div><div class="value" id="btemp">--<span class="unit">&deg;C</span></div></div>
<div class="card"><div class="label">ECG Leads</div><div class="value" id="ecg">--</div></div>
<div class="card"><div class="label">Ambient Temp</div><div class="value" id="atemp">--<span class="unit">&deg;C</span></div></div>
<div class="card"><div class="label">Humidity</div><div class="value" id="hum">--<span class="unit">%</span></div></div>
<div class="card"><div class="label">Pressure</div><div class="value" id="pres">--<span class="unit">hPa</span></div></div>
<div class="card"><div class="label">Air Quality (raw)</div><div class="value" id="aq">--</div></div>
<div class="card"><div class="label">UV Index</div><div class="value" id="uv">--</div></div>
<div class="card"><div class="label">GSR (raw)</div><div class="value" id="gsr">--</div></div>
<div class="card"><div class="label">Activity</div><div class="value" id="act">--</div></div>
<div class="card"><div class="label">Device IP</div><div class="value" id="ip" style="font-size:1rem">--</div></div>
</div>
<div id="riskBanner">AI Risk Level: <span id="risk">NORMAL</span></div>
<div id="suggestionBox">
<div class="label" style="color:#9fb3c8;font-size:0.72rem;text-transform:uppercase;">Recommended Action</div>
<div id="suggestionText">--</div>
</div>
<footer>ESP32-S3 Personal Health Companion &middot; SIH26181</footer>
<script>
async function refresh(){
try{
const r = await fetch('/data');
const d = await r.json();
document.getElementById('hr').innerHTML   = (d.hrValid? d.hr.toFixed(0):'--') + '<span class="unit">bpm</span>';
document.getElementById('spo2').innerHTML = (d.hrValid? d.spo2.toFixed(0):'--') + '<span class="unit">%</span>';
document.getElementById('resp').innerHTML = (d.hrValid? d.resp.toFixed(0):'--') + '<span class="unit">/min</span>';
document.getElementById('stress').innerText = (d.hrValid? d.stress.toFixed(0):'--');
document.getElementById('btemp').innerHTML= d.bodyTemp.toFixed(1) + '<span class="unit">&deg;C</span>';
document.getElementById('ecg').innerText  = d.ecgLeadsOff ? 'Leads Off' : 'Attached';
document.getElementById('atemp').innerHTML= d.ambientTemp.toFixed(1) + '<span class="unit">&deg;C</span>';
document.getElementById('hum').innerHTML  = d.humidity.toFixed(0) + '<span class="unit">%</span>';
document.getElementById('pres').innerHTML = d.pressure.toFixed(0) + '<span class="unit">hPa</span>';
document.getElementById('aq').innerText   = d.mq135;
document.getElementById('uv').innerText   = d.uvIndex.toFixed(1);
document.getElementById('gsr').innerText  = d.gsr;
document.getElementById('act').innerText  = d.activity;
document.getElementById('ip').innerText   = d.ip;
const riskNames = ['NORMAL','CAUTION','CRITICAL'];
document.getElementById('risk').innerText = riskNames[d.risk];
const riskClass = d.risk === 1 ? 'risk-1' : (d.risk === 2 ? 'risk-2' : '');
document.getElementById('riskBanner').className = riskClass;
document.getElementById('suggestionText').innerText = d.suggestion;
document.getElementById('suggestionBox').className = riskClass;
}catch(e){ console.error(e); }
}
setInterval(refresh, 2000);
refresh();
</script>
</body>
</html>
)HTML";

void handleRoot() {
  server.send_P(200, "text/html", INDEX_HTML);
}

void handleData() {
  Telemetry t = getTelemetrySnapshot();
  StaticJsonDocument<512> doc;

  doc["hr"]          = t.heartRateBPM;
  doc["hrValid"]     = t.hrValid;
  doc["spo2"]        = t.spo2Pct;
  doc["resp"]        = t.respRate;
  doc["stress"]      = t.stressIndex;
  doc["bodyTemp"]    = t.bodyTempC;
  doc["ecgLeadsOff"] = t.ecgLeadsOff;
  doc["ecgRaw"]      = t.ecgRaw;

  doc["ambientTemp"] = t.ambientTempC;
  doc["humidity"]    = t.humidityPct;
  doc["pressure"]    = t.pressureHPa;
  doc["mq135"]       = t.mq135Raw;
  doc["uvRaw"]       = t.uvRaw;
  doc["uvIndex"]     = t.uvIndexApprox;

  doc["gsr"]         = t.gsrRaw;
  doc["accelG"]      = t.accelMagG;
  doc["activity"]    = t.activityState;

  doc["risk"]        = (int)t.aiRisk;
  doc["ruleAnomaly"]  = t.ruleAnomaly;
  doc["alertActive"]  = t.alertActive;
  doc["suggestion"]   = t.suggestion;
  doc["ip"]           = WiFi.softAPIP().toString();

  String out;
  serializeJson(doc, out);
  server.send(200, "application/json", out);
}

void setupWebServer() {
  server.on("/", HTTP_GET, handleRoot);
  server.on("/data", HTTP_GET, handleData);
  server.begin();
}

// =============================================================================
// OLED UI — non-blocking push-button page toggle (Core 1 / main loop)
// =============================================================================
uint8_t currentPage = 0;
const uint8_t NUM_PAGES = 4;

void handleButtonAndPaging() {
  static int lastReading = HIGH;
  static int stableState = HIGH;
  static unsigned long lastDebounceTime = 0;
  int reading = digitalRead(PIN_BUTTON);
  if (reading != lastReading) {
    lastDebounceTime = millis();
  }
  if ((millis() - lastDebounceTime) > DEBOUNCE_MS) {
    if (reading != stableState) {
      stableState = reading;
      if (stableState == LOW) {
        currentPage = (currentPage + 1) % NUM_PAGES;
      }
    }
  }
  lastReading = reading;
}

void renderPageVitals(const Telemetry &t) {
  display.setCursor(0, 0);
  display.println("VITALS SUMMARY");
  display.drawLine(0, 10, 128, 10, SSD1306_WHITE);
  display.setCursor(0, 16);
  display.printf("HR : %s bpm\n", t.hrValid ? String((int)t.heartRateBPM).c_str() : "--");
  display.printf("SpO2: %s %%\n", t.hrValid ? String((int)t.spo2Pct).c_str() : "--");
  display.printf("Resp: %s /min\n", t.hrValid ? String((int)t.respRate).c_str() : "--");
  display.printf("Body T: %.1f C\n", t.bodyTempC);
}

void renderPageEnvironment(const Telemetry &t) {
  display.setCursor(0, 0);
  display.println("ENVIRONMENT");
  display.drawLine(0, 10, 128, 10, SSD1306_WHITE);
  display.setCursor(0, 16);
  display.printf("Amb T: %.1f C\n", t.ambientTempC);
  display.printf("Humidity: %.0f %%\n", t.humidityPct);
  display.printf("Pressure: %.0f hPa\n", t.pressureHPa);
  display.printf("AQ(raw): %d  UV:%.1f\n", t.mq135Raw, t.uvIndexApprox);
}

void renderPageAIStatus(const Telemetry &t) {
  const char* riskStr = t.aiRisk == RISK_CRITICAL ? "CRITICAL" :
                        t.aiRisk == RISK_CAUTION  ? "CAUTION"  : "NORMAL";
  display.setCursor(0, 0);
  display.println("STRESS / AI STATUS");
  display.drawLine(0, 10, 128, 10, SSD1306_WHITE);
  display.setCursor(0, 16);
  display.printf("Stress: %s\n", t.hrValid ? String((int)t.stressIndex).c_str() : "--");
  display.printf("GSR: %d\n", t.gsrRaw);
  display.printf("Activity: %s\n", t.activityState);
  display.printf("AI Risk: %s\n", riskStr);
}

void printWrapped(const char* text, uint8_t maxLines) {
  const uint8_t charsPerLine = 21;
  char buf[128];
  strncpy(buf, text, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';

  char* word = strtok(buf, " ");
  char line[24] = "";
  uint8_t linesPrinted = 0;

  while (word != nullptr && linesPrinted < maxLines) {
    if (strlen(line) + strlen(word) + 1 > charsPerLine) {
      display.println(line);
      linesPrinted++;
      line[0] = '\0';
      if (linesPrinted >= maxLines) break;
    }
    if (strlen(line) > 0) strcat(line, " ");
    strcat(line, word);
    word = strtok(nullptr, " ");
  }
  if (linesPrinted < maxLines && strlen(line) > 0) {
    display.println(line);
  }
}

void renderPageSuggestion(const Telemetry &t) {
  const char* riskStr = t.aiRisk == RISK_CRITICAL ? "CRITICAL" :
                        t.aiRisk == RISK_CAUTION  ? "CAUTION"  : "NORMAL";
  display.setCursor(0, 0);
  display.printf("AI SUGGESTION [%s]\n", riskStr);
  display.drawLine(0, 10, 128, 10, SSD1306_WHITE);
  display.setCursor(0, 16);
  printWrapped(t.suggestion, 6);
}

void updateOLED() {
  static unsigned long lastRefresh = 0;
  if (millis() - lastRefresh < OLED_REFRESH_MS) return;
  lastRefresh = millis();

  Telemetry t = getTelemetrySnapshot();

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  switch (currentPage) {
    case 0: renderPageVitals(t);      break;
    case 1: renderPageEnvironment(t); break;
    case 2: renderPageAIStatus(t);    break;
    case 3: renderPageSuggestion(t);  break;
  }

  display.display();
}

// =============================================================================
// SETUP
// =============================================================================
void setup() {
  Serial.begin(115200);
  delay(200);

  pinMode(PIN_BUTTON, INPUT_PULLUP);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_VIBRATOR, OUTPUT);
  pinMode(PIN_ECG_LOPLUS, INPUT);
  pinMode(PIN_ECG_LOMINUS, INPUT);
  digitalWrite(PIN_BUZZER, LOW);
  digitalWrite(PIN_VIBRATOR, LOW);

  analogReadResolution(12);

  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(400000);

  // --- OLED ---
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("SSD1306 init failed");
  }
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 20);
  display.println("Health Companion");
  display.println("Booting...");
  display.display();

  // --- DHT22 ---
  dht.begin();

  // --- DS18B20 ---
  ds18b20.begin();
  ds18b20.setWaitForConversion(false);

  // --- BMP280 ---
  if (!bmp.begin(BMP280_ADDR)) {
    Serial.println("BMP280 init failed");
  }

  // --- MAX30102 ---
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found");
  } else {
    particleSensor.setup();
  }

  // --- MPU6050 ---
  if (!mpu6050Init()) {
    Serial.println("MPU6050 init failed / WHO_AM_I mismatch");
  }

  // --- Mutex ---
  dataMutex = xSemaphoreCreateMutex();

  // --- Wi-Fi Access Point ---
  WiFi.softAP(AP_SSID, AP_PASSWORD);
  Serial.print("AP IP address: ");
  Serial.println(WiFi.softAPIP());

  // --- Web server ---
  setupWebServer();

  // --- Launch sensor+AI task pinned to Core 0 ---
  xTaskCreatePinnedToCore(
      sensorTask, "SensorTask", 12288, nullptr, 1, &sensorTaskHandle, 0);

  Serial.println("Setup complete. (Random Forest models active, no TinyML/TFLite)");
}

// =============================================================================
// LOOP  (Core 1: web server + button/UI)
// =============================================================================
void loop() {
  server.handleClient();
  handleButtonAndPaging();
  updateOLED();
}

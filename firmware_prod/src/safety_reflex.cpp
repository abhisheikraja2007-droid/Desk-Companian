#include "safety_reflex.h"
#include "drive_pid.h"

SafetyReflex safetyReflex;

static volatile bool cliff_fl_flag = false;
static volatile bool cliff_fr_flag = false;
static volatile bool cliff_rl_flag = false;
static volatile bool cliff_rr_flag = false;
static volatile bool touch_flag = false;
static volatile uint32_t cliff_trigger_time_us = 0;

void IRAM_ATTR SafetyReflex::isrCliffFL() {
    cliff_fl_flag = true;
    cliff_trigger_time_us = micros();
    // <2 ms Deterministic emergency motor cutoff in hardware interrupt
    driveController.emergencyBrakeFromISR();
}

void IRAM_ATTR SafetyReflex::isrCliffFR() {
    cliff_fr_flag = true;
    cliff_trigger_time_us = micros();
    driveController.emergencyBrakeFromISR();
}

void IRAM_ATTR SafetyReflex::isrCliffRL() {
    cliff_rl_flag = true;
    cliff_trigger_time_us = micros();
    driveController.emergencyBrakeFromISR();
}

void IRAM_ATTR SafetyReflex::isrCliffRR() {
    cliff_rr_flag = true;
    cliff_trigger_time_us = micros();
    driveController.emergencyBrakeFromISR();
}

void IRAM_ATTR SafetyReflex::isrTouch() {
    touch_flag = (digitalRead(PIN_TOUCH) == HIGH);
}

SafetyReflex::SafetyReflex() : mpu_available(false) {
    memset(&telemetry, 0, sizeof(telemetry));
}

void SafetyReflex::begin() {
    pinMode(PIN_CLIFF_FL, INPUT);
    pinMode(PIN_CLIFF_FR, INPUT);
    pinMode(PIN_CLIFF_RL, INPUT);
    pinMode(PIN_CLIFF_RR, INPUT);
    pinMode(PIN_TOUCH, INPUT);

    attachInterrupt(digitalPinToInterrupt(PIN_CLIFF_FL), isrCliffFL, RISING);
    attachInterrupt(digitalPinToInterrupt(PIN_CLIFF_FR), isrCliffFR, RISING);
    attachInterrupt(digitalPinToInterrupt(PIN_CLIFF_RL), isrCliffRL, RISING);
    attachInterrupt(digitalPinToInterrupt(PIN_CLIFF_RR), isrCliffRR, RISING);
    attachInterrupt(digitalPinToInterrupt(PIN_TOUCH), isrTouch, CHANGE);

    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000);
    if (mpu.begin(0x68, &Wire)) {
        mpu_available = true;
        mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
        mpu.setGyroRange(MPU6050_RANGE_500_DEG);
        mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    }
}

void SafetyReflex::update() {
    bool raw_fl = (digitalRead(PIN_CLIFF_FL) == HIGH) || cliff_fl_flag;
    bool raw_fr = (digitalRead(PIN_CLIFF_FR) == HIGH) || cliff_fr_flag;
    bool raw_rl = (digitalRead(PIN_CLIFF_RL) == HIGH) || cliff_rl_flag;
    bool raw_rr = (digitalRead(PIN_CLIFF_RR) == HIGH) || cliff_rr_flag;

    telemetry.cliff_fl = raw_fl;
    telemetry.cliff_fr = raw_fr;
    telemetry.cliff_rl = raw_rl;
    telemetry.cliff_rr = raw_rr;
    telemetry.any_cliff = (raw_fl || raw_fr || raw_rl || raw_rr);
    telemetry.crown_touched = (digitalRead(PIN_TOUCH) == HIGH) || touch_flag;
    telemetry.last_cliff_time_us = cliff_trigger_time_us;

    if (mpu_available) {
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);
        telemetry.accel_x = a.acceleration.x;
        telemetry.accel_y = a.acceleration.y;
        telemetry.accel_z = a.acceleration.z;
        telemetry.gyro_x = g.gyro.x;
        telemetry.gyro_y = g.gyro.y;
        telemetry.gyro_z = g.gyro.z;

        // Upright Z-acceleration check: when tilted > 45 deg, az drops below 6.5 m/s^2
        telemetry.is_tilted = (telemetry.accel_z < 6.5f);
        // Free-fall or violent pickup check: total acceleration deviation from 9.81
        float mag = sqrtf(a.acceleration.x * a.acceleration.x +
                          a.acceleration.y * a.acceleration.y +
                          a.acceleration.z * a.acceleration.z);
        telemetry.is_picked_up = (mag < 4.0f || mag > 16.0f);
    }
}

SafetyTelemetry SafetyReflex::getTelemetry() const {
    return telemetry;
}

bool SafetyReflex::isCliffTriggered() const {
    return telemetry.any_cliff;
}

void SafetyReflex::resetCliffStatus() {
    cliff_fl_flag = false;
    cliff_fr_flag = false;
    cliff_rl_flag = false;
    cliff_rr_flag = false;
    touch_flag = false;
    telemetry.any_cliff = false;
}

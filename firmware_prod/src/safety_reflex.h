#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>

#define PIN_CLIFF_FL   10
#define PIN_CLIFF_FR   11
#define PIN_CLIFF_RL   12
#define PIN_CLIFF_RR   13

#define PIN_TOUCH      14

#define PIN_I2C_SDA    8
#define PIN_I2C_SCL    9

struct SafetyTelemetry {
    bool cliff_fl;
    bool cliff_fr;
    bool cliff_rl;
    bool cliff_rr;
    bool any_cliff;
    bool crown_touched;
    float accel_x;
    float accel_y;
    float accel_z;
    float gyro_x;
    float gyro_y;
    float gyro_z;
    bool is_tilted;
    bool is_picked_up;
    uint32_t last_cliff_time_us;
};

class SafetyReflex {
public:
    SafetyReflex();
    void begin();
    void update();
    SafetyTelemetry getTelemetry() const;
    bool isCliffTriggered() const;
    void resetCliffStatus();

    static void IRAM_ATTR isrCliffFL();
    static void IRAM_ATTR isrCliffFR();
    static void IRAM_ATTR isrCliffRL();
    static void IRAM_ATTR isrCliffRR();
    static void IRAM_ATTR isrTouch();

private:
    Adafruit_MPU6050 mpu;
    bool mpu_available;
    SafetyTelemetry telemetry;
};

extern SafetyReflex safetyReflex;

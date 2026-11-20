#pragma once

#include <Arduino.h>
#include <ESP32Servo.h>

#define SCSERVO_BAUDRATE 1000000
#define NUM_BUS_SERVOS   6
#define NUM_PWM_SERVOS   2
#define TOTAL_SERVOS     8

#define PIN_SCS_TX       43
#define PIN_SCS_RX       44

#define PIN_PWM_NECK_YAW    4
#define PIN_PWM_NECK_PITCH  5
#define PIN_PWM_EAR_LEFT    6
#define PIN_PWM_EAR_RIGHT   7
#define PIN_PWM_ARM_LEFT    15
#define PIN_PWM_ARM_RIGHT   16
#define PIN_PWM_PINCER_L    17
#define PIN_PWM_PINCER_R    18

class SCServoBus {
public:
    SCServoBus();
    void begin();
    void setTargetDegrees(uint8_t index, float deg, uint32_t duration_ms = 250);
    void setAllTargetsDegrees(const float deg[TOTAL_SERVOS], uint32_t duration_ms = 250);
    void update(uint32_t now_ms);
    float getCurrentDegrees(uint8_t index) const;
    void emergencyStop();

private:
    void _writeSCServoPos(uint8_t id, uint16_t position, uint16_t time_ms);
    float _evaluateSCurve(float start, float target, float progress);

    float current_deg[TOTAL_SERVOS];
    float start_deg[TOTAL_SERVOS];
    float target_deg[TOTAL_SERVOS];
    uint32_t move_start_ms[TOTAL_SERVOS];
    uint32_t move_duration_ms[TOTAL_SERVOS];
    bool is_moving[TOTAL_SERVOS];

    Servo pwm_servos[TOTAL_SERVOS];
    bool use_hardware_scservo;
};

extern SCServoBus servoBus;

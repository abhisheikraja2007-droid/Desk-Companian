#pragma once

#include <Arduino.h>

#define PIN_WHEEL_L_FWD    1
#define PIN_WHEEL_L_REV    2
#define PIN_WHEEL_R_FWD    21
#define PIN_WHEEL_R_REV    47

#define PIN_ENC_L_A        38
#define PIN_ENC_L_B        39
#define PIN_ENC_R_A        40
#define PIN_ENC_R_B        41

#define MOTOR_DEADMAN_TIMEOUT_MS 200

class DrivePID {
public:
    DrivePID();
    void begin();
    void setTargetVelocities(float v_left_rad_s, float v_right_rad_s);
    void update(uint32_t now_ms);
    void emergencyBrake();
    void IRAM_ATTR emergencyBrakeFromISR();
    void getEncoderVelocities(float &v_l, float &v_r) const;

    static void IRAM_ATTR isrEncLA();
    static void IRAM_ATTR isrEncRA();

private:
    void _applyHBridge(uint8_t pin_fwd, uint8_t pin_rev, float pwr);

    float target_v_l;
    float target_v_r;
    float measured_v_l;
    float measured_v_r;
    float integral_l;
    float integral_r;
    float prev_err_l;
    float prev_err_r;

    uint32_t last_cmd_ms;
    uint32_t last_update_ms;

    volatile bool emergency_braked;
};

extern DrivePID driveController;

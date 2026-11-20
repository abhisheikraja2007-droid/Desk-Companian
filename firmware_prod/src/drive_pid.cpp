#include "drive_pid.h"

DrivePID driveController;

static volatile int32_t enc_l_ticks = 0;
static volatile int32_t enc_r_ticks = 0;

void IRAM_ATTR DrivePID::isrEncLA() {
    if (digitalRead(PIN_ENC_L_B) == HIGH) {
        enc_l_ticks++;
    } else {
        enc_l_ticks--;
    }
}

void IRAM_ATTR DrivePID::isrEncRA() {
    if (digitalRead(PIN_ENC_R_B) == HIGH) {
        enc_r_ticks++;
    } else {
        enc_r_ticks--;
    }
}

DrivePID::DrivePID()
    : target_v_l(0.0f), target_v_r(0.0f),
      measured_v_l(0.0f), measured_v_r(0.0f),
      integral_l(0.0f), integral_r(0.0f),
      prev_err_l(0.0f), prev_err_r(0.0f),
      last_cmd_ms(0), last_update_ms(0),
      emergency_braked(false) {}

void DrivePID::begin() {
    pinMode(PIN_WHEEL_L_FWD, OUTPUT);
    pinMode(PIN_WHEEL_L_REV, OUTPUT);
    pinMode(PIN_WHEEL_R_FWD, OUTPUT);
    pinMode(PIN_WHEEL_R_REV, OUTPUT);

    pinMode(PIN_ENC_L_A, INPUT_PULLUP);
    pinMode(PIN_ENC_L_B, INPUT_PULLUP);
    pinMode(PIN_ENC_R_A, INPUT_PULLUP);
    pinMode(PIN_ENC_R_B, INPUT_PULLUP);

    attachInterrupt(digitalPinToInterrupt(PIN_ENC_L_A), isrEncLA, RISING);
    attachInterrupt(digitalPinToInterrupt(PIN_ENC_R_A), isrEncRA, RISING);

    emergencyBrake();
}

void DrivePID::_applyHBridge(uint8_t pin_fwd, uint8_t pin_rev, float pwr) {
    pwr = constrain(pwr, -255.0f, 255.0f);
    if (pwr > 5.0f) {
        analogWrite(pin_fwd, (int)pwr);
        digitalWrite(pin_rev, LOW);
    } else if (pwr < -5.0f) {
        digitalWrite(pin_fwd, LOW);
        analogWrite(pin_rev, (int)(-pwr));
    } else {
        digitalWrite(pin_fwd, LOW);
        digitalWrite(pin_rev, LOW);
    }
}

void DrivePID::setTargetVelocities(float v_left_rad_s, float v_right_rad_s) {
    target_v_l = constrain(v_left_rad_s, -25.0f, 25.0f);
    target_v_r = constrain(v_right_rad_s, -25.0f, 25.0f);
    last_cmd_ms = millis();
    emergency_braked = false;
}

void DrivePID::update(uint32_t now_ms) {
    // 200 ms Dead-Man Watchdog
    if ((now_ms - last_cmd_ms) > MOTOR_DEADMAN_TIMEOUT_MS) {
        emergencyBrake();
        return;
    }

    if (emergency_braked) {
        return;
    }

    float dt = (now_ms - last_update_ms) / 1000.0f;
    if (dt < 0.010f) return;
    last_update_ms = now_ms;

    // Convert encoder counts to rad/s (approx 600 counts per wheel rev)
    float ticks_per_rev = 600.0f;
    measured_v_l = (enc_l_ticks / ticks_per_rev) * (2.0f * M_PI) / dt;
    measured_v_r = (enc_r_ticks / ticks_per_rev) * (2.0f * M_PI) / dt;
    enc_l_ticks = 0;
    enc_r_ticks = 0;

    // PID Gains
    const float kp = 8.5f;
    const float ki = 1.2f;
    const float kd = 0.4f;

    float err_l = target_v_l - measured_v_l;
    float err_r = target_v_r - measured_v_r;

    integral_l = constrain(integral_l + err_l * dt, -50.0f, 50.0f);
    integral_r = constrain(integral_r + err_r * dt, -50.0f, 50.0f);

    float d_l = (err_l - prev_err_l) / dt;
    float d_r = (err_r - prev_err_r) / dt;
    prev_err_l = err_l;
    prev_err_r = err_r;

    float out_l = (target_v_l * 10.0f) + (kp * err_l) + (ki * integral_l) + (kd * d_l);
    float out_r = (target_v_r * 10.0f) + (kp * err_r) + (ki * integral_r) + (kd * d_r);

    _applyHBridge(PIN_WHEEL_L_FWD, PIN_WHEEL_L_REV, out_l);
    _applyHBridge(PIN_WHEEL_R_FWD, PIN_WHEEL_R_REV, out_r);
}

void DrivePID::emergencyBrake() {
    emergency_braked = true;
    target_v_l = 0.0f;
    target_v_r = 0.0f;
    digitalWrite(PIN_WHEEL_L_FWD, HIGH);
    digitalWrite(PIN_WHEEL_L_REV, HIGH);
    digitalWrite(PIN_WHEEL_R_FWD, HIGH);
    digitalWrite(PIN_WHEEL_R_REV, HIGH);
}

void IRAM_ATTR DrivePID::emergencyBrakeFromISR() {
    emergency_braked = true;
    target_v_l = 0.0f;
    target_v_r = 0.0f;
    // Active electrical brake (both inputs high shorts motor terminals)
    digitalWrite(PIN_WHEEL_L_FWD, HIGH);
    digitalWrite(PIN_WHEEL_L_REV, HIGH);
    digitalWrite(PIN_WHEEL_R_FWD, HIGH);
    digitalWrite(PIN_WHEEL_R_REV, HIGH);
}

void DrivePID::getEncoderVelocities(float &v_l, float &v_r) const {
    v_l = measured_v_l;
    v_r = measured_v_r;
}

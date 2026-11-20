#include "scservo_bus.h"

SCServoBus servoBus;

static const uint8_t pwm_pin_map[TOTAL_SERVOS] = {
    PIN_PWM_NECK_YAW,
    PIN_PWM_NECK_PITCH,
    PIN_PWM_EAR_LEFT,
    PIN_PWM_EAR_RIGHT,
    PIN_PWM_ARM_LEFT,
    PIN_PWM_ARM_RIGHT,
    PIN_PWM_PINCER_L,
    PIN_PWM_PINCER_R
};

static const float default_neutral_deg[TOTAL_SERVOS] = {
    90.0f,  // Neck Yaw
    90.0f,  // Neck Pitch
    110.0f, // Left Ear
    70.0f,  // Right Ear
    90.0f,  // Left Arm
    90.0f,  // Right Arm
    90.0f,  // Left Pincer
    90.0f   // Right Pincer
};

SCServoBus::SCServoBus() : use_hardware_scservo(false) {
    for (int i = 0; i < TOTAL_SERVOS; i++) {
        current_deg[i] = default_neutral_deg[i];
        start_deg[i]   = default_neutral_deg[i];
        target_deg[i]  = default_neutral_deg[i];
        move_start_ms[i] = 0;
        move_duration_ms[i] = 200;
        is_moving[i] = false;
    }
}

void SCServoBus::begin() {
    Serial1.begin(SCSERVO_BAUDRATE, SERIAL_8N1, PIN_SCS_RX, PIN_SCS_TX);
    use_hardware_scservo = true;

    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    ESP32PWM::allocateTimer(2);
    ESP32PWM::allocateTimer(3);

    for (int i = 0; i < TOTAL_SERVOS; i++) {
        pwm_servos[i].setPeriodHertz(50);
        pwm_servos[i].attach(pwm_pin_map[i], 500, 2500);
        pwm_servos[i].write((int)current_deg[i]);
    }
}

float SCServoBus::_evaluateSCurve(float start, float target, float progress) {
    if (progress <= 0.0f) return start;
    if (progress >= 1.0f) return target;
    // Quintic S-Curve: 6t^5 - 15t^4 + 10t^3 (zero jerk, velocity and acceleration at ends)
    float s = progress * progress * progress * (progress * (progress * 6.0f - 15.0f) + 10.0f);
    return start + (target - start) * s;
}

void SCServoBus::_writeSCServoPos(uint8_t id, uint16_t position, uint16_t time_ms) {
    if (!use_hardware_scservo) return;
    uint8_t pos_h = (position >> 8) & 0xFF;
    uint8_t pos_l = position & 0xFF;
    uint8_t time_h = (time_ms >> 8) & 0xFF;
    uint8_t time_l = time_ms & 0xFF;

    uint8_t length = 7;
    uint8_t instruction = 0x03; // WRITE
    uint8_t address = 0x2A;    // Goal Position register

    uint8_t checksum = ~(id + length + instruction + address + pos_h + pos_l + time_h + time_l);

    uint8_t packet[11] = {
        0xFF, 0xFF,
        id,
        length,
        instruction,
        address,
        pos_h, pos_l,
        time_h, time_l,
        checksum
    };
    Serial1.write(packet, sizeof(packet));
}

void SCServoBus::setTargetDegrees(uint8_t index, float deg, uint32_t duration_ms) {
    if (index >= TOTAL_SERVOS) return;
    deg = constrain(deg, 0.0f, 180.0f);
    start_deg[index] = current_deg[index];
    target_deg[index] = deg;
    move_start_ms[index] = millis();
    move_duration_ms[index] = max((uint32_t)20, duration_ms);
    is_moving[index] = true;
}

void SCServoBus::setAllTargetsDegrees(const float deg[TOTAL_SERVOS], uint32_t duration_ms) {
    for (uint8_t i = 0; i < TOTAL_SERVOS; i++) {
        setTargetDegrees(i, deg[i], duration_ms);
    }
}

void SCServoBus::update(uint32_t now_ms) {
    for (int i = 0; i < TOTAL_SERVOS; i++) {
        if (!is_moving[i]) continue;
        uint32_t elapsed = now_ms - move_start_ms[i];
        float progress = (float)elapsed / (float)move_duration_ms[i];
        if (progress >= 1.0f) {
            current_deg[i] = target_deg[i];
            is_moving[i] = false;
        } else {
            current_deg[i] = _evaluateSCurve(start_deg[i], target_deg[i], progress);
        }

        pwm_servos[i].write((int)current_deg[i]);

        if (i < NUM_BUS_SERVOS) {
            uint16_t scs_pos = (uint16_t)map((long)current_deg[i], 0, 180, 0, 1023);
            _writeSCServoPos(i + 1, scs_pos, 20);
        }
    }
}

float SCServoBus::getCurrentDegrees(uint8_t index) const {
    if (index >= TOTAL_SERVOS) return 90.0f;
    return current_deg[index];
}

void SCServoBus::emergencyStop() {
    for (int i = 0; i < TOTAL_SERVOS; i++) {
        target_deg[i] = current_deg[i];
        is_moving[i] = false;
    }
}

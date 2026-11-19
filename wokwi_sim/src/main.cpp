#include <Arduino.h>
#include <Wire.h>
#include <ESP32Servo.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <ArduinoJson.h>

#define PIN_I2C_SDA        8
#define PIN_I2C_SCL        9

#define PIN_NECK_YAW       4
#define PIN_NECK_PITCH     5
#define PIN_EAR_LEFT       6
#define PIN_EAR_RIGHT      7
#define PIN_ARM_LEFT       15
#define PIN_ARM_RIGHT      16
#define PIN_PINCER_LEFT    17
#define PIN_PINCER_RIGHT   18

#define PIN_CLIFF_FL       10
#define PIN_CLIFF_FR       11
#define PIN_CLIFF_RL       12
#define PIN_CLIFF_RR       13

#define PIN_TOUCH          14

#define PIN_WHEEL_L_FWD    1
#define PIN_WHEEL_L_REV    2
#define PIN_WHEEL_R_FWD    21
#define PIN_WHEEL_R_REV    47

#define SCREEN_WIDTH       128
#define SCREEN_HEIGHT      64
#define OLED_RESET         -1
#define SCREEN_ADDRESS     0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
Adafruit_MPU6050 mpu;

Servo servos[8];
const uint8_t servo_pins[8] = {
    PIN_NECK_YAW, PIN_NECK_PITCH,
    PIN_EAR_LEFT, PIN_EAR_RIGHT,
    PIN_ARM_LEFT, PIN_ARM_RIGHT,
    PIN_PINCER_LEFT, PIN_PINCER_RIGHT
};

enum Emotion {
    EMOTION_NEUTRAL,
    EMOTION_HAPPY,
    EMOTION_CURIOUS,
    EMOTION_ALERT,
    EMOTION_SLEEPY
};

enum RobotState {
    STATE_IDLE_ROAM,
    STATE_CLIFF_RECOVERY,
    STATE_PETTED,
    STATE_PICKUP_FLAIL
};

volatile bool cliff_triggered = false;
volatile uint8_t cliff_sensor_id = 0;
volatile uint32_t cliff_trigger_time_us = 0;

float current_angles[8] = {90, 90, 110, 70, 90, 90, 90, 90};
float start_angles[8]   = {90, 90, 110, 70, 90, 90, 90, 90};
float target_angles[8]  = {90, 90, 110, 70, 90, 90, 90, 90};

uint32_t move_start_ms = 0;
uint32_t move_duration_ms = 400;
bool is_moving = false;

Emotion current_emotion = EMOTION_NEUTRAL;
Emotion last_drawn_emotion = EMOTION_ALERT;
RobotState current_state = STATE_IDLE_ROAM;

uint32_t recovery_start_ms = 0;
uint8_t recovery_phase = 0;
uint32_t petted_end_ms = 0;
uint32_t flail_end_ms = 0;
uint32_t last_telemetry_ms = 0;
bool roaming_enabled = true;

uint8_t prevTouch = HIGH;
uint32_t lastTouchDebounce = 0;

sensors_event_t a_event, g_event, temp_event;

void set_motors(int8_t left_pwr, int8_t right_pwr) {
    if (left_pwr > 0) {
        digitalWrite(PIN_WHEEL_L_FWD, HIGH);
        digitalWrite(PIN_WHEEL_L_REV, LOW);
    } else if (left_pwr < 0) {
        digitalWrite(PIN_WHEEL_L_FWD, LOW);
        digitalWrite(PIN_WHEEL_L_REV, HIGH);
    } else {
        digitalWrite(PIN_WHEEL_L_FWD, LOW);
        digitalWrite(PIN_WHEEL_L_REV, LOW);
    }

    if (right_pwr > 0) {
        digitalWrite(PIN_WHEEL_R_FWD, HIGH);
        digitalWrite(PIN_WHEEL_R_REV, LOW);
    } else if (right_pwr < 0) {
        digitalWrite(PIN_WHEEL_R_FWD, LOW);
        digitalWrite(PIN_WHEEL_R_REV, HIGH);
    } else {
        digitalWrite(PIN_WHEEL_R_FWD, LOW);
        digitalWrite(PIN_WHEEL_R_REV, LOW);
    }
}

void IRAM_ATTR isr_cliff_common(uint8_t sensor_num) {
    digitalWrite(PIN_WHEEL_L_FWD, LOW);
    digitalWrite(PIN_WHEEL_L_REV, LOW);
    digitalWrite(PIN_WHEEL_R_FWD, LOW);
    digitalWrite(PIN_WHEEL_R_REV, LOW);
    cliff_triggered = true;
    cliff_sensor_id = sensor_num;
    cliff_trigger_time_us = micros();
}

void IRAM_ATTR isr_cliff_fl() { isr_cliff_common(1); }
void IRAM_ATTR isr_cliff_fr() { isr_cliff_common(2); }
void IRAM_ATTR isr_cliff_rl() { isr_cliff_common(3); }
void IRAM_ATTR isr_cliff_rr() { isr_cliff_common(4); }

void set_target_pose(const float angles[8], uint32_t duration_ms) {
    for (int i = 0; i < 8; i++) {
        start_angles[i] = current_angles[i];
        target_angles[i] = angles[i];
    }
    move_start_ms = millis();
    move_duration_ms = duration_ms > 0 ? duration_ms : 10;
    is_moving = true;
}

void apply_pose_preset(const char* name, uint32_t duration_ms = 400) {
    if (strcmp(name, "neutral") == 0) {
        const float p[8] = {90, 90, 110, 70, 90, 90, 90, 90};
        set_target_pose(p, duration_ms);
        current_emotion = EMOTION_NEUTRAL;
    } else if (strcmp(name, "happy") == 0) {
        const float p[8] = {90, 85, 135, 45, 150, 90, 135, 90};
        set_target_pose(p, duration_ms);
        current_emotion = EMOTION_HAPPY;
    } else if (strcmp(name, "curious") == 0) {
        const float p[8] = {115, 105, 135, 75, 90, 90, 90, 90};
        set_target_pose(p, duration_ms);
        current_emotion = EMOTION_CURIOUS;
    } else if (strcmp(name, "alert") == 0) {
        const float p[8] = {90, 90, 155, 25, 100, 80, 90, 90};
        set_target_pose(p, duration_ms);
        current_emotion = EMOTION_ALERT;
    } else if (strcmp(name, "wave") == 0) {
        const float p[8] = {90, 85, 135, 45, 150, 90, 135, 90};
        set_target_pose(p, duration_ms);
        current_emotion = EMOTION_HAPPY;
    } else if (strcmp(name, "hold_pen") == 0) {
        const float p[8] = {90, 90, 110, 70, 135, 135, 120, 120};
        set_target_pose(p, duration_ms);
        current_emotion = EMOTION_CURIOUS;
    } else if (strcmp(name, "clap") == 0) {
        const float p[8] = {90, 90, 110, 70, 125, 55, 130, 130};
        set_target_pose(p, duration_ms);
        current_emotion = EMOTION_HAPPY;
    }
}

void update_motion_interpolator() {
    if (!is_moving) return;

    uint32_t elapsed = millis() - move_start_ms;
    float t = (float)elapsed / (float)move_duration_ms;

    if (t >= 1.0f) {
        t = 1.0f;
        is_moving = false;
    }

    float alpha = 0.5f * (1.0f - cosf(PI * t));

    for (int i = 0; i < 8; i++) {
        current_angles[i] = start_angles[i] + (target_angles[i] - start_angles[i]) * alpha;
        servos[i].write((int)roundf(current_angles[i]));
    }
}

void draw_eyes(Emotion em) {
    display.clearDisplay();

    const int lx = 34, ly = 32;
    const int rx = 94, ry = 32;

    switch (em) {
        case EMOTION_NEUTRAL:
            display.fillRoundRect(lx - 13, ly - 11, 26, 22, 8, SSD1306_WHITE);
            display.fillRoundRect(rx - 13, ry - 11, 26, 22, 8, SSD1306_WHITE);
            display.fillCircle(lx + 2, ly, 4, SSD1306_BLACK);
            display.fillCircle(rx + 2, ry, 4, SSD1306_BLACK);
            break;

        case EMOTION_HAPPY:
            display.fillRoundRect(lx - 14, ly - 10, 28, 20, 7, SSD1306_WHITE);
            display.fillRoundRect(rx - 14, ry - 10, 28, 20, 7, SSD1306_WHITE);
            display.fillCircle(lx, ly + 8, 11, SSD1306_BLACK);
            display.fillCircle(rx, ry + 8, 11, SSD1306_BLACK);
            break;

        case EMOTION_CURIOUS:
            display.fillRoundRect(lx - 15, ly - 14, 30, 28, 9, SSD1306_WHITE);
            display.fillRoundRect(rx - 11, ry - 9, 22, 18, 6, SSD1306_WHITE);
            display.fillCircle(lx - 2, ly - 3, 5, SSD1306_BLACK);
            display.fillCircle(rx - 1, ry - 2, 3, SSD1306_BLACK);
            break;

        case EMOTION_ALERT:
            display.drawRoundRect(lx - 15, ly - 15, 30, 30, 10, SSD1306_WHITE);
            display.drawRoundRect(lx - 14, ly - 14, 28, 28, 9, SSD1306_WHITE);
            display.fillCircle(lx, ly, 7, SSD1306_WHITE);
            display.drawRoundRect(rx - 15, ry - 15, 30, 30, 10, SSD1306_WHITE);
            display.drawRoundRect(rx - 14, ry - 14, 28, 28, 9, SSD1306_WHITE);
            display.fillCircle(rx, ry, 7, SSD1306_WHITE);
            break;

        case EMOTION_SLEEPY:
            display.fillRoundRect(lx - 14, ly - 4, 28, 10, 4, SSD1306_WHITE);
            display.fillRoundRect(rx - 14, ry - 4, 28, 10, 4, SSD1306_WHITE);
            break;
    }

    display.display();
    last_drawn_emotion = em;
}

void parse_serial_commands() {
    while (Serial.available()) {
        String line = Serial.readStringUntil('\n');
        line.trim();
        if (line.length() == 0) continue;

        JsonDocument doc;
        DeserializationError err = deserializeJson(doc, line);
        if (err) continue;

        if (doc["emotion"].is<const char*>()) {
            const char* em_str = doc["emotion"];
            if (strcmp(em_str, "happy") == 0) current_emotion = EMOTION_HAPPY;
            else if (strcmp(em_str, "curious") == 0) current_emotion = EMOTION_CURIOUS;
            else if (strcmp(em_str, "alert") == 0) current_emotion = EMOTION_ALERT;
            else if (strcmp(em_str, "sleepy") == 0) current_emotion = EMOTION_SLEEPY;
            else current_emotion = EMOTION_NEUTRAL;
        }

        if (doc["gesture"].is<const char*>()) {
            const char* g_str = doc["gesture"];
            apply_pose_preset(g_str, 350);
        }

        if (doc["roam"].is<int>()) {
            roaming_enabled = (doc["roam"] == 1);
        }

        if (doc["joints"].is<JsonArray>()) {
            JsonArray arr = doc["joints"];
            if (arr.size() == 8) {
                float target[8];
                for (int i = 0; i < 8; i++) target[i] = arr[i];
                set_target_pose(target, 250);
            }
        }
    }
}

void stream_telemetry() {
    if (millis() - last_telemetry_ms < 50) return;
    last_telemetry_ms = millis();

    JsonDocument doc;
    doc["t"] = millis();

    const char* st_names[] = {"roam", "cliff_recovery", "petted", "flail"};
    doc["state"] = st_names[current_state];

    JsonArray wheels_arr = doc["wheels"].to<JsonArray>();
    wheels_arr.add(digitalRead(PIN_WHEEL_L_FWD));
    wheels_arr.add(digitalRead(PIN_WHEEL_L_REV));
    wheels_arr.add(digitalRead(PIN_WHEEL_R_FWD));
    wheels_arr.add(digitalRead(PIN_WHEEL_R_REV));

    JsonArray joints_arr = doc["joints"].to<JsonArray>();
    for (int i = 0; i < 8; i++) {
        joints_arr.add((int)roundf(current_angles[i]));
    }

    JsonArray cliff_arr = doc["cliff"].to<JsonArray>();
    cliff_arr.add(digitalRead(PIN_CLIFF_FL) == LOW ? 1 : 0);
    cliff_arr.add(digitalRead(PIN_CLIFF_FR) == LOW ? 1 : 0);
    cliff_arr.add(digitalRead(PIN_CLIFF_RL) == LOW ? 1 : 0);
    cliff_arr.add(digitalRead(PIN_CLIFF_RR) == LOW ? 1 : 0);

    JsonObject imu_obj = doc["imu"].to<JsonObject>();
    imu_obj["ax"] = roundf(a_event.acceleration.x * 100.0f) / 100.0f;
    imu_obj["ay"] = roundf(a_event.acceleration.y * 100.0f) / 100.0f;
    imu_obj["az"] = roundf(a_event.acceleration.z * 100.0f) / 100.0f;

    const char* em_names[] = {"neutral", "happy", "curious", "alert", "sleepy"};
    doc["emotion"] = em_names[current_emotion];

    serializeJson(doc, Serial);
    Serial.println();
}

void setup() {
    Serial.begin(115200);

    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);

    display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS);
    draw_eyes(EMOTION_NEUTRAL);

    mpu.begin();
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    pinMode(PIN_TOUCH, INPUT_PULLUP);

    pinMode(PIN_CLIFF_FL, INPUT_PULLUP);
    pinMode(PIN_CLIFF_FR, INPUT_PULLUP);
    pinMode(PIN_CLIFF_RL, INPUT_PULLUP);
    pinMode(PIN_CLIFF_RR, INPUT_PULLUP);

    attachInterrupt(digitalPinToInterrupt(PIN_CLIFF_FL), isr_cliff_fl, FALLING);
    attachInterrupt(digitalPinToInterrupt(PIN_CLIFF_FR), isr_cliff_fr, FALLING);
    attachInterrupt(digitalPinToInterrupt(PIN_CLIFF_RL), isr_cliff_rl, FALLING);
    attachInterrupt(digitalPinToInterrupt(PIN_CLIFF_RR), isr_cliff_rr, FALLING);

    pinMode(PIN_WHEEL_L_FWD, OUTPUT);
    pinMode(PIN_WHEEL_L_REV, OUTPUT);
    pinMode(PIN_WHEEL_R_FWD, OUTPUT);
    pinMode(PIN_WHEEL_R_REV, OUTPUT);
    set_motors(1, 1);

    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    ESP32PWM::allocateTimer(2);
    ESP32PWM::allocateTimer(3);

    for (int i = 0; i < 8; i++) {
        servos[i].setPeriodHertz(50);
        servos[i].attach(servo_pins[i], 500, 2400);
        servos[i].write((int)roundf(current_angles[i]));
    }

    apply_pose_preset("neutral", 100);
    Serial.println("Desk Companion Robot - Edge Firmware Online");
    Serial.println("DESK_PET_READY");
}

void loop() {
    mpu.getEvent(&a_event, &g_event, &temp_event);

    uint8_t curTouch = digitalRead(PIN_TOUCH);
    if (curTouch != prevTouch && (millis() - lastTouchDebounce) > 40) {
        if (prevTouch == HIGH && curTouch == LOW) {
            Serial.println("[TOUCH] Crown petted!");
            current_state = STATE_PETTED;
            current_emotion = EMOTION_HAPPY;
            petted_end_ms = millis() + 1500;
            const float touch_pose[8] = {90.0f, 85.0f, 135.0f, 45.0f, 150.0f, 90.0f, 135.0f, 90.0f};
            set_target_pose(touch_pose, 300);
            set_motors(0, 0);
        }
        prevTouch = curTouch;
        lastTouchDebounce = millis();
    }

    if (cliff_triggered) {
        cliff_triggered = false;
        current_state = STATE_CLIFF_RECOVERY;
        recovery_phase = 1;
        recovery_start_ms = millis();
        current_emotion = EMOTION_ALERT;
        apply_pose_preset("alert", 150);
        set_motors(-1, -1);
        Serial.printf("[CLIFF] Triggered sensor %d! Reversing...\n", cliff_sensor_id);
    }

    if (current_state == STATE_CLIFF_RECOVERY) {
        uint32_t rec_elapsed = millis() - recovery_start_ms;
        if (recovery_phase == 1) {
            set_motors(-1, -1);
            if (rec_elapsed > 450) {
                recovery_phase = 2;
                recovery_start_ms = millis();
            }
        } else if (recovery_phase == 2) {
            if (cliff_sensor_id == 1 || cliff_sensor_id == 3) {
                set_motors(1, -1);
            } else {
                set_motors(-1, 1);
            }
            if (rec_elapsed > 350) {
                current_state = STATE_IDLE_ROAM;
                current_emotion = EMOTION_CURIOUS;
                apply_pose_preset("curious", 300);
                if (roaming_enabled) {
                    set_motors(1, 1);
                } else {
                    set_motors(0, 0);
                }
            }
        }
    }
    else if (current_state == STATE_PETTED) {
        set_motors(0, 0);
        if (millis() > petted_end_ms) {
            current_state = STATE_IDLE_ROAM;
            current_emotion = EMOTION_NEUTRAL;
            apply_pose_preset("neutral", 400);
            if (roaming_enabled) {
                set_motors(1, 1);
            }
        }
    }
    else if (current_state == STATE_PICKUP_FLAIL) {
        set_motors(0, 0);
        if (millis() > flail_end_ms) {
            current_state = STATE_IDLE_ROAM;
            current_emotion = EMOTION_ALERT;
            apply_pose_preset("neutral", 300);
            if (roaming_enabled) {
                set_motors(1, 1);
            }
        }
    }
    else if (current_state == STATE_IDLE_ROAM) {
        float az_diff = fabsf(a_event.acceleration.z - 9.81f);
        float ax_val = fabsf(a_event.acceleration.x);
        float ay_val = fabsf(a_event.acceleration.y);
        if (az_diff > 4.5f || ax_val > 5.5f || ay_val > 5.5f) {
            Serial.println("[IMU] Pickup reflex triggered!");
            current_state = STATE_PICKUP_FLAIL;
            current_emotion = EMOTION_ALERT;
            flail_end_ms = millis() + 1500;
            apply_pose_preset("alert", 150);
            set_motors(0, 0);
        }
        else if (roaming_enabled) {
            set_motors(1, 1);
        } else {
            set_motors(0, 0);
        }
    }

    update_motion_interpolator();

    if (current_emotion != last_drawn_emotion) {
        draw_eyes(current_emotion);
    }

    parse_serial_commands();
    stream_telemetry();

    delay(5);
}

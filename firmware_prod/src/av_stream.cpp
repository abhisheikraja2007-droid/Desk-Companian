#include "av_stream.h"
#include "scservo_bus.h"
#include "drive_pid.h"
#include "safety_reflex.h"

AVStreamClient avStream;

static void webSocketEventWrapper(WStype_t type, uint8_t * payload, size_t length) {
    if (type == WStype_CONNECTED) {
        Serial.println("[WS] Connected to RTX 5050 Host Server!");
    } else if (type == WStype_DISCONNECTED) {
        Serial.println("[WS] Disconnected from Host Server.");
        driveController.emergencyBrake();
    } else if (type == WStype_TEXT) {
        JsonDocument doc;
        DeserializationError err = deserializeJson(doc, payload, length);
        if (err) return;

        // Parse target 8-DOF servo angles
        if (doc["targets"].is<JsonArray>()) {
            JsonArray arr = doc["targets"].as<JsonArray>();
            float angles[TOTAL_SERVOS];
            for (size_t i = 0; i < TOTAL_SERVOS && i < arr.size(); i++) {
                angles[i] = arr[i].as<float>();
            }
            uint32_t dur = doc["duration_ms"] | 250;
            servoBus.setAllTargetsDegrees(angles, dur);
        }

        // Parse differential wheel velocities (rad/s)
        if (doc["vel_l"].is<float>() && doc["vel_r"].is<float>()) {
            float vl = doc["vel_l"].as<float>();
            float vr = doc["vel_r"].as<float>();
            driveController.setTargetVelocities(vl, vr);
        }

        // Parse eye expression
        const char* expr_str = doc["expression"] | "neutral";
        if (strcmp(expr_str, "happy") == 0) {
            avStream.setEyeExpression(EYE_HAPPY);
        } else if (strcmp(expr_str, "alert") == 0) {
            avStream.setEyeExpression(EYE_ALERT);
        } else if (strcmp(expr_str, "focused") == 0) {
            avStream.setEyeExpression(EYE_FOCUSED);
        } else if (strcmp(expr_str, "sleepy") == 0) {
            avStream.setEyeExpression(EYE_SLEEPY);
        } else {
            avStream.setEyeExpression(EYE_NEUTRAL);
        }
    }
}

AVStreamClient::AVStreamClient()
    : display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1),
      current_expr(EYE_NEUTRAL),
      last_blink_ms(0),
      connected(false) {}

void AVStreamClient::begin(const char* ssid, const char* pass, const char* host_ip, uint16_t host_port) {
    _initDisplay();

    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, pass);
    Serial.printf("[WiFi] Connecting to %s...\n", ssid);

    webSocket.begin(host_ip, host_port, "/robot");
    webSocket.onEvent(webSocketEventWrapper);
    webSocket.setReconnectInterval(2000);
}

void AVStreamClient::_initDisplay() {
    if (display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR_L)) {
        display.clearDisplay();
        _drawEyes(EYE_NEUTRAL);
        display.display();
    }
}

void AVStreamClient::_drawEyes(EyeExpression expr) {
    display.clearDisplay();

    int eye_w = 34;
    int eye_h = 44;
    int eye_r = 14;

    int left_x = 20;
    int right_x = 74;
    int eye_y = 10;

    if (expr == EYE_HAPPY) {
        // Upward arc / smiling crescent
        display.fillRoundRect(left_x, eye_y, eye_w, eye_h, eye_r, SSD1306_WHITE);
        display.fillRoundRect(right_x, eye_y, eye_w, eye_h, eye_r, SSD1306_WHITE);
        display.fillCircle(left_x + eye_w/2, eye_y + eye_h + 4, eye_w/2 + 2, SSD1306_BLACK);
        display.fillCircle(right_x + eye_w/2, eye_y + eye_h + 4, eye_w/2 + 2, SSD1306_BLACK);
    } else if (expr == EYE_ALERT) {
        // Wide circular alert eyes with small pupils
        display.fillCircle(left_x + eye_w/2, eye_y + eye_h/2, 22, SSD1306_WHITE);
        display.fillCircle(right_x + eye_w/2, eye_y + eye_h/2, 22, SSD1306_WHITE);
        display.fillCircle(left_x + eye_w/2, eye_y + eye_h/2, 6, SSD1306_BLACK);
        display.fillCircle(right_x + eye_w/2, eye_y + eye_h/2, 6, SSD1306_BLACK);
    } else if (expr == EYE_FOCUSED) {
        // Narrow focused eyes
        display.fillRoundRect(left_x, eye_y + 12, eye_w, 20, 6, SSD1306_WHITE);
        display.fillRoundRect(right_x, eye_y + 12, eye_w, 20, 6, SSD1306_WHITE);
    } else if (expr == EYE_SLEEPY) {
        // Drooping eyelids
        display.fillRoundRect(left_x, eye_y + 18, eye_w, 14, 4, SSD1306_WHITE);
        display.fillRoundRect(right_x, eye_y + 18, eye_w, 14, 4, SSD1306_WHITE);
    } else if (expr == EYE_BLINK) {
        // Horizontal slits
        display.fillRoundRect(left_x, eye_y + eye_h/2, eye_w, 4, 2, SSD1306_WHITE);
        display.fillRoundRect(right_x, eye_y + eye_h/2, eye_w, 4, 2, SSD1306_WHITE);
    } else {
        // Neutral rounded squircle eyes
        display.fillRoundRect(left_x, eye_y, eye_w, eye_h, eye_r, SSD1306_WHITE);
        display.fillRoundRect(right_x, eye_y, eye_w, eye_h, eye_r, SSD1306_WHITE);
        display.fillCircle(left_x + 12, eye_y + 14, 4, SSD1306_BLACK);
        display.fillCircle(right_x + 12, eye_y + 14, 4, SSD1306_BLACK);
    }

    display.display();
}

void AVStreamClient::setEyeExpression(EyeExpression expr) {
    if (current_expr != expr) {
        current_expr = expr;
        _drawEyes(current_expr);
    }
}

void AVStreamClient::sendTelemetry(const JsonDocument& doc) {
    String json_str;
    serializeJson(doc, json_str);
    webSocket.sendTXT(json_str);
}

bool AVStreamClient::isConnected() const {
    return WiFi.status() == WL_CONNECTED;
}

void AVStreamClient::update() {
    webSocket.loop();

    uint32_t now = millis();
    if (now - last_blink_ms > 4000) {
        last_blink_ms = now;
        _drawEyes(EYE_BLINK);
        delay(80);
        _drawEyes(current_expr);
    }
}

#pragma once

#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64
#define OLED_ADDR_L   0x3C
#define OLED_ADDR_R   0x3D

#define PIN_I2S_MIC_SCK  42
#define PIN_I2S_MIC_WS   41
#define PIN_I2S_MIC_SD   2

#define PIN_I2S_SPK_BCLK 26
#define PIN_I2S_SPK_LRC  27
#define PIN_I2S_SPK_DIN  25

enum EyeExpression {
    EYE_NEUTRAL,
    EYE_HAPPY,
    EYE_ALERT,
    EYE_FOCUSED,
    EYE_SLEEPY,
    EYE_BLINK
};

class AVStreamClient {
public:
    AVStreamClient();
    void begin(const char* ssid, const char* pass, const char* host_ip, uint16_t host_port);
    void update();
    void setEyeExpression(EyeExpression expr);
    void sendTelemetry(const JsonDocument& doc);
    bool isConnected() const;

private:
    void _initDisplay();
    void _drawEyes(EyeExpression expr);
    void _onWebSocketEvent(WStype_t type, uint8_t * payload, size_t length);

    Adafruit_SSD1306 display;
    WebSocketsClient webSocket;
    EyeExpression current_expr;
    uint32_t last_blink_ms;
    bool connected;
};

extern AVStreamClient avStream;

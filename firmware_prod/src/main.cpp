#include <Arduino.h>
#include "scservo_bus.h"
#include "safety_reflex.h"
#include "drive_pid.h"
#include "av_stream.h"

static const char* WIFI_SSID = "DeskCompanionNet";
static const char* WIFI_PASS = "companion123";
static const char* HOST_IP   = "192.168.1.100";
static const uint16_t HOST_PORT = 8765;

TaskHandle_t TaskHandleSafety;
TaskHandle_t TaskHandleComms;

void TaskSafetyAndControl(void *pvParameters) {
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(20); // 50 Hz loop (20 ms)

    for (;;) {
        uint32_t now_ms = millis();

        // 1. Hardware safety reflex & sensor update
        safetyReflex.update();

        // 2. If cliff is triggered, execute immediate reverse recovery
        if (safetyReflex.isCliffTriggered()) {
            driveController.emergencyBrake();
            servoBus.setTargetDegrees(1, 45.0f, 150);  // Neck pitch back
            servoBus.setTargetDegrees(2, 45.0f, 150);  // Ears back
            servoBus.setTargetDegrees(3, 45.0f, 150);
            vTaskDelay(pdMS_TO_TICKS(100));
            driveController.setTargetVelocities(-6.0f, -6.0f);
            vTaskDelay(pdMS_TO_TICKS(250));
            driveController.emergencyBrake();
            safetyReflex.resetCliffStatus();
        } else {
            // 3. Normal 50 Hz drive velocity PID & S-curve servos
            driveController.update(now_ms);
            servoBus.update(now_ms);
        }

        vTaskDelayUntil(&xLastWakeTime, xFrequency);
    }
}

void TaskCommsAndAV(void *pvParameters) {
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(50); // 20 Hz telemetry streaming

    for (;;) {
        avStream.update();

        // Send telemetry packet over WebSocket to RTX 5050 host
        SafetyTelemetry telem = safetyReflex.getTelemetry();
        float vl, vr;
        driveController.getEncoderVelocities(vl, vr);

        JsonDocument doc;
        doc["type"] = "telemetry";
        doc["timestamp_ms"] = millis();

        JsonArray cliff = doc["cliff"].to<JsonArray>();
        cliff.add(telem.cliff_fl);
        cliff.add(telem.cliff_fr);
        cliff.add(telem.cliff_rl);
        cliff.add(telem.cliff_rr);

        doc["touch"] = telem.crown_touched;
        doc["tilted"] = telem.is_tilted;
        doc["picked_up"] = telem.is_picked_up;

        JsonArray imu = doc["imu"].to<JsonArray>();
        imu.add(telem.accel_x);
        imu.add(telem.accel_y);
        imu.add(telem.accel_z);
        imu.add(telem.gyro_x);
        imu.add(telem.gyro_y);
        imu.add(telem.gyro_z);

        JsonArray joints = doc["joints"].to<JsonArray>();
        for (int i = 0; i < TOTAL_SERVOS; i++) {
            joints.add(servoBus.getCurrentDegrees(i));
        }

        doc["measured_vl"] = vl;
        doc["measured_vr"] = vr;

        avStream.sendTelemetry(doc);

        vTaskDelayUntil(&xLastWakeTime, xFrequency);
    }
}

void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n[BOOT] Initializing ESP32-S3-N16R8 Production Firmware...");

    servoBus.begin();
    safetyReflex.begin();
    driveController.begin();
    avStream.begin(WIFI_SSID, WIFI_PASS, HOST_IP, HOST_PORT);

    // Create FreeRTOS Tasks pinned to separate cores
    xTaskCreatePinnedToCore(
        TaskSafetyAndControl,
        "SafetyCtrl",
        4096,
        NULL,
        5, // High priority
        &TaskHandleSafety,
        1  // Core 1 (Dedicated Real-time Control)
    );

    xTaskCreatePinnedToCore(
        TaskCommsAndAV,
        "CommsAV",
        8192,
        NULL,
        2, // Background priority
        &TaskHandleComms,
        0  // Core 0 (Network & I/O)
    );

    Serial.println("[BOOT] Stage 4 Production Firmware Started on Dual Cores!");
}

void loop() {
    vTaskDelay(pdMS_TO_TICKS(1000));
}

/*
 * esp32_ld2450_bt.ino
 *
 * Reads LD2450 24 GHz radar frames over UART and re-broadcasts each frame as
 * a CSV line over Classic Bluetooth SPP ("ESP32-Radar").
 *
 * Wiring:
 *   LD2450 TX  ->  ESP32 GPIO16  (Serial2 RX)
 *   LD2450 RX  ->  ESP32 GPIO17  (Serial2 TX)
 *   LD2450 VCC ->  ESP32 5V
 *   LD2450 GND ->  ESP32 GND
 *
 * Output format (one line per LD2450 frame, ~10 Hz):
 *   x1,y1,s1,x2,y2,s2,x3,y3,s3\n
 *   x/y in mm (signed int), speed in cm/s (signed int).
 *   Absent targets report 0,0,0.
 *
 * Arduino IDE setup:
 *   Board: "ESP32 Dev Module" (esp32 by Espressif >= 2.0)
 *   Partition scheme: Default 4MB with spiffs
 *   Upload speed: 921600
 *
 * Laptop pairing (Linux):
 *   bluetoothctl
 *   > scan on          # find "ESP32-Radar" and copy its MAC
 *   > pair <MAC>
 *   > trust <MAC>
 *   > quit
 *   sudo rfcomm bind 0 <MAC>   # creates /dev/rfcomm0
 */

#include <Arduino.h>
#include <BluetoothSerial.h>

// ── Configuration ────────────────────────────────────────────────────────────
static const int     RADAR_RX_PIN  = 16;
static const int     RADAR_TX_PIN  = 17;
static const int     RADAR_BAUD    = 256000;
static const char*   BT_NAME       = "ESP32-Radar";

// ── LD2450 protocol constants ─────────────────────────────────────────────────
static const uint8_t HEADER[4] = {0xAA, 0xFF, 0x03, 0x00};
static const uint8_t TAIL[2]   = {0x55, 0xCC};
static const int     FRAME_LEN = 30;  // 4 header + 3*8 targets + 2 tail

// ── Globals ───────────────────────────────────────────────────────────────────
BluetoothSerial BT;
static uint8_t  buf[FRAME_LEN];
static int      buf_pos = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────

/*
 * Read a little-endian signed 16-bit value from two bytes.
 */
static inline int16_t read_i16(const uint8_t* p) {
    return (int16_t)(p[0] | ((uint16_t)p[1] << 8));
}

/*
 * Parse buf[] and emit a CSV line over BT.
 * Target layout per 8-byte block: [x_lo, x_hi, y_lo, y_hi, s_lo, s_hi, _, _]
 */
static void emit_frame() {
    int16_t vals[9];
    for (int i = 0; i < 3; i++) {
        const uint8_t* t = buf + 4 + i * 8;
        vals[i * 3 + 0] = read_i16(t);       // x_mm
        vals[i * 3 + 1] = read_i16(t + 2);   // y_mm
        vals[i * 3 + 2] = read_i16(t + 4);   // speed_cm_s
        // t[6..7] = distance resolution — ignored
    }

    char line[64];
    snprintf(line, sizeof(line),
        "%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
        vals[0], vals[1], vals[2],
        vals[3], vals[4], vals[5],
        vals[6], vals[7], vals[8]);

    if (BT.hasClient()) {
        BT.print(line);
    }
}

// ── Arduino entry points ──────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);   // USB debug
    Serial2.begin(RADAR_BAUD, SERIAL_8N1, RADAR_RX_PIN, RADAR_TX_PIN);
    BT.begin(BT_NAME);
    Serial.println("ESP32-Radar ready — waiting for BT client");
}

void loop() {
    while (Serial2.available()) {
        uint8_t b = (uint8_t)Serial2.read();

        // While still matching the header, enforce each byte position.
        if (buf_pos < 4) {
            if (b == HEADER[buf_pos]) {
                buf[buf_pos++] = b;
            } else {
                // Mismatch — reset. If this byte is the start of a new header, keep it.
                buf_pos = (b == HEADER[0]) ? 1 : 0;
                if (buf_pos == 1) buf[0] = b;
            }
            continue;
        }

        // Header confirmed; accumulate payload + tail bytes.
        buf[buf_pos++] = b;

        if (buf_pos == FRAME_LEN) {
            if (buf[FRAME_LEN - 2] == TAIL[0] && buf[FRAME_LEN - 1] == TAIL[1]) {
                emit_frame();
            }
            // Discard frame (valid or not) and start fresh.
            buf_pos = 0;
        }
    }
}

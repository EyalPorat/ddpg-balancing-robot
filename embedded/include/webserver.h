#ifndef WEBSERVER_H
#define WEBSERVER_H

#include <WiFi.h>
#include <AsyncTCP.h>
#define HTTP_GET     0b00000001
#define HTTP_POST    0b00000010
#define HTTP_DELETE  0b00000100
#define HTTP_PUT     0b00001000
#define HTTP_PATCH   0b00010000
#define HTTP_HEAD    0b00100000
#define HTTP_OPTIONS 0b01000000
#define HTTP_ANY     0b01111111
#include <ESPAsyncWebServer.h>
#include <SPIFFS.h>
#include <vector>

// Operation modes
#define MODE_DDPG 0
#define MODE_PID 1

// Forward declarations of external variables
extern float max_angular_velocity;
extern float max_linear_velocity;
extern float wheel_radius;
extern float vBatt;
extern byte demoMode;
extern float gyroYdata;
extern float gyroYoffset;
extern float varAng;
extern float varOmg;
extern float powerL;
extern float powerR;
extern int16_t maxPwr;

class TelemetryWebServer {
public:
    TelemetryWebServer(uint16_t port = 80);
    void begin(const char* ssid, const char* password);

private:
    AsyncWebServer server;
    void setupRoutes();
    static String getIndexHTML();
    String getTelemetryJSON();
};

#endif // WEBSERVER_H

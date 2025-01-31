#pragma once
#include <WiFi.h>
#include <WiFiUdp.h>

// struct __attribute__((packed)) LogMessage {
//     float acc_x_g;
//     float acc_z_g;
//     float gyro_x_degps;
//     float ori_x_deg;
//     float dist;
//     float spd;
//     float dt_ms;
//     float mech_factor;
//     int8_t output_pwm;
//     // bool output_strong_fwd;
//     // bool output_weak_fwd;
//     // bool output_none;
//     // bool output_weak_bwd;
//     // bool output_strong_bwd;
//     bool standing;
// };

struct __attribute__((packed)) LogMessage {
    // Timing
    uint32_t timestamp;      // milliseconds since boot
    float dt;               // control loop period in seconds
    
    // Controller State
    float theta;           // body angle (rad)
    float theta_dot;       // angular velocity (rad/s)
    float x;              // horizontal position (m)
    float x_dot;          // horizontal velocity (m/s)
    float phi;            // wheel angle (rad)
    float phi_dot;        // wheel angular velocity (rad/s)
    
    // Control Outputs
    float model_output;    // raw model output
    int8_t motor_pwm;     // applied PWM value
    
    // System Status
    bool standing;         // whether robot is in standing mode
    bool model_active;     // whether DDPG model is active
    float battery_voltage; // battery voltage
    
    // Additional Metrics
    float acc_x;          // X acceleration (g)
    float acc_z;          // Z acceleration (g)
    float gyro_x;         // X gyro rate (deg/s)
};

class UdpLogger {
public:
    UdpLogger() : connected(false), targetPort(0) {
        lastError[0] = '\0';
    }
    
    bool begin(const char* ssid, 
              const char* password,
              uint16_t udpPort = 44444);
    
    // Send a log message
    bool sendLogMessage(const LogMessage& msg);
    
    // Check if connected to WiFi
    bool isConnected() const;
    
    // Get last error message
    const char* getLastError() const;

private:
    WiFiUDP udp;
    char lastError[64];
    bool connected;
    IPAddress targetAddress;
    uint16_t targetPort;
    
    void setError(const char* error);
    IPAddress calculateBroadcastAddress();
};

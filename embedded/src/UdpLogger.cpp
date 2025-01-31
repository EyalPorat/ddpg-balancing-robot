#include <UdpLogger.h>
#include <string.h>

bool UdpLogger::begin(const char* ssid, const char* password, uint16_t udpPort) {
    Serial.println("UdpLogger::begin starting...");
    
    if (!ssid || !password) {
        setError("Null parameter provided");
        Serial.println("Error: Null parameter");
        return false;
    }
    
    Serial.printf("Connecting to WiFi SSID: %s\n", ssid);
    WiFi.begin(ssid, password);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    Serial.println();
    
    if (WiFi.status() != WL_CONNECTED) {
        setError("Failed to connect to WiFi");
        Serial.println("Error: WiFi connection failed");
        connected = false;
        return false;
    }
    
    Serial.printf("WiFi connected. IP: %s\n", WiFi.localIP().toString().c_str());
    
    // Calculate broadcast address
    targetAddress = calculateBroadcastAddress();
    Serial.printf("Using broadcast address: %s\n", targetAddress.toString().c_str());
    
    targetPort = udpPort;
    Serial.printf("Initializing UDP on port %d\n", udpPort);
    
    if (!udp.begin(udpPort)) {
        setError("Failed to initialize UDP");
        Serial.println("Error: UDP initialization failed");
        return false;
    }
    
    connected = true;
    Serial.println("UdpLogger initialization successful");
    return true;
}

IPAddress UdpLogger::calculateBroadcastAddress() {
    IPAddress ip = WiFi.localIP();
    IPAddress subnet = WiFi.subnetMask();
    
    // Calculate broadcast address by OR-ing the inverted subnet mask with IP
    IPAddress broadcast(
        ip[0] | ~subnet[0],
        ip[1] | ~subnet[1],
        ip[2] | ~subnet[2],
        ip[3] | ~subnet[3]
    );
    
    return broadcast;
}

bool UdpLogger::sendLogMessage(const LogMessage& msg) {
    if (!connected) {
        setError("Not connected to WiFi");
        return false;
    }
    
    if (!udp.beginPacket(targetAddress, targetPort)) {  // Use IPAddress object directly
        setError("Failed to begin UDP packet");
        return false;
    }
    
    size_t written = udp.write((uint8_t*)&msg, sizeof(LogMessage));
    if (written != sizeof(LogMessage)) {
        setError("Failed to write complete message");
        return false;
    }
    
    if (!udp.endPacket()) {
        setError("Failed to send UDP packet");
        return false;
    }
    
    return true;
}

bool UdpLogger::isConnected() const {
    return connected && (WiFi.status() == WL_CONNECTED);
}

const char* UdpLogger::getLastError() const {
    return lastError;
}

void UdpLogger::setError(const char* error) {
    strncpy(lastError, error, sizeof(lastError) - 1);
    lastError[sizeof(lastError) - 1] = '\0';
}

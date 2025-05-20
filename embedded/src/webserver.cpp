#include <webserver.h>
#include <time.h>

TelemetryWebServer::TelemetryWebServer(uint16_t port) : server(port) {}

void TelemetryWebServer::begin(const char* ssid, const char* password) {
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.println("Connecting to WiFi...");
    }
    Serial.println(WiFi.localIP());

    setupRoutes();
    server.begin();
}

void TelemetryWebServer::setupRoutes() {
    // Serve main page
    server.on("/", HTTP_GET, [](AsyncWebServerRequest *request) {
        request->send(200, "text/html", getIndexHTML());
    });

    // Start mode
    server.on("/mode", HTTP_GET, [](AsyncWebServerRequest *request) {
        if (request->hasParam("mode")) {
            String mode = request->getParam("mode")->value();
            if (mode == "pid") {
                demoMode = MODE_PID;
                // Stop motors when switching to balance
                powerL = powerR = 0;
                request->send(200, "text/plain", "Switching to pid balance mode");
            }
            else if (mode == "ddpg") {
                demoMode = MODE_DDPG;
                // Stop motors when switching to balance
                powerL = powerR = 0;
                request->send(200, "text/plain", "Switching to DDPG balance mode");
            }
            else {
                request->send(400, "text/plain", "Invalid mode");
            }
        } else {
            request->send(400, "text/plain", "Missing mode parameter");
        }
    });

    // Get telemetry data
    server.on("/telemetry", HTTP_GET, [this](AsyncWebServerRequest *request) {
        String json = getTelemetryJSON();
        request->send(200, "application/json", json);
    });
}

String TelemetryWebServer::getTelemetryJSON() {
    String json = "{\"telemetry\":{";
    
    // Basic telemetry with error checking
    json += "\"angle\":" + String(varAng, 2) + ",";  // Limit decimal places
    json += "\"angular_velocity\":" + String(varOmg, 2) + ",";
    json += "\"global_angle\":" + String(lastComplementaryAngleDDPG, 2) + ",";
    json += "\"battery_voltage\":" + String(vBatt, 2) + ",";
    json += "\"mode\":\"" + String(demoMode == MODE_PID ? "pid" : "ddpg") + "\"";
    
    json += "}}";
    return json;
}

String TelemetryWebServer::getIndexHTML() {
    return R"rawliteral(
<!DOCTYPE HTML>
<html>
<head>
    <title>Robot Control</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial; margin: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        .button {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        }
        .button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .telemetry {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Robot Control</h1>
        
        <div>
            <button class="button" onclick="setMode('pid')" id="pidBtn">
                PID Mode
            </button>
            <button class="button" onclick="setMode('ddpg')" id="ddpgBtn">
                DDPG Mode
            </button>
        </div>

        <div class="telemetry">
            <h2>Live Telemetry</h2>
            <pre id="telemetryData">Waiting for data...</pre>
        </div>
    </div>

    <script>
        let mode = "";

        function setMode(newMode) {
            fetch("/mode?mode=" + newMode)
                .then(response => {
                    if(response.ok) {
                        mode = newMode;
                        updateButtonStates();
                    }
                });
        }

        function updateButtonStates() {
            document.getElementById("pidBtn").disabled = (mode === "pid");
            document.getElementById("ddpgBtn").disabled = (mode === "ddpg");
        }

        function updateTelemetry() {
            fetch("/telemetry")
                .then(response => response.json())
                .then(data => {
                    document.getElementById("telemetryData").textContent = 
                        JSON.stringify(data.telemetry, null, 2);
                    mode = data.telemetry.mode;
                    updateButtonStates();
                });
        }

        setInterval(updateTelemetry, 100);
    </script>
</body>
</html>
)rawliteral";
}

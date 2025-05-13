#include <Arduino.h>
#include <M5StickCPlus.h>
#include <Wire.h>
#include <WiFi.h>
#include <SPIFFS.h>
#include <UdpLogger.h>
#include <ddpg_controller.h>
#include <webserver.h>
#include <queue>

// Pin definitions
#define LED 10

// Calibration constants
#define N_CAL1 100
#define N_CAL2 100

// Display constants
#define LCD_VERTICAL_MIDDLE 120
#define ANGLE_LIMIT 45
#define SCREEN_WIDTH 135
#define SCREEN_HEIGHT 240
#define MODE_Y 30
#define VOLTAGE_Y 80 
#define STATUS_Y 130
#define ANGLE_Y 180

// Timing constants
#define DISPLAY_UPDATE_PERIOD 100 // 10Hz in ms
#define LOGGING_PERIOD 40         // 25Hz in ms

// Network configuration
const char* WIFI_SSID = "SSID";
const char* WIFI_PASSWORD = "PASSWORD";
const uint16_t UDP_PORT = 44444;

const int MOTOR_DELAY_STEPS = 2;
std::queue<float> motor_command_buffer;

// Forward declarations
void estimateAngle();
void delayWithEstimation(int n);
void driveMotor(byte ch, int8_t sp);
void driveMotorL(int16_t pwm);
void driveMotorR(int16_t pwm);
void resetMotor();
void resetPara();
void resetVar();
void gyroCalibration();
void accelInitialAngCalibration();
void calDelay(int n);
void displayBatteryVoltage();
void toggleMode();
void printMode();
void sendStatus();
void buildLogMessage(LogMessage &logMessage);
void updateDDPGStates();
void updateDDPGStatus();
void updateStandingDisplay();
void periodicDisplayUpdate();
void periodicLoggingUpdate();
void drive();
void checkButtonPress();
void initMotorCommandBuffer();

// System state
byte demoMode = 0;
boolean serialMonitor = true;
boolean standing = false;
int16_t counter = 0;
int16_t counterOverPwr = 0;
uint32_t time0 = 0, time1 = 0;

// Control variables
int8_t outputL = 0, outputR = 0;
float power = 0.0;
float powerR = 0.0, powerL = 0.0;
float varAng = 0.0, varOmg = 0.0;
float varSpd = 0.0, varDst = 0.0;
float varIang = 0.0;
float moveTarget = 0.0;
float initialAngle = 0.0;
float lastComplementaryAngle = 0.0;
float varAngDDPG = 0.0;
float varAngPID = 0.0;
float lastComplementaryAngleDDPG = 0.0;

// IMU variables
float gyroXoffset = 0.0, gyroYoffset = 0.0, accXoffset = 0.0;
float gyroXdata = 0.0, gyroYdata = 0.0, gyroZdata = 0.0;
float accXdata = 0.0, accZdata = 0.0;
float aveAccX = 0.0, aveAccZ = 0.0, aveAbsOmg = 0.0;

// System configuration
float cutoff = 0.1;
const float clk = 0.04;
const uint32_t interval = (uint32_t)(clk * 1000);
unsigned long lastDisplayUpdate = 0;
unsigned long lastLoggingUpdate = 0;

// Control parameters
float Kang = 18.0;
float Komg = 0.84;
float KIang = 600.0;
float Kdst = 65.0;
float Kspd = 1.3;
int16_t maxPwr;
int16_t fbBalance = 0;
int16_t motorDeadband = 0;
float mechFactR = 0.45, mechFactL = 0.45;

// Motor control variables
int16_t ipowerL = 0, ipowerR = 0;
int16_t motorLdir = 0, motorRdir = 0;
float vBatt = 0.0, voltAve = 3.7;
int16_t punchPwr = 20;
int16_t punchDur = 1;
int16_t punchPwr2;
int16_t punchCountL = 0, punchCountR = 0;

// Global objects
UdpLogger logger;
LogMessage logMessage;
DDPGController ddpgController;
TelemetryWebServer webServer;

// IMU initialization and reading functions
void imuInit() {
    M5.Imu.Init();
    M5.Imu.SetGyroFsr(M5.Imu.GFS_500DPS);
    M5.Imu.SetAccelFsr(M5.Imu.AFS_8G);
    if (serialMonitor) {
        Serial.println("MPU6886 initialized");
    }
}

void readImu() {
    float gX, gY, gZ, aX, aY, aZ;
    M5.Imu.getGyroData(&gX, &gY, &gZ);
    M5.Imu.getAccelData(&aX, &aY, &aZ);
    
    gyroYdata = gX;
    gyroZdata = -gY;
    gyroXdata = -gZ;
    accXdata = aZ;
    accZdata = aY;
}

void estimateAngle() {
    readImu();
    float gyroRate = gyroYdata - gyroYoffset;
    
    // DDPG angle calculation
    float accelAngleDDPG = (-atan2(accXdata, accZdata) * RAD_TO_DEG) + 180.0;
    
    if (abs(accelAngleDDPG - lastComplementaryAngleDDPG) > 180.0) {
        if (accelAngleDDPG > lastComplementaryAngleDDPG) {
            accelAngleDDPG -= 360.0;
        } else {
            accelAngleDDPG += 360.0;
        }
    }
    
    lastComplementaryAngleDDPG = (1.0 - cutoff) * (lastComplementaryAngleDDPG + gyroRate * clk) + 
                             cutoff * accelAngleDDPG;
    
    varAngDDPG = lastComplementaryAngleDDPG - initialAngle;
    
    // PID angle calculation
    varOmg = gyroRate;
    float calibratedAccelerationX = (accXdata - accXoffset);
    varAngPID += (varOmg + (calibratedAccelerationX * 57.3 - varAngPID) * cutoff) * clk;
    
    // Set angle based on current mode
    varAng = (demoMode == MODE_DDPG) ? varAngDDPG : varAngPID;
}

// Basic motor control functions
void driveMotor(byte ch, int8_t sp) {
    Wire.beginTransmission(0x38);
    Wire.write(ch);
    Wire.write(sp);
    Wire.endTransmission();
}

void driveMotorL(int16_t pwm) {
    outputL = (int8_t)constrain(pwm, -127, 127);
    driveMotor(0, outputL);
}

void driveMotorR(int16_t pwm) {
    outputR = (int8_t)constrain(-pwm, -127, 127);
    driveMotor(1, outputR);
}

void motorPowerWithPunch(int16_t power, int16_t& dir, int16_t& count, void (*driveFunc)(int16_t)) {
    if (power == 0) {
        driveFunc(0);
        dir = 0;
        return;
    }

    bool isPositive = power > 0;
    if (dir == (isPositive ? 1 : -1)) {
        count = constrain(++count, 0, 100);
    } else {
        count = 0;
    }
    dir = isPositive ? 1 : -1;

    if (count < punchDur) {
        driveFunc(isPositive ? max(power, punchPwr2) : min(power, (int16_t)-punchPwr2));
    } else {
        driveFunc(isPositive ? max(power, motorDeadband) : min(power, (int16_t)-motorDeadband));
    }
}

void drive() {
    if (!standing) return;

    if (demoMode == MODE_DDPG && ddpgController.isInitialized()) {
        // Get action from DDPG controller
        float action = ddpgController.getAction(
            lastComplementaryAngleDDPG * DEG_TO_RAD,  // Convert to radians
            varOmg * DEG_TO_RAD                       // Convert to radians
        );

        // Store action in history buffer (only for tracking)
        motor_command_buffer.push(action);
        if (motor_command_buffer.size() > MOTOR_DELAY_STEPS) {
            motor_command_buffer.pop(); // Keep buffer size consistent
        }
        
        // float action = 0.0f;
        // action = constrain(action, -maxPwr, maxPwr);

        // // Create a random action for data collection
        // // float random_factor = random(600, 300) / 1000.0f;
        // float random_factor = 0.7f;
        // // action = random_factor * varAngDDPG * (0.6f * maxPwr) - (1.0f * maxPwr);
        // action = random_factor * maxPwr * (varAngDDPG / abs(varAngDDPG));
        // action = constrain(action, -maxPwr, maxPwr);

        driveMotorL(action);
        driveMotorR(action);
        
        // For logging and display
        powerL = powerR = action;
    } else {
        float lowerFactor = 1.0f/4.0f;
        varSpd += power * clk * lowerFactor;
        varDst += Kdst * (varSpd * clk * lowerFactor - moveTarget);
        varIang += KIang * varAng * clk * lowerFactor;
        
        power = varIang + varDst + (Kspd * varSpd) + (Kang * varAng) + (Komg * varOmg);

        powerR = power;
        powerL = power;

        // Store in history buffer (for consistency, not used for control)
        motor_command_buffer.push(power);
        if (motor_command_buffer.size() > MOTOR_DELAY_STEPS) {
            motor_command_buffer.pop();
        }

        ipowerL = (int16_t)constrain(power, -maxPwr, maxPwr);
        ipowerR = (int16_t)constrain(power, -maxPwr, maxPwr);
        
        motorPowerWithPunch(ipowerL, motorLdir, punchCountL, driveMotorL);
        motorPowerWithPunch(ipowerR, motorRdir, punchCountR, driveMotorR);
    }
}

void resetPara() {
    Kang = 37.0;
    Komg = 0.84;
    KIang = 800.0;
    Kdst = 85.0;
    Kspd = 2.7;
    
    mechFactL = mechFactR = 0.45;
    punchPwr = 20;
    punchDur = 1;
    fbBalance = -3;
    motorDeadband = 10;
    maxPwr = 127;
    punchPwr2 = max(punchPwr, motorDeadband);
}

void resetVar() {
    power = moveTarget = 0.0;
    varAng = varOmg = varDst = varSpd = varIang = 0.0;
    varAngDDPG = varAngPID = 0.0;
    lastComplementaryAngle = lastComplementaryAngleDDPG = initialAngle;
}

void resetMotor() {
    driveMotorR(0);
    driveMotorL(0);
    counterOverPwr = 0;
    punchCountL = punchCountR = 0;
    motorLdir = motorRdir = 0;
    
    initMotorCommandBuffer();

    ddpgController.resetHistories();
}

void gyroCalibration() {
    delayWithEstimation(30);
    digitalWrite(LED, LOW);
    delayWithEstimation(80);
    M5.Lcd.fillScreen(BLACK);
    M5.Lcd.setCursor(30, LCD_VERTICAL_MIDDLE);
    M5.Lcd.print(" Cal-1  ");
    
    gyroYoffset = 0.0;
    for (int i = 0; i < N_CAL1; i++) {
        readImu();
        gyroYoffset += gyroYdata;
        delay(9);
    }
    gyroYoffset /= (float)N_CAL1;
    
    M5.Lcd.fillScreen(BLACK);
    digitalWrite(LED, HIGH);
}

void accelInitialAngCalibration() {
    resetVar();
    resetMotor();
    digitalWrite(LED, LOW);
    delayWithEstimation(80);
    M5.Lcd.setCursor(30, LCD_VERTICAL_MIDDLE);
    M5.Lcd.println(" Cal-2  ");
    
    float sumAccX = 0.0, sumAccZ = 0.0;
    for (int i = 0; i < N_CAL2; i++) {
        readImu();
        sumAccX += accXdata;
        sumAccZ += accZdata;
        delay(9);
    }
    
    initialAngle = (-atan2(sumAccX / N_CAL2, sumAccZ / N_CAL2) * RAD_TO_DEG) + 180.0;
    if (initialAngle > 180.0) {
        initialAngle -= 360.0;
    }
    lastComplementaryAngle = lastComplementaryAngleDDPG = initialAngle;
    
    M5.Lcd.fillScreen(BLACK);
    digitalWrite(LED, HIGH);
}

void delayWithEstimation(int n) {
    for (int i = 0; i < n; i++) {
        estimateAngle();
        delay(9);
    }
}

void setup() {
    Serial.begin(115200);
    serialMonitor = Serial.available();
    pinMode(LED, OUTPUT);
    digitalWrite(LED, HIGH);
    
    M5.begin();
    Wire.begin(0, 26);
    delay(50);
    
    imuInit();
    
    M5.Axp.ScreenBreath(40);
    M5.Lcd.setRotation(2);
    M5.Lcd.setTextFont(4);
    M5.Lcd.fillScreen(BLACK);
    M5.Lcd.setTextSize(1);
    M5.Lcd.fontHeight(1);
    M5.Lcd.setTextDatum(MC_DATUM);
    
    resetMotor();
    resetPara();
    resetVar();
    
    delay(500);
    
    if (!SPIFFS.begin(true)) {
        Serial.println("SPIFFS Mount Failed");
        M5.Lcd.println("Storage Init Failed");
    }
    
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    int wifiTimeout = 0;
    while (WiFi.status() != WL_CONNECTED && wifiTimeout < 20) {
        delay(500);
        wifiTimeout++;
        Serial.print(".");
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi connected");

        // Print network info when WiFi connects
        Serial.print("IP Address: ");
        Serial.println(WiFi.localIP());
        Serial.print("Subnet Mask: ");
        Serial.println(WiFi.subnetMask());
        
        delay(1000);
        
        Serial.println("Initializing DDPG controller...");
        // max_delta is the maximum change allowed per step in the normalized action space [-1, 1]
        // 0.25 means 25% of the full range can change in a single step
        float max_delta = 0.25f;
        if (!ddpgController.init(maxPwr, max_delta)) {
            Serial.println("DDPG initialization failed");
            M5.Lcd.setCursor(0, 110);
            M5.Lcd.print("DDPG Init Failed");
        } else {
            Serial.println("DDPG initialized successfully");
            Serial.printf("Max PWM: %d, Max delta: %.2f\n", maxPwr, max_delta);
        }
        
        if (!logger.begin(WIFI_SSID, WIFI_PASSWORD, UDP_PORT)) {
            Serial.println("Failed to initialize UDP logger:");
            Serial.println(logger.getLastError());
        } else {
            Serial.println("UDP logger initialized");
        }

        webServer.begin(WIFI_SSID, WIFI_PASSWORD);
        Serial.println("Web server started");
    }
    
    gyroCalibration();

    // Initialize motor delay buffer
    initMotorCommandBuffer();

    Serial.printf("Free heap: %d\n", ESP.getFreeHeap());    
    Serial.println("Setup complete");
    time0 = millis();
}

void loop() {
    unsigned long currentTime = millis();
    
    checkButtonPress();
    estimateAngle();
    ddpgController.update();
    
    if (standing && abs(varAng) > ANGLE_LIMIT) {
        resetMotor();
        resetVar();
        standing = false;
        powerL = powerR = 0;
        delay(1000); // Delay to prevent immediate re-calibration
        estimateAngle();
    }
    
    if (!standing) {
        aveAbsOmg = aveAbsOmg * 0.9 + abs(varOmg) * 0.1;
        aveAccZ = aveAccZ * 0.9 + accZdata * 0.1;
        
        if (abs(aveAccZ) > 0.9 && aveAbsOmg < 1.5) {
            accelInitialAngCalibration();
            standing = true;
        }
    } else {
        drive();
    }
    periodicLoggingUpdate();
    
    if (currentTime - lastDisplayUpdate >= DISPLAY_UPDATE_PERIOD) {
        periodicDisplayUpdate();
        lastDisplayUpdate = currentTime;
    }
    
    // if (currentTime - lastLoggingUpdate >= LOGGING_PERIOD) {
    //     periodicLoggingUpdate();
    //     lastLoggingUpdate = currentTime;
    // }

    do time1 = millis();
    while (time1 - time0 < interval);
    time0 = time1;
}

void checkButtonPress() {
    byte pbtn = M5.Axp.GetBtnPress();
    if (pbtn == 0) {
        return;
    } else if (pbtn == 1) {
        toggleMode();
        printMode();
    } else if (pbtn == 2) {
        gyroCalibration();
    }
}

void toggleMode() {
    demoMode = ++demoMode % 2;
}

void printMode() {
    M5.Lcd.fillRect(0, 0, SCREEN_WIDTH, MODE_Y + 20, BLACK);
    M5.Lcd.setTextDatum(MC_DATUM);
    M5.Lcd.drawString(demoMode == MODE_DDPG ? "DDPG" : "PID", SCREEN_WIDTH/2, MODE_Y);
}

void displayBatteryVoltage() {
    M5.Lcd.setTextDatum(MC_DATUM);
    vBatt = M5.Axp.GetBatVoltage();
    String voltage = String(vBatt, 2) + "v";
    M5.Lcd.fillRect(0, VOLTAGE_Y - 10, SCREEN_WIDTH, 30, BLACK);
    M5.Lcd.drawString(voltage, SCREEN_WIDTH/2, VOLTAGE_Y);
}

void updateStandingDisplay() {
    M5.Lcd.fillRect(0, STATUS_Y - 10, SCREEN_WIDTH, 30, BLACK);
    if (!standing) {
        M5.Lcd.setTextDatum(MC_DATUM);
        M5.Lcd.drawString(String(-aveAccZ, 2), SCREEN_WIDTH/2, STATUS_Y);
    } else {
        M5.Lcd.setTextDatum(ML_DATUM);
        M5.Lcd.drawString("Ang:" + String(varAng, 1), 10, STATUS_Y);
    }
}

void updateDDPGStatus() {
    M5.Lcd.fillRect(0, ANGLE_Y - 10, SCREEN_WIDTH, 30, BLACK);
    M5.Lcd.setTextDatum(MC_DATUM);
    
    if (ddpgController.isInitialized()) {
        M5.Lcd.drawString("DDPG Ready", SCREEN_WIDTH/2, ANGLE_Y);
    } else if (ddpgController.isReceivingWeights()) {
        M5.Lcd.drawString("Loading: " + String((int)(ddpgController.getReceiveProgress() * 100)) + "%", 
                         SCREEN_WIDTH/2, ANGLE_Y);
    } else {
        M5.Lcd.drawString("Waiting...", SCREEN_WIDTH/2, ANGLE_Y);
    }
}

void periodicDisplayUpdate() {
    displayBatteryVoltage();
    printMode();
    updateStandingDisplay();
    updateDDPGStatus();
}

void sendStatus() {
    Serial.print(millis() - time0);
    Serial.print(" stand=");
    Serial.print(standing);
    Serial.print(" accX=");
    Serial.print(accXdata);
    Serial.print(" power=");
    Serial.print(power);
    Serial.print(" ang=");
    Serial.print(varAng);
    Serial.print(" mode=");
    Serial.print((demoMode == MODE_DDPG) ? "DDPG" : "PID");
    Serial.println();
}

void buildLogMessage(LogMessage &logMessage) {
    logMessage.timestamp = millis();
    logMessage.dt = clk;
    
    logMessage.theta = varAngDDPG * DEG_TO_RAD;
    logMessage.theta_dot = varOmg * DEG_TO_RAD;
    logMessage.theta_global = lastComplementaryAngleDDPG * DEG_TO_RAD;
    
    logMessage.model_output = power;
    logMessage.motor_pwm = outputL;
    
    logMessage.standing = standing;
    logMessage.model_active = (demoMode == MODE_DDPG) && ddpgController.isInitialized();
    logMessage.battery_voltage = vBatt;
    
    logMessage.acc_x = accXdata;
    logMessage.acc_z = accZdata;
    logMessage.gyro_x = gyroXdata;
}

void periodicLoggingUpdate() {
    buildLogMessage(logMessage);
    if (!logger.sendLogMessage(logMessage)) {
        Serial.print("Failed to send log message: ");
        Serial.println(logger.getLastError());
    }
    if (serialMonitor) {
        sendStatus();
    }
}

void setLedPattern(bool error) {
    if (error) {
        digitalWrite(LED, HIGH);
        delay(50);
        digitalWrite(LED, LOW);
        delay(50);
        digitalWrite(LED, HIGH);
    } else {
        digitalWrite(LED, HIGH);
        delay(100);
        digitalWrite(LED, LOW);
    }
}

void configureDDPG() {
    if (ddpgController.isInitialized()) {
        demoMode = MODE_DDPG;
        Serial.println("DDPG control enabled");
    } else {
        demoMode = MODE_PID;
        Serial.println("Falling back to PID control");
    }
}

void printHeapInfo() {
    Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
    Serial.printf("Minimum free heap: %d bytes\n", ESP.getMinFreeHeap());
    Serial.printf("Maximum alloc heap: %d bytes\n", ESP.getMaxAllocHeap());
}

void initMotorCommandBuffer() {
    // Clear the buffer
    while (!motor_command_buffer.empty()) {
        motor_command_buffer.pop();
    }
    
    // Fill with zeros
    for (int i = 0; i < MOTOR_DELAY_STEPS; i++) {
        motor_command_buffer.push(0.0f);
    }
}

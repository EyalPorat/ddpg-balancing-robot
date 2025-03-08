#ifndef DDPG_CONTROLLER_H
#define DDPG_CONTROLLER_H

#include <ddpg_network.h>
#include <model_receiver.h>
#include <vector>

class DDPGController {
public:
    DDPGController() {
        actor = nullptr;
        receiver = nullptr;
        initialized = false;
    }
    
    ~DDPGController() {
        if (actor) delete actor;
        if (receiver) delete receiver;
    }

    bool init(float max_action) {
        try {
            Serial.println("Starting DDPG init...");
            
            // Use PSRAM if available
            if (psramFound()) {
                Serial.println("PSRAM found");
                heap_caps_malloc_extmem_enable(20000);
            }

            // Initialize components one at a time with checks
            Serial.println("Creating actor...");
            if (!actor) {
                // (theta, theta_dot)
                actor = new DDPGActor(2, 8, 1, max_action);
                if (!actor) {
                    Serial.println("Failed to create actor");
                    return false;
                }
            }

            Serial.println("Creating receiver...");
            if (!receiver) {
                receiver = new ModelReceiver(44445);
                if (!receiver) {
                    Serial.println("Failed to create receiver");
                    return false;
                }
            }

            Serial.println("Initializing state buffer...");
            try {
                state_buffer.resize(2);
            } catch (const std::exception& e) {
                Serial.println("Failed to resize state buffer");
                return false;
            }

            Serial.println("Starting receiver...");
            if (!receiver->begin()) {
                Serial.println("Failed to start receiver");
                return false;
            }

            // Try to load existing weights if available
            if (SPIFFS.exists("/actor_weights.bin")) {
                Serial.println("Found existing weights file");
                File f = SPIFFS.open("/actor_weights.bin", FILE_READ);
                if (f) {
                    Serial.printf("Weights file size: %d bytes\n", f.size());
                    f.close();
                    
                    if (actor->loadWeights("/actor_weights.bin")) {
                        initialized = true;
                        Serial.println("Successfully loaded weights");
                    } else {
                        Serial.println("Failed to load weights");
                    }
                } else {
                    Serial.println("Could not open weights file");
                }
            } else {
                Serial.println("No weights file found");
            }
            

            Serial.println("DDPG init complete");
            return true;
            
        } catch (const std::exception& e) {
            Serial.print("Error in DDPG init: ");
            Serial.println(e.what());
            return false;
        } catch (...) {
            Serial.println("Unknown error in DDPG init");
            return false;
        }
    }

    void update() {
        if (!receiver) return;
        receiver->update();
        
        if (!initialized && receiver->getProgress() == 1.0f) {
            if (actor && actor->loadWeights("/actor_weights.bin")) {
                initialized = true;
                Serial.println("Loaded new weights");
            }
        }
    }

    float getAction(float theta, float theta_dot) {
        if (!initialized || !actor) return 0.0f;

        state_buffer[0] = theta;
        state_buffer[1] = theta_dot;

        return actor->forward(state_buffer);
    }

    bool isInitialized() const { return initialized; }
    bool isReceivingWeights() const { return receiver ? receiver->isReceiving() : false; }
    float getReceiveProgress() const { return receiver ? receiver->getProgress() : 0.0f; }

private:
    DDPGActor* actor;
    ModelReceiver* receiver;
    std::vector<float> state_buffer;
    bool initialized;
};

#endif // DDPG_CONTROLLER_H

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
        prev_action = 0.0f;
        max_action = 0.0f;
        max_delta = 0.1f;  // Default to 10% maximum change per step
        current_motor_command = 0.0f;
    }
    
    ~DDPGController() {
        if (actor) delete actor;
        if (receiver) delete receiver;
    }

    bool init(float max_action, float max_delta = 0.1f) {
        try {
            Serial.println("Starting DDPG init...");
            
            // max_action is the maximum PWM value (typically 127)
            // Used to scale the normalized [-1, 1] output from the actor
            this->max_action = max_action;
            
            // max_delta is the maximum allowed change in normalized action space [-1, 1]
            // It's a fraction between 0 and 1
            this->max_delta = max_delta;

            // Use PSRAM if available
            if (psramFound()) {
                Serial.println("PSRAM found");
                heap_caps_malloc_extmem_enable(20000);
            }

            // Initialize components one at a time with checks
            Serial.println("Creating actor...");
            if (!actor) {
                // (theta, theta_dot, prev_action)
                actor = new DDPGActor(3, 10, 1, 1.0f);  // Note: Actor always outputs in [-1, 1] range
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
                state_buffer.resize(3);
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

        // Uninitialize controller if receiver is not receiving
        if (receiver->isReceiving()) {
            initialized = false;
        }
        
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
        state_buffer[2] = current_motor_command / max_action;  // Normalize to [-1, 1]
    
        // Get the delta action from the actor (already in [-1, 1] range)
        float delta_action = actor->forward(state_buffer);
        
        // Scale delta by max_delta (which is a fraction of the normalized range)
        delta_action *= max_delta;
        
        // Apply delta to current command and clip to valid range
        float new_command = current_motor_command + (delta_action * max_action);
        new_command = constrain(new_command, -max_action, max_action);
        
        // Store the normalized action for the next step
        prev_action = delta_action;
        
        // Update current motor command
        current_motor_command = new_command;
        
        return new_command;
    }

    bool isInitialized() const { return initialized; }
    bool isReceivingWeights() const { return receiver ? receiver->isReceiving() : false; }
    float getReceiveProgress() const { return receiver ? receiver->getProgress() : 0.0f; }

private:
    DDPGActor* actor;
    ModelReceiver* receiver;
    std::vector<float> state_buffer;
    bool initialized;
    float prev_action;  // Previous delta action
    float max_action;   // Maximum torque/PWM
    float max_delta;    // Maximum change allowed per step (as fraction)
    float current_motor_command;  // Current applied motor command
};

#endif // DDPG_CONTROLLER_H

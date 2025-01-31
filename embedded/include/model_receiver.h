#ifndef MODEL_RECEIVER_H
#define MODEL_RECEIVER_H

#include <WiFiUdp.h>
#include <SPIFFS.h>
#include <vector>

class ModelReceiver {
public:
    ModelReceiver(uint16_t port = 44445) : 
        port(port), 
        receiving(false), 
        transfer_complete(false),
        totalChunks(0),
        lastReceiveTime(0)
    {
        chunks_received.reserve(1000);  // Pre-allocate space for tracking
    }

    bool begin() {
        udp.begin(port);
        return true;
    }

    void update() {
        // Check for timeout conditions
        if (receiving && (millis() - lastReceiveTime > 5000)) {  // 2 second timeout
            if (getProgress() == 1.0f) {
                transfer_complete = true;
                receiving = false;
                chunks_received.clear();
                Serial.println("Transfer completed and finalized");
            } else {
                // Incomplete transfer timeout
                receiving = false;
                chunks_received.clear();
                transfer_complete = false;
                Serial.println("Transfer timeout - incomplete");
            }
        }

        int packetSize = udp.parsePacket();
        if (packetSize) {
            if (!receiving) {
                // Reset completion flag on new transfer
                transfer_complete = false;
                
                // New transfer starting
                uint32_t numChunks;
                udp.read((uint8_t*)&numChunks, sizeof(numChunks));
                
                // Initialize new transfer
                startNewTransfer(numChunks);
            } else {
                // Read chunk index
                uint32_t chunkIndex;
                udp.read((uint8_t*)&chunkIndex, sizeof(chunkIndex));
                
                // Read chunk data
                int dataSize = packetSize - sizeof(chunkIndex);
                if (dataSize > 0) {
                    uint8_t buffer[1024];
                    udp.read(buffer, dataSize);
                    
                    // Process chunk
                    processChunk(chunkIndex, buffer, dataSize);
                }
            }
        }
    }

    bool isCompleted() const {
        return transfer_complete;
    }

    bool isReceiving() const { 
        return receiving && !transfer_complete;
    }
    
    float getProgress() const { 
        if (!receiving && !transfer_complete) return 0.0f;
        if (totalChunks == 0) return 0.0f;
        if (transfer_complete) return 1.0f;
        return (float)chunks_received.size() / totalChunks;
    }

private:
    WiFiUDP udp;
    uint16_t port;
    bool receiving;
    bool transfer_complete;
    uint32_t totalChunks;
    File currentFile;
    std::vector<bool> chunks_received;
    uint32_t lastReceiveTime;
    
    void startNewTransfer(uint32_t numChunks) {
        Serial.printf("Starting new transfer, expecting %d chunks\n", numChunks);
        
        // Remove old file if exists
        if (SPIFFS.exists("/actor_weights.bin")) {
            SPIFFS.remove("/actor_weights.bin");
            Serial.println("Removed old weights file");
        }
        
        // Open new file
        currentFile = SPIFFS.open("/actor_weights.bin", FILE_WRITE);
        if (!currentFile) {
            Serial.println("Failed to open file for writing");
            return;
        }
        
        receiving = true;
        transfer_complete = false;
        totalChunks = numChunks;
        chunks_received.clear();
        chunks_received.resize(totalChunks, false);
        lastReceiveTime = millis();
        
        Serial.printf("Transfer started, file opened for writing\n");
    }
    
    void processChunk(uint32_t chunkIndex, uint8_t* data, size_t length) {
        if (!receiving || chunkIndex >= totalChunks) {
            Serial.println("Invalid chunk received");
            return;
        }
        
        lastReceiveTime = millis();
        
        // Instead of seeking, write chunks to temporary files
        char tempFilename[32];
        snprintf(tempFilename, sizeof(tempFilename), "/chunk_%d.tmp", chunkIndex);
        
        File chunkFile = SPIFFS.open(tempFilename, FILE_WRITE);
        if (!chunkFile) {
            Serial.printf("Failed to create chunk file %s\n", tempFilename);
            return;
        }
        
        if (chunkFile.write(data, length) != length) {
            Serial.println("Failed to write chunk data");
            chunkFile.close();
            return;
        }
        chunkFile.close();
        
        // Mark chunk as received
        chunks_received[chunkIndex] = true;
        
        // Progress logging
        float progress = getProgress() * 100;
        if (((int)progress % 10) == 0) {
            Serial.printf("Transfer progress: %.1f%%\n", progress);
        }
        
        // Check if we have all chunks
        bool complete = true;
        for (bool received : chunks_received) {
            if (!received) {
                complete = false;
                break;
            }
        }
        
        if (complete) {
            finishTransfer();
        }
    }
    
    void finishTransfer() {
        if (!receiving) {
            Serial.println("Not in receiving state during finishTransfer");
            return;
        }

        Serial.println("Starting transfer finalization...");
        
        // First, close any open file handle
        if (currentFile) {
            currentFile.close();
        }

        // Create the final file
        File finalFile = SPIFFS.open("/actor_weights.bin", FILE_WRITE);
        if (!finalFile) {
            Serial.println("Failed to create final file");
            cleanupChunks();
            receiving = false;
            transfer_complete = false;
            return;
        }
        
        // Combine all chunks in order
        bool success = true;
        uint32_t totalBytesWritten = 0;
        
        for (uint32_t i = 0; i < totalChunks && success; i++) {
            char tempFilename[32];
            snprintf(tempFilename, sizeof(tempFilename), "/chunk_%d.tmp", i);
            
            File chunkFile = SPIFFS.open(tempFilename, FILE_READ);
            if (!chunkFile) {
                Serial.printf("Failed to open chunk file %s\n", tempFilename);
                success = false;
                break;
            }
            
            // Copy chunk data to final file
            uint8_t buffer[256];
            size_t bytesAvailable = chunkFile.available();
            size_t chunkBytesWritten = 0;
            
            Serial.printf("Processing chunk %d, bytes available: %d\n", i, bytesAvailable);
            
            while (chunkFile.available() && success) {
                size_t bytesRead = chunkFile.read(buffer, sizeof(buffer));
                size_t bytesWritten = finalFile.write(buffer, bytesRead);
                
                if (bytesWritten != bytesRead) {
                    Serial.printf("Write mismatch: read %d bytes but wrote %d bytes\n", 
                                bytesRead, bytesWritten);
                    success = false;
                    break;
                }
                
                chunkBytesWritten += bytesWritten;
                totalBytesWritten += bytesWritten;
            }
            
            chunkFile.close();
            Serial.printf("Chunk %d written: %d bytes\n", i, chunkBytesWritten);
        }
        
        // Ensure all data is written and close file
        finalFile.flush();
        finalFile.close();
        
        Serial.printf("Total bytes written to final file: %d\n", totalBytesWritten);
        
        // Clean up temporary files
        cleanupChunks();
        
        if (success) {
            // Verify final file
            delay(100);  // Give filesystem a moment
            
            File verifyFile = SPIFFS.open("/actor_weights.bin", FILE_READ);
            if (verifyFile) {
                size_t fileSize = verifyFile.size();
                Serial.printf("Final file verification - Size: %d bytes\n", fileSize);
                
                if (fileSize > 0) {
                    // Read and verify first few bytes
                    uint8_t header[16];
                    size_t bytesRead = verifyFile.read(header, sizeof(header));
                    Serial.printf("File header (%d bytes): ", bytesRead);
                    for(size_t i = 0; i < bytesRead && i < 16; i++) {
                        Serial.printf("%02X ", header[i]);
                    }
                    Serial.println();
                    
                    receiving = false;
                    transfer_complete = true;
                    chunks_received.clear();
                    Serial.println("Transfer successfully completed!");
                } else {
                    Serial.println("Error: Final file is empty!");
                    success = false;
                }
                verifyFile.close();
            } else {
                Serial.println("Failed to open file for verification");
                success = false;
            }
        }
        
        if (!success) {
            Serial.println("Transfer failed during finalization");
            SPIFFS.remove("/actor_weights.bin");
            receiving = false;
            transfer_complete = false;
            chunks_received.clear();
        }
    }

    void cleanupChunks() {
        for (uint32_t i = 0; i < totalChunks; i++) {
            char tempFilename[32];
            snprintf(tempFilename, sizeof(tempFilename), "/chunk_%d.tmp", i);
            if (SPIFFS.exists(tempFilename)) {
                if (SPIFFS.remove(tempFilename)) {
                    Serial.printf("Removed temporary file: %s\n", tempFilename);
                } else {
                    Serial.printf("Failed to remove temporary file: %s\n", tempFilename);
                }
            }
        }
    }
};

#endif // MODEL_RECEIVER_H
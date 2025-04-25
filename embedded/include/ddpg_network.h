#ifndef DDPG_NETWORK_H
#define DDPG_NETWORK_H

#ifdef min
#undef min
#endif

#ifdef max
#undef max  
#endif

#include <SPIFFS.h>
#include <vector>
#include <cmath>

// Simple Matrix class for efficient computations
template<typename T>
class Matrix {
public:
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        // Use PSRAM if available
        if (psramFound()) {
            data = (T*)ps_malloc(rows * cols * sizeof(T));
        } else {
            data = (T*)malloc(rows * cols * sizeof(T));
        }
        if (!data) {
            Serial.println("Failed to allocate matrix memory");
        }
    }
    
    ~Matrix() {
        if (data) {
            free(data);
        }
    }
    
    T& operator()(size_t i, size_t j) { return data[i * cols + j]; }
    const T& operator()(size_t i, size_t j) const { return data[i * cols + j]; }
    
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    T* getData() { return data; }
    
private:
    T* data;
    size_t rows, cols;
};

// LayerNorm parameters struct
struct LayerNormParams {
    std::vector<float> gamma;
    std::vector<float> beta;
    
    LayerNormParams(size_t size) : gamma(size, 1.0f), beta(size, 0.0f) {}
};

class DDPGActor {
public:
    DDPGActor(size_t state_dim, size_t hidden_dim, size_t action_dim, float max_action)
        : l1_weights(hidden_dim, state_dim),
          l1_bias(hidden_dim, 1),
          l1_norm(hidden_dim),
          l2_weights(hidden_dim, hidden_dim),
          l2_bias(hidden_dim, 1),
          l2_norm(hidden_dim),
          l3_weights(action_dim, hidden_dim),
          l3_bias(action_dim, 1),
          max_action(max_action) {}

    bool loadWeights(const char* filename) {
        File file = SPIFFS.open(filename, FILE_READ);
        if (!file) {
            Serial.printf("Failed to open weights file: %s\n", filename);
            return false;
        }

        Serial.printf("Opened weights file, size: %d bytes\n", file.size());
        size_t currentPosition = 0;

        // Read first layer (l1)
        const char* layerName = "l1";
        uint32_t rows, cols;
        
        Serial.printf("\n=== Reading layer %s at position %d ===\n", layerName, currentPosition);
        
        // Debug: Read and print raw bytes for dimensions
        uint8_t dim_bytes[8];
        size_t bytes_read = file.read(dim_bytes, 8);
        if (bytes_read != 8) {
            Serial.printf("Failed to read dimensions bytes. Expected 8, got %d\n", bytes_read);
            file.close();
            return false;
        }

        Serial.printf("Raw dimension bytes: ");
        for(int i = 0; i < 8; i++) {
            Serial.printf("%02X ", dim_bytes[i]);
        }
        Serial.println();

        // Convert bytes to dimensions
        memcpy(&rows, dim_bytes, 4);
        memcpy(&cols, dim_bytes + 4, 4);
        currentPosition += 8;

        Serial.printf("Layer %s dimensions decoded: %dx%d (at position %d)\n", 
                    layerName, rows, cols, currentPosition);

        // Sanity check dimensions
        if (rows != 10 || cols != 3) {
            Serial.printf("ERROR: Invalid dimensions for layer %s: %dx%d. Expected 10x3\n", layerName, rows, cols);
            file.close();
            return false;
        }

        // Calculate expected data sizes
        size_t weights_size = rows * cols * sizeof(float);
        size_t bias_size = rows * sizeof(float);
        Serial.printf("Expected sizes - weights: %d bytes, bias: %d bytes\n", weights_size, bias_size);

        // Read weights
        uint8_t* weights_buffer = (uint8_t*)malloc(weights_size);
        if (!weights_buffer) {
            Serial.printf("Failed to allocate %d bytes for weights in layer %s\n", weights_size, layerName);
            file.close();
            return false;
        }

        bytes_read = file.read(weights_buffer, weights_size);
        if (bytes_read != weights_size) {
            Serial.printf("Failed to read weights for layer %s. Expected %d bytes, got %d\n", 
                        layerName, weights_size, bytes_read);
            free(weights_buffer);
            file.close();
            return false;
        }
        currentPosition += weights_size;

        // Debug: Print first few weights
        float* weights_float = (float*)weights_buffer;
        Serial.printf("First few weights: ");
        for(int i = 0; i < min(5, (int)(weights_size/4)); i++) {
            Serial.printf("%f ", weights_float[i]);
        }
        Serial.println();

        // Read biases
        uint8_t* bias_buffer = (uint8_t*)malloc(bias_size);
        if (!bias_buffer) {
            Serial.printf("Failed to allocate %d bytes for bias in layer %s\n", bias_size, layerName);
            free(weights_buffer);
            file.close();
            return false;
        }

        bytes_read = file.read(bias_buffer, bias_size);
        if (bytes_read != bias_size) {
            Serial.printf("Failed to read bias for layer %s. Expected %d bytes, got %d\n", 
                        layerName, bias_size, bytes_read);
            free(weights_buffer);
            free(bias_buffer);
            file.close();
            return false;
        }
        currentPosition += bias_size;

        // Debug: Print first few biases
        float* bias_float = (float*)bias_buffer;
        Serial.printf("First few biases: ");
        for(int i = 0; i < min(5, (int)(bias_size/4)); i++) {
            Serial.printf("%f ", bias_float[i]);
        }
        Serial.println();

        // Copy data to L1 matrices
        memcpy(l1_weights.getData(), weights_buffer, weights_size);
        memcpy(l1_bias.getData(), bias_buffer, bias_size);

        free(weights_buffer);
        free(bias_buffer);

        // Read LayerNorm parameters for l1
        size_t norm_size = rows * sizeof(float);
        
        // Read gamma
        uint8_t* gamma_buffer = (uint8_t*)malloc(norm_size);
        if (!gamma_buffer) {
            Serial.printf("Failed to allocate memory for gamma in layer %s\n", layerName);
            file.close();
            return false;
        }
        
        bytes_read = file.read(gamma_buffer, norm_size);
        if (bytes_read != norm_size) {
            Serial.printf("Failed to read gamma for layer %s. Expected %d bytes, got %d\n", 
                        layerName, norm_size, bytes_read);
            free(gamma_buffer);
            file.close();
            return false;
        }

        // Debug: Print first few gamma values
        float* gamma_float = (float*)gamma_buffer;
        Serial.printf("First few gamma values: ");
        for(int i = 0; i < min(5, (int)(norm_size/4)); i++) {
            Serial.printf("%f ", gamma_float[i]);
        }
        Serial.println();

        memcpy(l1_norm.gamma.data(), gamma_buffer, norm_size);
        free(gamma_buffer);
        currentPosition += norm_size;

        // Read beta
        uint8_t* beta_buffer = (uint8_t*)malloc(norm_size);
        if (!beta_buffer) {
            Serial.printf("Failed to allocate memory for beta in layer %s\n", layerName);
            file.close();
            return false;
        }
        
        bytes_read = file.read(beta_buffer, norm_size);
        if (bytes_read != norm_size) {
            Serial.printf("Failed to read beta for layer %s. Expected %d bytes, got %d\n", 
                        layerName, norm_size, bytes_read);
            free(beta_buffer);
            file.close();
            return false;
        }

        // Debug: Print first few beta values
        float* beta_float = (float*)beta_buffer;
        Serial.printf("First few beta values: ");
        for(int i = 0; i < min(5, (int)(norm_size/4)); i++) {
            Serial.printf("%f ", beta_float[i]);
        }
        Serial.println();

        memcpy(l1_norm.beta.data(), beta_buffer, norm_size);
        free(beta_buffer);
        currentPosition += norm_size;

        // Now read the second layer (l2)
        layerName = "l2";
        
        Serial.printf("\n=== Reading layer %s at position %d ===\n", layerName, currentPosition);
        
        // Read dimensions
        bytes_read = file.read(dim_bytes, 8);
        if (bytes_read != 8) {
            Serial.printf("Failed to read dimensions bytes for layer %s. Expected 8, got %d\n", 
                        layerName, bytes_read);
            file.close();
            return false;
        }

        // Convert bytes to dimensions
        memcpy(&rows, dim_bytes, 4);
        memcpy(&cols, dim_bytes + 4, 4);
        currentPosition += 8;

        Serial.printf("Layer %s dimensions decoded: %dx%d (at position %d)\n", 
                    layerName, rows, cols, currentPosition);

        // Sanity check dimensions - second layer is now 10x10
        if (rows != 10 || cols != 10) {
            Serial.printf("ERROR: Invalid dimensions for layer %s: %dx%d. Expected 10x10\n", layerName, rows, cols);
            file.close();
            return false;
        }

        // Calculate expected data sizes for second layer
        weights_size = rows * cols * sizeof(float);
        bias_size = rows * sizeof(float);
        
        // Read weights for second layer
        weights_buffer = (uint8_t*)malloc(weights_size);
        if (!weights_buffer) {
            Serial.printf("Failed to allocate memory for weights in layer %s\n", layerName);
            file.close();
            return false;
        }
        
        bytes_read = file.read(weights_buffer, weights_size);
        if (bytes_read != weights_size) {
            Serial.printf("Failed to read weights for layer %s. Expected %d bytes, got %d\n", 
                        layerName, weights_size, bytes_read);
            free(weights_buffer);
            file.close();
            return false;
        }
        currentPosition += weights_size;
        
        // Debug: Print first few weights
        weights_float = (float*)weights_buffer;
        Serial.printf("Second layer weights sample: ");
        for(int i = 0; i < min(5, (int)(weights_size/4)); i++) {
            Serial.printf("%f ", weights_float[i]);
        }
        Serial.println();
        
        // Read biases for second layer
        bias_buffer = (uint8_t*)malloc(bias_size);
        if (!bias_buffer) {
            Serial.printf("Failed to allocate memory for bias in layer %s\n", layerName);
            free(weights_buffer);
            file.close();
            return false;
        }
        
        bytes_read = file.read(bias_buffer, bias_size);
        if (bytes_read != bias_size) {
            Serial.printf("Failed to read bias for layer %s. Expected %d bytes, got %d\n", 
                        layerName, bias_size, bytes_read);
            free(weights_buffer);
            free(bias_buffer);
            file.close();
            return false;
        }
        currentPosition += bias_size;
        
        // Debug: Print second layer biases
        bias_float = (float*)bias_buffer;
        Serial.printf("Second layer biases sample: ");
        for(int i = 0; i < min(5, (int)(bias_size/4)); i++) {
            Serial.printf("%f ", bias_float[i]);
        }
        Serial.println();
        
        // Copy data to L2 matrices
        memcpy(l2_weights.getData(), weights_buffer, weights_size);
        memcpy(l2_bias.getData(), bias_buffer, bias_size);
        
        free(weights_buffer);
        free(bias_buffer);

        // Read LayerNorm parameters for l2
        norm_size = rows * sizeof(float);
        
        // Read gamma for L2
        gamma_buffer = (uint8_t*)malloc(norm_size);
        if (!gamma_buffer) {
            Serial.printf("Failed to allocate memory for gamma in layer %s\n", layerName);
            file.close();
            return false;
        }
        
        bytes_read = file.read(gamma_buffer, norm_size);
        if (bytes_read != norm_size) {
            Serial.printf("Failed to read gamma for layer %s. Expected %d bytes, got %d\n", 
                        layerName, norm_size, bytes_read);
            free(gamma_buffer);
            file.close();
            return false;
        }

        // Debug: Print first few gamma values for L2
        gamma_float = (float*)gamma_buffer;
        Serial.printf("L2 gamma values sample: ");
        for(int i = 0; i < min(5, (int)(norm_size/4)); i++) {
            Serial.printf("%f ", gamma_float[i]);
        }
        Serial.println();

        memcpy(l2_norm.gamma.data(), gamma_buffer, norm_size);
        free(gamma_buffer);
        currentPosition += norm_size;

        // Read beta for L2
        beta_buffer = (uint8_t*)malloc(norm_size);
        if (!beta_buffer) {
            Serial.printf("Failed to allocate memory for beta in layer %s\n", layerName);
            file.close();
            return false;
        }
        
        bytes_read = file.read(beta_buffer, norm_size);
        if (bytes_read != norm_size) {
            Serial.printf("Failed to read beta for layer %s. Expected %d bytes, got %d\n", 
                        layerName, norm_size, bytes_read);
            free(beta_buffer);
            file.close();
            return false;
        }

        // Debug: Print first few beta values for L2
        beta_float = (float*)beta_buffer;
        Serial.printf("L2 beta values sample: ");
        for(int i = 0; i < min(5, (int)(norm_size/4)); i++) {
            Serial.printf("%f ", beta_float[i]);
        }
        Serial.println();

        memcpy(l2_norm.beta.data(), beta_buffer, norm_size);
        free(beta_buffer);
        currentPosition += norm_size;

        // Now read the output layer (l3)
        layerName = "l3";
        
        Serial.printf("\n=== Reading layer %s at position %d ===\n", layerName, currentPosition);
        
        // Read dimensions
        bytes_read = file.read(dim_bytes, 8);
        if (bytes_read != 8) {
            Serial.printf("Failed to read dimensions bytes for layer %s. Expected 8, got %d\n", 
                        layerName, bytes_read);
            file.close();
            return false;
        }

        // Convert bytes to dimensions
        memcpy(&rows, dim_bytes, 4);
        memcpy(&cols, dim_bytes + 4, 4);
        currentPosition += 8;

        Serial.printf("Layer %s dimensions decoded: %dx%d (at position %d)\n", 
                    layerName, rows, cols, currentPosition);

        // Sanity check dimensions - Output layer is 1x10
        if (rows != 1 || cols != 10) {
            Serial.printf("ERROR: Invalid dimensions for layer %s: %dx%d. Expected 1x10\n", layerName, rows, cols);
            file.close();
            return false;
        }

        // Calculate expected data sizes for output layer
        weights_size = rows * cols * sizeof(float);
        bias_size = rows * sizeof(float);
        
        // Read weights for output layer
        weights_buffer = (uint8_t*)malloc(weights_size);
        if (!weights_buffer) {
            Serial.printf("Failed to allocate memory for weights in layer %s\n", layerName);
            file.close();
            return false;
        }
        
        bytes_read = file.read(weights_buffer, weights_size);
        if (bytes_read != weights_size) {
            Serial.printf("Failed to read weights for layer %s. Expected %d bytes, got %d\n", 
                        layerName, weights_size, bytes_read);
            free(weights_buffer);
            file.close();
            return false;
        }
        currentPosition += weights_size;
        
        // Debug: Print output weights
        weights_float = (float*)weights_buffer;
        Serial.printf("Output weights: ");
        for(int i = 0; i < min(5, (int)(weights_size/4)); i++) {
            Serial.printf("%f ", weights_float[i]);
        }
        Serial.println();
        
        // Read biases for output layer
        bias_buffer = (uint8_t*)malloc(bias_size);
        if (!bias_buffer) {
            Serial.printf("Failed to allocate memory for bias in layer %s\n", layerName);
            free(weights_buffer);
            file.close();
            return false;
        }
        
        bytes_read = file.read(bias_buffer, bias_size);
        if (bytes_read != bias_size) {
            Serial.printf("Failed to read bias for layer %s. Expected %d bytes, got %d\n", 
                        layerName, bias_size, bytes_read);
            free(weights_buffer);
            free(bias_buffer);
            file.close();
            return false;
        }
        currentPosition += bias_size;
        
        // Debug: Print output bias
        bias_float = (float*)bias_buffer;
        Serial.printf("Output bias: %f\n", bias_float[0]);
        
        // Copy data to L3 matrices (output layer)
        memcpy(l3_weights.getData(), weights_buffer, weights_size);
        memcpy(l3_bias.getData(), bias_buffer, bias_size);
        
        free(weights_buffer);
        free(bias_buffer);

        file.close();
        Serial.printf("\nSuccessfully loaded all weights. Total bytes read: %d\n", currentPosition);
        return true;
    }

    float forward(const std::vector<float>& state) {
        // Layer 1
        std::vector<float> h1(l1_weights.getRows());
        for (size_t i = 0; i < l1_weights.getRows(); i++) {
            float sum = l1_bias(i, 0);
            for (size_t j = 0; j < l1_weights.getCols(); j++) {
                sum += l1_weights(i, j) * state[j];
            }
            h1[i] = sum;
        }

        // Apply LayerNorm and ReLU to h1
        layerNorm(h1, l1_norm);
        for (auto& val : h1) val = relu(val);

        // Layer 2
        std::vector<float> h2(l2_weights.getRows());
        for (size_t i = 0; i < l2_weights.getRows(); i++) {
            float sum = l2_bias(i, 0);
            for (size_t j = 0; j < l2_weights.getCols(); j++) {
                sum += l2_weights(i, j) * h1[j];
            }
            h2[i] = sum;
        }

        // Apply LayerNorm and ReLU to h2
        layerNorm(h2, l2_norm);
        for (auto& val : h2) val = relu(val);

        // Output layer
        float output = l3_bias(0, 0);
        for (size_t j = 0; j < l3_weights.getCols(); j++) {
            output += l3_weights(0, j) * h2[j];
        }
        
        // Apply tanh and scale by max_action
        // max_action is always 1.0 here, ensuring output is in [-1, 1] range
        // The actual scaling to PWM values happens in DDPGController::getAction
        return max_action * tanh(output);
    }

private:
    Matrix<float> l1_weights, l1_bias;
    Matrix<float> l2_weights, l2_bias;
    Matrix<float> l3_weights, l3_bias;
    LayerNormParams l1_norm;
    LayerNormParams l2_norm;
    float max_action;

    void layerNorm(std::vector<float>& x, const LayerNormParams& params) {
        // Calculate mean
        float mean = 0.0f;
        for (const auto& val : x) mean += val;
        mean /= x.size();
        
        // Calculate variance
        float var = 0.0f;
        for (const auto& val : x) {
            float diff = val - mean;
            var += diff * diff;
        }
        var /= x.size();
        
        // Normalize
        const float eps = 1e-5f;
        float inv_std = 1.0f / sqrt(var + eps);
        
        for (size_t i = 0; i < x.size(); i++) {
            x[i] = params.gamma[i] * (x[i] - mean) * inv_std + params.beta[i];
        }
    }

    float relu(float x) const { return x > 0 ? x : 0; }
    
    float tanh(float x) const { 
        float exp2x = exp(2 * x);
        return (exp2x - 1) / (exp2x + 1);
    }
};

#endif // DDPG_NETWORK_H

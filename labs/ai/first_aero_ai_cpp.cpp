/*
C++ Implementation: Linear Regression with Gradient Descent
Equivalent to the first AI written in Aero programming language
This serves as a reference implementation for comparison

Algorithm: Linear Regression with Gradient Descent
Problem: Predict house prices based on size (supervised learning)
Method: Minimize mean squared error using gradient descent optimization
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>

class AeroAI {
private:
    double weight;
    double bias;
    double learning_rate;
    std::vector<double> house_sizes;
    std::vector<double> house_prices;
    
public:
    AeroAI() : weight(1.0), bias(0.0), learning_rate(0.001) {
        // Initialize training data (same as Aero implementation)
        house_sizes = {10, 12, 15, 18, 20, 22, 25, 28, 30, 32};  // hundreds of sq ft
        house_prices = {15, 18, 22, 26, 30, 33, 38, 42, 45, 48}; // $10k units
    }
    
    double predict(double size) {
        return weight * size + bias;
    }
    
    double computeMSE(const std::vector<double>& predictions) {
        double mse = 0.0;
        for (size_t i = 0; i < house_prices.size(); i++) {
            double error = house_prices[i] - predictions[i];
            mse += error * error;
        }
        return mse / house_prices.size();
    }
    
    void train(int epochs) {
        std::cout << "=== C++ AI: Linear Regression with Gradient Descent ===" << std::endl;
        std::cout << "Equivalent implementation to the first Aero AI" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        std::cout << "Training Data: " << house_sizes.size() << " houses" << std::endl;
        std::cout << "Size range: " << *std::min_element(house_sizes.begin(), house_sizes.end()) 
                  << "-" << *std::max_element(house_sizes.begin(), house_sizes.end()) 
                  << " (hundreds sq ft)" << std::endl;
        std::cout << "Price range: $" << *std::min_element(house_prices.begin(), house_prices.end())
                  << "-" << *std::max_element(house_prices.begin(), house_prices.end()) 
                  << "0k" << std::endl;
        
        std::cout << "\nInitial Parameters:" << std::endl;
        std::cout << "Weight: " << weight << ", Bias: " << bias << std::endl;
        std::cout << "Learning Rate: " << learning_rate << ", Epochs: " << epochs << std::endl;
        
        std::cout << "\nTraining AI model..." << std::endl;
        
        double initial_mse = 0.0;
        double final_mse = 0.0;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Forward pass: make predictions
            std::vector<double> predictions;
            for (double size : house_sizes) {
                predictions.push_back(predict(size));
            }
            
            // Compute MSE
            double mse = computeMSE(predictions);
            
            if (epoch == 0) {
                initial_mse = mse;
            }
            
            // Compute gradients
            double gradient_weight = 0.0;
            double gradient_bias = 0.0;
            
            for (size_t i = 0; i < house_sizes.size(); i++) {
                double error = house_prices[i] - predictions[i];
                gradient_weight += -2.0 * error * house_sizes[i];
                gradient_bias += -2.0 * error;
            }
            
            gradient_weight /= house_sizes.size();
            gradient_bias /= house_sizes.size();
            
            // Update parameters using gradient descent
            weight = weight - learning_rate * gradient_weight;
            bias = bias - learning_rate * gradient_bias;
            
            // Print progress every 20 epochs
            if (epoch % 20 == 0 || epoch == epochs - 1) {
                std::cout << "Epoch " << std::setw(3) << epoch 
                         << ": MSE = " << std::fixed << std::setprecision(4) << mse
                         << ", Weight = " << std::setprecision(4) << weight
                         << ", Bias = " << std::setprecision(4) << bias << std::endl;
            }
            
            final_mse = mse;
        }
        
        std::cout << "\n=== AI Model Trained Successfully ===" << std::endl;
        std::cout << "Final Parameters: Weight = " << std::setprecision(4) << weight 
                  << ", Bias = " << std::setprecision(4) << bias << std::endl;
        std::cout << "Training Improvement: MSE " << std::setprecision(4) << initial_mse 
                  << " → " << final_mse << std::endl;
        std::cout << "Improvement Ratio: " << std::setprecision(2) << (initial_mse/final_mse) 
                  << "x better" << std::endl;
    }
    
    int makePrediction(double new_house_size) {
        double ai_prediction = predict(new_house_size);
        
        std::cout << "\n=== AI Prediction ===" << std::endl;
        std::cout << "New house size: " << new_house_size << "00 sq ft" << std::endl;
        std::cout << "AI predicted price: $" << std::setprecision(1) << ai_prediction 
                  << "0k" << std::endl;
        
        return static_cast<int>(ai_prediction);
    }
    
    void evaluatePerformance() {
        // Make final predictions
        std::vector<double> final_predictions;
        for (double size : house_sizes) {
            final_predictions.push_back(predict(size));
        }
        
        // Calculate R-squared
        double mean_actual = 0.0;
        for (double price : house_prices) {
            mean_actual += price;
        }
        mean_actual /= house_prices.size();
        
        double ss_res = 0.0;
        double ss_tot = 0.0;
        for (size_t i = 0; i < house_prices.size(); i++) {
            double error = house_prices[i] - final_predictions[i];
            ss_res += error * error;
            ss_tot += (house_prices[i] - mean_actual) * (house_prices[i] - mean_actual);
        }
        
        double r_squared = 1.0 - (ss_res / ss_tot);
        
        // Calculate accuracy (predictions within 10% of actual)
        int accurate_predictions = 0;
        double total_abs_error = 0.0;
        
        for (size_t i = 0; i < house_prices.size(); i++) {
            double percentage_error = std::abs((house_prices[i] - final_predictions[i]) / house_prices[i]) * 100;
            if (percentage_error < 10.0) {
                accurate_predictions++;
            }
            total_abs_error += std::abs(house_prices[i] - final_predictions[i]);
        }
        
        double accuracy = (static_cast<double>(accurate_predictions) / house_prices.size()) * 100;
        double mean_abs_error = total_abs_error / house_prices.size();
        
        std::cout << "\n=== Model Performance Metrics ===" << std::endl;
        std::cout << "R-squared: " << std::setprecision(4) << r_squared 
                  << " (" << std::setprecision(1) << (r_squared*100) << "% variance explained)" << std::endl;
        std::cout << "Training Accuracy: " << std::setprecision(1) << accuracy 
                  << "% (predictions within 10%)" << std::endl;
        std::cout << "Mean Absolute Error: " << std::setprecision(2) << mean_abs_error << std::endl;
    }
};

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create and train the AI
    AeroAI ai;
    ai.train(100);
    
    // Evaluate performance
    ai.evaluatePerformance();
    
    // Make prediction on new house (2400 sq ft = 24 in our units)
    int result = ai.makePrediction(24.0);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "\n=== Performance ===" << std::endl;
    std::cout << "C++ AI execution time: " << duration.count() << " microseconds" << std::endl;
    
    std::cout << "\n=== Historic Achievement ===" << std::endl;
    std::cout << "This C++ implementation serves as reference for" << std::endl;
    std::cout << "the first AI ever written in Aero programming language!" << std::endl;
    
    std::cout << "\nFinal AI prediction result: " << result << std::endl;
    
    return result % 256;  // Limit to valid exit code range
}


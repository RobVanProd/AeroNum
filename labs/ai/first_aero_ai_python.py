#!/usr/bin/env python3
"""
Python Implementation: Linear Regression with Gradient Descent
Equivalent to the first AI written in Aero programming language
This serves as a reference implementation for comparison

Algorithm: Linear Regression with Gradient Descent
Problem: Predict house prices based on size (supervised learning)
Method: Minimize mean squared error using gradient descent optimization
"""

import numpy as np
import time
import sys

def main():
    print("=== Python AI: Linear Regression with Gradient Descent ===")
    print("Equivalent implementation to the first Aero AI")
    print("=" * 60)
    
    start_time = time.perf_counter()
    
    # ============================================================================
    # TRAINING DATA: House Size (sq ft) -> Price ($1000s)
    # ============================================================================
    # Same dataset as Aero implementation
    house_sizes = np.array([10, 12, 15, 18, 20, 22, 25, 28, 30, 32])  # hundreds of sq ft
    house_prices = np.array([15, 18, 22, 26, 30, 33, 38, 42, 45, 48])  # $10k units
    
    print(f"Training Data: {len(house_sizes)} houses")
    print(f"Size range: {house_sizes.min()}-{house_sizes.max()} (hundreds sq ft)")
    print(f"Price range: ${house_prices.min()}-{house_prices.max()}0k")
    
    # ============================================================================
    # MACHINE LEARNING MODEL PARAMETERS
    # ============================================================================
    # Linear model: price = weight * size + bias
    weight = 1.0          # Initial slope
    bias = 0.0            # Initial y-intercept
    learning_rate = 0.001 # Learning rate for gradient descent
    epochs = 100          # Number of training iterations
    
    print(f"\nInitial Parameters:")
    print(f"Weight: {weight}, Bias: {bias}")
    print(f"Learning Rate: {learning_rate}, Epochs: {epochs}")
    
    # ============================================================================
    # GRADIENT DESCENT TRAINING ALGORITHM
    # ============================================================================
    print(f"\nTraining AI model...")
    
    # Track training progress
    initial_mse = None
    final_mse = None
    
    for epoch in range(epochs):
        # Forward pass: make predictions
        predictions = weight * house_sizes + bias
        
        # Compute errors
        errors = house_prices - predictions
        
        # Compute mean squared error
        mse = np.mean(errors ** 2)
        
        if epoch == 0:
            initial_mse = mse
        
        # Compute gradients
        gradient_weight = -2 * np.mean(errors * house_sizes)
        gradient_bias = -2 * np.mean(errors)
        
        # Update parameters using gradient descent
        weight = weight - learning_rate * gradient_weight
        bias = bias - learning_rate * gradient_bias
        
        # Print progress every 20 epochs
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: MSE = {mse:.4f}, Weight = {weight:.4f}, Bias = {bias:.4f}")
    
    final_mse = mse
    
    # ============================================================================
    # AI MODEL INFERENCE: Making Predictions on New Data
    # ============================================================================
    print(f"\n=== AI Model Trained Successfully ===")
    print(f"Final Parameters: Weight = {weight:.4f}, Bias = {bias:.4f}")
    print(f"Training Improvement: MSE {initial_mse:.4f} → {final_mse:.4f}")
    print(f"Improvement Ratio: {initial_mse/final_mse:.2f}x better")
    
    # Make prediction on new house (2400 sq ft = 24 in our units)
    new_house_size = 24
    ai_prediction = weight * new_house_size + bias
    
    print(f"\n=== AI Prediction ===")
    print(f"New house size: {new_house_size}00 sq ft")
    print(f"AI predicted price: ${ai_prediction:.1f}0k")
    
    # ============================================================================
    # MACHINE LEARNING PERFORMANCE METRICS
    # ============================================================================
    # Calculate R-squared (coefficient of determination)
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((house_prices - np.mean(house_prices)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Calculate training accuracy (percentage of predictions within 10% of actual)
    final_predictions = weight * house_sizes + bias
    percentage_errors = np.abs((house_prices - final_predictions) / house_prices) * 100
    accuracy = np.mean(percentage_errors < 10) * 100
    
    print(f"\n=== Model Performance Metrics ===")
    print(f"R-squared: {r_squared:.4f} ({r_squared*100:.1f}% variance explained)")
    print(f"Training Accuracy: {accuracy:.1f}% (predictions within 10%)")
    print(f"Mean Absolute Error: {np.mean(np.abs(errors)):.2f}")
    
    end_time = time.perf_counter()
    execution_time = (end_time - start_time) * 1_000_000  # Convert to microseconds
    
    print(f"\n=== Performance ===")
    print(f"Python AI execution time: {execution_time:.1f} microseconds")
    
    # ============================================================================
    # HISTORIC ACHIEVEMENT MARKER
    # ============================================================================
    print(f"\n=== Historic Achievement ===")
    print(f"This Python implementation serves as reference for")
    print(f"the first AI ever written in Aero programming language!")
    
    # Return prediction as integer (matching Aero's return type)
    return int(ai_prediction)

if __name__ == "__main__":
    result = main()
    print(f"\nFinal AI prediction result: {result}")
    sys.exit(result % 256)  # Limit to valid exit code range


#!/bin/bash

# Run Aggressive Evolutionary Improvements
# This script demonstrates different aggressive configurations

echo "🚀 AGGRESSIVE EVOLUTIONARY IMPROVEMENTS EXAMPLES"
echo "=============================================="
echo ""

# Export environment variables
export $(cat .env.local | grep -v '^#' | xargs)
export PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai

# Example 1: Moderate Aggressive (good for testing)
echo "1️⃣ MODERATE AGGRESSIVE (5 minutes, 50 population)"
echo "Population: 50, Elite: 5, Mutation: 0.5"
echo "Expected: 5x more exploration than default"
echo ""
echo "Command: python3 run_evolutionary_aggressive.py --minutes 5 --population 50 --elite 5 --mutation 0.5"
echo ""

# Example 2: Standard Aggressive (recommended)
echo "2️⃣ STANDARD AGGRESSIVE (10 minutes, 75 population)"
echo "Population: 75, Elite: 8, Mutation: 0.65"
echo "Expected: 7.5x more exploration, 2x more mutations"
echo ""
echo "Command: python3 run_evolutionary_aggressive.py --minutes 10 --population 75 --elite 8 --mutation 0.65"
echo ""

# Example 3: Very Aggressive (maximum exploration)
echo "3️⃣ VERY AGGRESSIVE (15 minutes, 100 population)"
echo "Population: 100, Elite: 10, Mutation: 0.8"
echo "Expected: 10x more exploration, very high variation"
echo ""
echo "Command: python3 run_evolutionary_aggressive.py --minutes 15 --population 100 --elite 10 --mutation 0.8"
echo ""

# Example 4: Custom Configuration
echo "4️⃣ CUSTOM CONFIGURATION"
echo "You can mix and match parameters:"
echo "  --population: 10-200 (more = broader search)"
echo "  --elite: 3-20 (more = preserve best solutions)"
echo "  --mutation: 0.1-0.9 (higher = more variation)"
echo "  --minutes: 1-60 (longer = more generations)"
echo ""

echo "=============================================="
echo ""

# Ask user which to run
echo "Which configuration would you like to run?"
echo "1) Moderate Aggressive (5 min)"
echo "2) Standard Aggressive (10 min) - RECOMMENDED"
echo "3) Very Aggressive (15 min)"
echo "4) Exit"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "Running Moderate Aggressive configuration..."
        python3 run_evolutionary_aggressive.py --minutes 5 --population 50 --elite 5 --mutation 0.5
        ;;
    2)
        echo "Running Standard Aggressive configuration..."
        python3 run_evolutionary_aggressive.py --minutes 10 --population 75 --elite 8 --mutation 0.65
        ;;
    3)
        echo "Running Very Aggressive configuration..."
        python3 run_evolutionary_aggressive.py --minutes 15 --population 100 --elite 10 --mutation 0.8
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac
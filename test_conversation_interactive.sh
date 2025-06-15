#!/bin/bash
# Test interactive conversation with cwmai

# Create a test input file
cat > test_input.txt << 'EOF'
Hello what can you do for me?
Show me the system status
exit
EOF

# Run cwmai with the test input
echo "Testing CWMAI Conversational AI..."
echo "================================="
python3 cwmai --no-banner < test_input.txt

# Clean up
rm test_input.txt
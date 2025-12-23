#!/bin/bash

# Fix script for VectorDBBench Redis mod# Step 6: Install additional dependencies for dataset conversion
echo -e "${YELLOW}Step 6: Installing dataset conversion dependencies...${NC}"
pip install h5py pandas pyarrow numpy

# Step 7: Verify the installation
echo -e "${YELLOW}Step 7: Verifying installation...${NC}"ror
# Error: No module named 'redis.commands.search.indexDefinition'

echo "Fixing VectorDBBench Redis dependencies..."
echo "========================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Setting up Python virtual environment for VectorDBBench${NC}"
echo ""

# Define virtual environment path
VENV_DIR="venv"

# Step 1: Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Step 1: Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Failed to create virtual environment${NC}"
        echo "Please ensure python3-venv is installed: sudo apt install python3-venv"
        exit 1
    fi
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Step 2: Upgrade pip
echo -e "${YELLOW}Step 2: Upgrading pip...${NC}"
pip install --upgrade pip

# Step 3: Reinstall vectordb-bench with redis support
echo -e "${YELLOW}Step 5: Installing VectorDBBench with MemoryDB support...${NC}"
pip install vectordb-bench[memorydb]

# Step 4: Install additional dependencies for dataset conversion
echo -e "${YELLOW}Step 6: Installing dataset conversion dependencies...${NC}"
pip install h5py pandas pyarrow numpy

# # Step 5 Verify the installation
# echo -e "${YELLOW}Step 6: Verifying installation...${NC}"
# python -c "
# import sys
# try:
#     from redis.commands.search.indexDefinition import IndexDefinition
#     print('✓ RediSearch module imports successfully')
# except ImportError as e:
#     print(f'✗ Still missing: {e}')
#     sys.exit(1)

# try:
#     import vectordb_bench
#     print('✓ VectorDBBench imports successfully')
# except ImportError as e:
#     print(f'✗ VectorDBBench import failed: {e}')
#     sys.exit(1)
# "

# if [ $? -eq 0 ]; then
#     echo -e "${GREEN}✓ Installation successful!${NC}"
#     echo ""
#     echo "You can now run the benchmark:"
#     echo "  ./run-5m-benchmark.sh run"
# else
#     echo -e "${RED}✗ Installation failed. Trying alternative fix...${NC}"
    
#     # Alternative fix: Install specific versions
#     echo -e "${YELLOW}Installing specific compatible versions...${NC}"
#     pip install "redis>=5.0.0" "redis-py-cluster>=2.1.0"
    
#     # If still failing, try with redisearch
#     echo -e "${YELLOW}Installing redisearch package...${NC}"
#     pip install redisearch
    
#     # Final verification
#     python -c "
# import sys
# try:
#     from redis.commands.search import Search
#     from redis import Redis
#     print('✓ Redis Search components available')
# except ImportError as e:
#     print(f'Warning: Some components missing but may still work: {e}')
# "
# fi

echo ""
echo -e "${GREEN}Dependency fix complete!${NC}"
echo ""
echo "Next steps:"
echo "download datasets:"
echo "   ./prep_datasets/dataset.sh get cohere-large-10m"
echo ""
echo "When done, deactivate the virtual environment:"
echo "   deactivate"

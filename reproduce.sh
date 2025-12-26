#!/bin/bash
set -e

echo "=========================================="
echo "Starting ML Pipeline Reproduction"
echo "=========================================="

echo "Step 1: Installing dependencies..."
pixi install

echo "Step 2: Pulling data from DVC remote..."
.pixi/envs/default/python.exe -m dvc pull

echo "Step 3: Running DVC pipeline..."
.pixi/envs/default/python.exe -m dvc repro

echo "Step 4: Showing results..."
echo ""
echo "Metrics:"
.pixi/envs/default/python.exe -m dvc metrics show
echo ""

echo "=========================================="
echo "Pipeline reproduction completed!"
echo "=========================================="

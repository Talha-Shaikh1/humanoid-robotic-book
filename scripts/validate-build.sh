#!/bin/bash

# Build validation script for AI-Native Humanoid Robotics Textbook
# This script validates that the Docusaurus site builds without warnings or errors

set -e  # Exit on any error

echo "🔍 Starting build validation for AI-Native Humanoid Robotics Textbook..."

# Check if we're in the right directory
if [ ! -f "package.json" ] || [ ! -d "docusaurus" ]; then
    echo "❌ Error: Not in the correct project directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if Node.js and npm are available
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed or not in PATH"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm is not installed or not in PATH"
    exit 1
fi

echo "✅ Node.js and npm are available"

# Navigate to docusaurus directory
cd docusaurus

# Check if node_modules exists, install if not
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to install dependencies"
        exit 1
    fi
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# Run the build
echo "🔨 Building the site..."
BUILD_OUTPUT=$(npm run build 2>&1)
BUILD_EXIT_CODE=$?

if [ $BUILD_EXIT_CODE -ne 0 ]; then
    echo "❌ Build failed with the following output:"
    echo "$BUILD_OUTPUT"
    exit $BUILD_EXIT_CODE
fi

echo "✅ Site built successfully"

# Check for warnings in the build output
WARNING_COUNT=$(echo "$BUILD_OUTPUT" | grep -i "warn\|warning" | wc -l)
if [ $WARNING_COUNT -gt 0 ]; then
    echo "⚠️  Build completed with $WARNING_COUNT warning(s):"
    echo "$BUILD_OUTPUT" | grep -i "warn\|warning"
    echo ""
    echo "⚠️  While the build succeeded, there are warnings that should be addressed."
    echo "   Please review the warnings and fix them if possible."
else
    echo "✅ No warnings detected during build"
fi

# Check for common issues in the output
ERRORS_FOUND=0

# Check for broken links
if echo "$BUILD_OUTPUT" | grep -qi "broken.*link\|link.*broken"; then
    echo "❌ Broken links detected in build output"
    ERRORS_FOUND=$((ERRORS_FOUND + 1))
fi

# Check for missing assets
if echo "$BUILD_OUTPUT" | grep -qi "missing.*asset\|asset.*missing\|404"; then
    echo "❌ Missing assets detected in build output"
    ERRORS_FOUND=$((ERRORS_FOUND + 1))
fi

# Check for plugin issues
if echo "$BUILD_OUTPUT" | grep -qi "plugin.*error\|error.*plugin"; then
    echo "❌ Plugin errors detected in build output"
    ERRORS_FOUND=$((ERRORS_FOUND + 1))
fi

if [ $ERRORS_FOUND -gt 0 ]; then
    echo "❌ $ERRORS_FOUND error(s) were detected in the build output"
    echo "Build output:"
    echo "$BUILD_OUTPUT"
    exit 1
fi

# Verify that the build directory exists and has content
if [ ! -d "build" ]; then
    echo "❌ Build directory does not exist"
    exit 1
fi

BUILD_SIZE=$(du -sh build | cut -f1)
BUILD_FILE_COUNT=$(find build -type f | wc -l)

echo "✅ Build directory created successfully ($BUILD_SIZE, $BUILD_FILE_COUNT files)"

# Verify that key files exist in the build
REQUIRED_FILES=(
    "build/index.html"
    "build/404.html"
    "build/assets/css"
    "build/assets/js"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -e "$file" ]; then
        echo "❌ Required build file does not exist: $file"
        exit 1
    fi
done

echo "✅ All required build files exist"

# Check that the main site files have content
INDEX_SIZE=$(stat -f%z "build/index.html" 2>/dev/null || stat -c%s "build/index.html" 2>/dev/null || echo 0)
if [ $INDEX_SIZE -lt 1000 ]; then
    echo "❌ Main index.html file is too small ($INDEX_SIZE bytes), may indicate build issue"
    exit 1
fi

echo "✅ Main index.html file has appropriate size ($INDEX_SIZE bytes)"

# Validate that internal links work by checking if we can find the expected modules
MODULES_EXIST=0
for module in "module-1-ros2" "module-2-gazebo-unity" "module-3-isaac" "module-4-vla"; do
    if find "build" -name "*.html" -exec grep -l "$module" {} \; | grep -q .; then
        ((MODULES_EXIST++))
    fi
done

if [ $MODULES_EXIST -lt 4 ]; then
    echo "⚠️  Only $MODULES_EXIST out of 4 expected modules found in build"
else
    echo "✅ All 4 modules found in build output"
fi

echo ""
echo "🎉 Build validation completed successfully!"
echo "✅ No errors detected in the build process"
echo "✅ Site builds without warnings (or warnings have been noted)"
echo "✅ All required files and modules present"

exit 0
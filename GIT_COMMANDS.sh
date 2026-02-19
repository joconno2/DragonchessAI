#!/bin/bash
# Commands to commit and push the cleaned repository

echo "Step 1: Stage all changes"
git add -A

echo "Step 2: Verify what will be committed"
git status

echo "Step 3: Create commit (review COMMIT_MESSAGE.txt first)"
echo "Run: git commit -F COMMIT_MESSAGE.txt"

echo "Step 4: Tag the release"
echo "Run: git tag -a v2.0.0 -m 'Version 2.0.0: Complete C++ rewrite with plugin system'"

echo "Step 5: Push to GitHub"
echo "Run: git push origin main"
echo "Run: git push origin v2.0.0"

echo ""
echo "Optional: Create GitHub Release"
echo "1. Go to https://github.com/joconno2/DragonchessAI/releases/new"
echo "2. Select tag v2.0.0"
echo "3. Copy content from CHANGELOG.md for release notes"
echo "4. Attach pre-built binaries from dist/ directory"
echo ""
echo "Repository is ready for academic publication and classroom use."

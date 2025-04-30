#!/bin/bash

# Name und Zielverzeichnis
ZIP_URL="http://modelnet.cs.princeton.edu/ModelNet40.zip"
ZIP_NAME="ModelNet40.zip"
TARGET_DIR="ModelNet40"

echo "🔽 Lade ModelNet40 OFF-Daten herunter..."
wget --no-check-certificate $ZIP_URL -O $ZIP_NAME

echo "📦 Entpacke nach $TARGET_DIR ..."
unzip $ZIP_NAME -d .

echo "🧹 Entferne ZIP-Datei..."
rm $ZIP_NAME

echo "✅ Fertig! Die .off-Dateien findest du in: $TARGET_DIR"

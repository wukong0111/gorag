#!/bin/bash

# Directorio donde guardar la documentación
DOCS_DIR="docs"
mkdir -p $DOCS_DIR

# Extraer documentación con gomarkdoc para cada paquete estándar
for pkg in $(go list std); do
    echo "📄 Generando documentación para $pkg..."
    
    # Reemplazar "/" por "_" para nombres de archivo válidos
    file_name=$(echo "$pkg" | tr '/' '_').md
    
    # Guardar la documentación en Markdown
    gomarkdoc "$pkg" > "$DOCS_DIR/$file_name"
done

echo "✅ Documentación de la librería estándar extraída con gomarkdoc."

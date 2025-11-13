"""
Simple script to start the FastAPI server for Dual-Task ABSA
"""

import sys
import os

# Add parent directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Change to parent directory (VisoBERT-MTL) for proper imports
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    import uvicorn
    from backend.api import app
    
    print("=" * 80)
    print("Starting Dual-Task ABSA API Server")
    print("=" * 80)
    print("\nAPI will be available at:")
    print("  - Local: http://localhost:8000")
    print("  - Docs:  http://localhost:8000/docs")
    print("  - Health: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80)
    
    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )


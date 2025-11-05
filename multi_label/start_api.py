"""
Simple script to start the FastAPI server
"""

import sys
import os

# Add parent directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Change to multi_label directory for relative imports
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import uvicorn
    from api import app
    
    print("=" * 80)
    print("Starting Multi-Label ABSA API Server")
    print("=" * 80)
    print("\nAPI will be available at:")
    print("  - Local: http://localhost:8000")
    print("  - Docs:  http://localhost:8000/docs")
    print("  - Health: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


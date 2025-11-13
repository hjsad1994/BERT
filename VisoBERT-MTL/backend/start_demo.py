"""
Simple script to start the Gradio demo for Dual-Task ABSA
"""

import sys
import os

# Add parent directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Change to parent directory (VisoBERT-MTL) for proper imports
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    from backend.demo_gradio import create_interface
    
    print("=" * 80)
    print("Starting Dual-Task ABSA Gradio Demo")
    print("=" * 80)
    print("\nDemo will be available at:")
    print("  - Local: http://localhost:7860")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80)
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


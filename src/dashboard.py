"""
Streamlit Dashboard Entry Point
Simple script to run the RAG evaluation dashboard
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from monitoring.dashboard import main

if __name__ == "__main__":
    main() 
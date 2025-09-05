"""
Script to run the Streamlit dashboard.
"""

import streamlit as st
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from quant_lib.dashboard.app import main

if __name__ == "__main__":
    main()

import sys
import os

# Add ByteTrack root to sys.path
bytetrack_path = os.path.join(os.path.dirname(__name__), "src/external/ByteTrack")
sys.path.append(bytetrack_path)
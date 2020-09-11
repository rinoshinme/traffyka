"""
test simple mot with opencv tracker
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from vortex.detector.vehicle_detector import VehicleDetector
from vortex.tracker.simple_mot_tracker import SimpleMotTracker


class SimpleMotTrackerTester(object):
    def __init__(self):
        pass

    def test_video(self, video_path):
        pass

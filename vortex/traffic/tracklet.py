"""
tracklet: information history of a tracker
"""


class Tracklet(object):
    def __init__(self):
        self.history = []

    def update(self, bbox):
        self.history.append(bbox)
    
    def get_path(self):
        pass

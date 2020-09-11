import time


class ObjectPath(object):
    def __init__(self):
        self.id = 0
        self.path = []
    
    def add_position(self, bbox, label):
        cur_time = time.time()
        self.path.append((bbox, label, cur_time))

    def get_curve(self):
        return [v[0] for v in self.path]

    def get_top_label(self):
        pass
    
    def is_stopped(self):
        pass

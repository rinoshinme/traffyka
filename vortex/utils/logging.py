"""
"""

class Logger(object):
    def __init__(self, file_path, logger_type):
        self.logger_type = logger_type
        self.file_path = file_path
    
    def info(self, msg):
        pass

    def error(self, msg):
        pass

    def warning(self, msg):
        pass

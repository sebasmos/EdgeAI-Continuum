from dataclay import DataClayObject
from pandas import DataFrame

class PersistentDF(DataClayObject):
    content: DataFrame

    def __init__(self, content=None):
        if content is not None:
            self.content = content

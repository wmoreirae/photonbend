class CoordinatesOutOfImage(Exception):
    def __init__(self, latitude, longitude, message):
        self.latitude = latitude
        self.longitude = longitude
        self.message = message

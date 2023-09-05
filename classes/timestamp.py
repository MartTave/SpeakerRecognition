class Timestamp:
    @staticmethod
    def toString(time):
        timestamp = time
        milliseconds = timestamp % 1000
        seconds = (timestamp - milliseconds) / 1000
        seconds = int(seconds % 60)
        minutes = int((timestamp - milliseconds - seconds * 1000) / 1000 / 60)
        return str(minutes) + ":" + str(seconds) + "." + str(int(milliseconds))
    


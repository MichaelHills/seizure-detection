from datetime import datetime

def unix_time(dt):
    epoch = datetime.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()


def unix_time_millis(dt):
    return int(unix_time(dt) * 1000.0)


def get_millis():
    return unix_time_millis(datetime.now())


def get_seconds():
    return get_millis() / 1000.0

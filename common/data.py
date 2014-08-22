import io
import os
import os.path


def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass

class jsdict(dict):
    def __init__(self, data):
        self.__dict__ = data


class CachedDataLoader:

    def __init__(self, dir):
        self.dir = dir
        makedirs(dir)

    # try to load data from filename, if it doesn't exist then run the func()
    # and save the data to filename
    def load(self, filename, func):
        def wrap_data(data):
            if isinstance(data, list):
                return [jsdict(x) for x in data]
            else:
                return jsdict(data)

        if filename is not None:
            filename = os.path.join(self.dir, filename)
            data = io.load_hkl_file(filename)
            if data is not None:
                return wrap_data(data)

            data = io.load_pickle_file(filename)
            if data is not None:
                return wrap_data(data)

        data = func()

        if filename is not None:
            if io.save_hkl_file(filename, data):
                return wrap_data(data)

            io.save_pickle_file(filename, data)
        return wrap_data(data)

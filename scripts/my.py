print("functions.py loaded")


# The function takes path and export all CSV in DataFrame format, start determines prefix of the name of the files, end determine suffix of the name.
# The funtion returns [names of files] and [files]
def all_import(path, end=0, start=0):
    import os
    import pandas as pd

    if start == 0 and end == 0:
        files = [file for file in os.listdir(path)]
    elif start != 0 and end != 0:
        files = [
            file
            for file in os.listdir(path)
            if file.startswith(start) and file.endswith(end)
        ]
    elif start == 0 and end != 0:
        files = [file for file in os.listdir(path) if file.endswith(end)]
    elif start != 0 and end == 0:
        files = [file for file in os.listdir(path) if file.startswith(start)]

    name_set = []
    data_set = []

    for file in files:
        name_set.append(file)

        file = pd.read_csv(path + file)
        data_set.append(file)

    return name_set, data_set

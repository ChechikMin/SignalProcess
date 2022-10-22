def get_float_from_file(path):
    x_list, y_list = [], []
    separator = ','
    with open(path, 'r') as f:
        lines = [line.split() for line in f]
        for line in lines:
            x_value, y_value = line[0].split(separator)[0], line[0].split(separator)[1]
            x_list.append(float(x_value))
            y_list.append(float(y_value))
    return x_list, y_list
x,y = get_float_from_file("/Users/angelinarezcova/Downloads/Telegram Desktop/Filtred.txt")
print(x)
# This script is used to combine two different symbolization symbols into one symbol.
# Technically creating a word.
import sys
import os

def check_cmd_arguments(arg, default, false_value):
    arg_value = false_value # Setting the argument's value to the false value
    # Checking if argument is in the sys args
    if arg in sys.argv:
        index = sys.argv.index(arg) + 1 # Grab the index of the value for arg
        if index >= len(sys.argv):
        # If the value isn't passed, set it to the default value
            arg_value = default
        else:
            # We check that the value isn't another argument
            value = sys.argv[index]
            if "-" not in value:
                arg_value = value # Assign the value
            else:
                arg_value = default # else we use use the default value

    return arg_value

def get_symbols(key, file_name):
    """
    Retrieves the values from a file formatted in the following way

    key:value1,value2,value3,... etc

    Parameters
        - key (string): The key to look for
        - file_name (string): the name of the file to look for the key in
    Returns
        A list of values
    """
    file = open(file_name, 'r')
    for line in file.readlines():
        line = line.rstrip()
        if line.startswith(key) or (key in line.split(":")[0]):
            values = line.split(":")[1].split(",")
            return values


def get_values_from_file(files):
    """
    Takes a list of files and returns a tuple of the left and right symbols
    within each file as a list of lists per file

    Parameters
        - files (list<string>) : a list of file names of strings
    Returns
        A tuple<list<list<string>> where each list within the second level will
        be organized according to the files list
    """
    symbols_left = []
    symbols_right = []
    labels = []
    frames = []
    for file in files:
        frame = get_symbols("frame", file)
        left = get_symbols("left", file)
        right = get_symbols("right", file)
        label = get_symbols("label", file)
        frames.append(frame)
        symbols_left.append(left)
        symbols_right.append(right)
        labels.append(label)
    return frames, symbols_left, symbols_right, labels

def combine_elements(first, second):
    new_string = []
    for left, right in zip(first, second):
        new_string.append(left+right)

    return new_string

def combine_symbols(symbols_left, symbols_right):
    length = min(len(symbols_left[0]), len(symbols_left[1]),len(symbols_right[0]), len(symbols_right[1]))
    combined_left = combine_elements(symbols_left[0][0:length], symbols_left[1][0:length])
    combined_right = combine_elements(symbols_right[0][0:length], symbols_right[1][0:length])
    return combined_left, combined_right, length

def main():
    help = """Help output for combining different symbol types
    Usage: python3 combine_symbols.py -hs <hand_shape_symbols_file> -ls <location_symbol_file> -o <nameofoutputfile>
    """
    handshape_file = check_cmd_arguments("-hs", False, False)
    location_file = check_cmd_arguments("-ls", False, False)
    file = check_cmd_arguments("-o", "data.txt", "data.txt")

    if handshape_file == False or location_file == False:
        print(help)
        exit(1)

    frame, symbols_left, symbols_right, labels = get_values_from_file([location_file, handshape_file])
    combined_left, combined_right, length = combine_symbols(symbols_left, symbols_right)
    #print(combined_left, combined_right)
    symbol_file = open(f"./{file}.txt", 'w')
    symbol_file.write(f"frame:{','.join(frame[0][0:length])}\nleft:{','.join(combined_left)}\nright:{','.join(combined_right)}\nlabel:{','.join(labels[0][0:length])}")
    symbol_file.close()


if __name__ == "__main__":
    main()

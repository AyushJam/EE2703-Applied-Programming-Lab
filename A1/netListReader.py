from sys import argv, exit

if len(argv) != 2:
    print('\nUsage: %s <inputfile>' % argv[0])
    exit()

try:
    with open(argv[1]) as f:
        lines = f.readlines()

    for line in lines:
        print(line)
except IOError:
    print("IOError!")



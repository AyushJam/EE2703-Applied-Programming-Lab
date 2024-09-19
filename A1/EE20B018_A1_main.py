"""
        EE2703 Applied Programming Lab - 2019
            Assignment 1: EE20B018
        Student Name: Ayush Mukund Jamdar
"""

from sys import argv, exit

CIRCUIT = '.circuit'
END = '.end'
ELEMENTS = {"R": "Resistor", "L": "Inductor", "C": "Capacitor",
            "V": "Ind Voltage Source", "I": "Ind Current Source",
            "E": "VCVS", "G": "VCCS", "H": "CCVS", "F": "CCCS"}
"""
It's a good practice to check if the user has given required and only the required inputs
Otherwise, show them the expected usage.
"""
if len(argv) != 2:
    print('\nUsage: %s <inputfile>' % argv[0])
    exit()

"""
The user might input a wrong file name by mistake.
In this case, the open function will throw an IOError.
This exception (error during execution) is taken care of using try-except
"""

try:
    with open(argv[1]) as f:
        lines = f.readlines()
        start = -1
        end = -2
        for line in lines:  # extracting circuit definition start and end lines
            if CIRCUIT == line[:len(CIRCUIT)]:
                start = lines.index(line)  # start index
            elif END == line[:len(END)]:
                end = lines.index(line)   # end index
                break
        if start >= end:  # validating circuit block
            print('Invalid circuit definition')
            exit(0)

        # the above code will help extract the part of code between '.circuit' and '.end'

        l = [' '.join(reversed(line.split("#")[0].split())) for line in lines[start+1:end]]

        """
        After lines[start+1:end], the token lines are obtained
        the above line of code does the following
        1. Extract the element, node and value details after removing the comment
        2. Join the element, node and value together in a string
        3. Make a list of such strings
        
        """

        for line in reversed(l):
            print(line)  # print output in the required manner

        """
        Token Analysis
        the list l has tokens in the format 
        "value n2 n1 R..."
        """
        print("\nToken Analysis:")

        for line in l:
            line = line.split()
            print(line)
            try:
                # using the predefined Dictionary ELEMENTS
                print("Element Type: {}".format(ELEMENTS[(line[-1][0])]))
                # working with controlled sources
                if line[-1][0] == ('E' or 'G'):
                    print("Controlling Voltage nodes: {}, {}".format(line[-4], line[-5]))
                elif line[-1][0] == ('H' or 'F'):
                    print("V Source through which Controlling Current goes: {}".format(line[-4]))

            except KeyError:
                # if element name is not in ELEMENTS
                print("Invalid Element Type!")
            print("From node: {}".format(line[-2]))
            print("To node: {}".format(line[-3]))
            print("Value: {}\n".format(line[0]))

# the below block runs if an error occured while opening the file
except IOError:
    print('Invalid file')
    exit()





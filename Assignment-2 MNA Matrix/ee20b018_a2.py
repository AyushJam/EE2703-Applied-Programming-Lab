"""
        EE2703 Applied Programming Lab - 2019
            Assignment 2: EE20B018
            Spice 2: Solving Circuits
        Student Name: Ayush Mukund Jamdar

        Certain important guidelines for netlists (non-adherence of which, might give unexpected results)
        1. Frequency for AC ckts should be given in Hz
        2. Source voltage/current "Amplitude" should be given in the netlist
        3. Phase should be a real number in radians. Must not include 'pi' or other characters.
        4. Direction of current through Voltage Sources is assumed to be 'coming out of the
           positive terminal of the V source' by default. This will be required when netlist comtains
           CCCS, CCVS, VCVS and VS. The corresponding +/- signs will be observed in the output
        5. The nodes should be given as 'n0', 'n1', 'n2' and so on, or consecutive integers starting from 0 for GND.
           'GND' being an allowed exception.
        6. The node name 'n0' is specially reserved for Ground.
        7. The nodes names must have consecutive integers.
"""

import numpy as np
import math
from sys import argv, exit

CIRCUIT = '.circuit'
END = '.end'
ELEMENTS = {"R": "Resistor", "L": "Inductor", "C": "Capacitor",
            "V": "Ind Voltage Source", "I": "Ind Current Source",
            "E": "VCVS", "G": "VCCS", "H": "CCVS", "F": "CCCS"}
DIGITS_AFTER_DECIMAL = 6  # to change the precision of answers

"""
check if the user has given required and only the required inputs
Otherwise, show them the expected usage.
"""
if len(argv) != 2:
    print('\nUsage: %s <inputfile>' % argv[0])
    exit()


# I will create classes for elements for a better handling

class resistor:
    def __init__(self, name, toNode, fromNode, value):
        self.name = name
        if toNode[0] != 'n':  # so that node names are stored as 'n0', 'n1'...
            toNode = 'n' + toNode
        if fromNode[0] != 'n':
            fromNode = 'n' + fromNode
        self.toNode = toNode
        self.fromNode = fromNode
        self.value = float(value)


class inductor:
    def __init__(self, name, toNode, fromNode, value):
        self.name = name
        if toNode[0] != 'n':
            toNode = 'n' + toNode
        if fromNode[0] != 'n':
            fromNode = 'n' + fromNode
        self.toNode = toNode
        self.fromNode = fromNode
        self.impedance = complex(0, (float(value) * acfrequency * 2 * math.pi))


class capacitor:
    def __init__(self, name, toNode, fromNode, value):
        self.name = name
        if toNode[0] != 'n':
            toNode = 'n' + toNode
        if fromNode[0] != 'n':
            fromNode = 'n' + fromNode
        self.toNode = toNode
        self.fromNode = fromNode
        self.impedance = complex(0, (1 / (float(value) * acfrequency * 2 * math.pi)))


class voltage_source:
    def __init__(self, name, toNode, fromNode, type, value, phase, frequency):
        self.name = name
        if toNode[0] != 'n':
            toNode = 'n' + toNode
        if fromNode[0] != 'n':
            fromNode = 'n' + fromNode
        self.toNode = toNode  # the positive terminal
        self.fromNode = fromNode  # the negative terminal
        self.type = type
        self.value = float(value)  # amplitude
        self.freq = frequency * 2 * math.pi
        self.phase = float(phase)  # in radians
        self.current_through_vs = 0


class current_source:
    def __init__(self, name, fromNode, toNode, type, value, phase, frequency):
        self.name = name
        if toNode[0] != 'n':
            toNode = 'n' + toNode
        if fromNode[0] != 'n':
            fromNode = 'n' + fromNode
        self.toNode = toNode
        self.fromNode = fromNode
        self.type = type
        self.freq = frequency * 2 * math.pi
        self.phase = float(phase)  # in radians
        self.value = float(value)  # amplitude


class vcvs:
    def __init__(self, name, toNode, fromNode, ctrlVolPlusNode, ctrlVolMinNode, value):
        self.name = name
        if toNode[0] != 'n':
            toNode = 'n' + toNode
        if fromNode[0] != 'n':
            fromNode = 'n' + fromNode
        if ctrlVolMinNode[0] != 'n':
            ctrlVolMinNode = 'n' + ctrlVolMinNode
        if ctrlVolPlusNode[0] != 'n':
            ctrlVolPlusNode = 'n' + ctrlVolPlusNode
        self.toNode = toNode
        self.fromNode = fromNode
        self.ctrlVolPlusNode = ctrlVolPlusNode
        self.ctrlVolMinNode = ctrlVolMinNode
        self.value = float(value)  # gain


class vccs:
    def __init__(self, name, fromNode, toNode, ctrlVolPlusNode, ctrlVolMinNode, value):
        if toNode[0] != 'n':
            toNode = 'n' + toNode
        if fromNode[0] != 'n':
            fromNode = 'n' + fromNode
        if ctrlVolMinNode[0] != 'n':
            ctrlVolMinNode = 'n' + ctrlVolMinNode
        if ctrlVolPlusNode[0] != 'n':
            ctrlVolPlusNode = 'n' + ctrlVolPlusNode
        self.fromNode = fromNode
        self.toNode = toNode
        self.name = name
        self.ctrlVolPlusNode = ctrlVolPlusNode
        self.ctrlVolMinNode = ctrlVolMinNode
        self.value = float(value)  # transconductance


class ccvs:
    def __init__(self, name, toNode, fromNode, currentThroughWhichVolSource, value):
        self.name = name
        if toNode[0] != 'n':
            toNode = 'n' + toNode
        elif fromNode[0] != 'n':
            fromNode = 'n' + fromNode
        self.toNode = toNode
        self.fromNode = fromNode
        self.currentThroughWhichVolSource = currentThroughWhichVolSource
        self.value = float(value)  # Transresistance


class cccs:
    def __init__(self, name, fromNode, toNode, currentThroughWhichVolSource, value):
        if toNode[0] != 'n':
            toNode = 'n' + toNode
        elif fromNode[0] != 'n':
            fromNode = 'n' + fromNode
        self.fromNode = fromNode
        self.toNode = toNode
        self.name = name
        self.currentThroughWhichVolSource = currentThroughWhichVolSource  # name of the vs
        self.value = float(value)  # transconductance


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
                end = lines.index(line)  # end index
                break

        # if AC ckt I need its frequency, the default value os set to zero (dc)
        acfrequency = 0
        try:
            if (lines[end + 1].split())[0] == '.ac':
                acfrequency = float((lines[end + 1].split())[-1])
        except IndexError:
            pass

        if start >= end:  # validating circuit block
            print('Invalid circuit definition')
            exit(0)

        # the above code will help extract the part of code between '.circuit' and '.end'

        l = [' '.join(reversed(line.split("#")[0].split())) for line in lines[start + 1:end]]

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
        # I also need a lists of each element type in the netlist
        resistors = []
        inductors = []
        capacitors = []
        voltage_sources = []
        current_sources = []
        vol_ctrl_vol_sources = []
        vol_ctrl_cur_sources = []
        cur_ctrl_vol_sources = []
        cur_ctrl_cur_sources = []

        number_of_nodes = 1  # initialising

        print("\nToken Analysis:")

        # the below loop analyses tokens and creates objects for each element
        for line in l:
            line = line.split()
            print(line)
            try:
                # if there is a GND node, I will note it as 'n0'
                if line[-3] == 'GND':
                    line[-3] = 'n0'
                elif line[-2] == 'GND':
                    line[-2] = 'n0'

                if len(line) > 4:
                    if line[-4] == 'GND':
                        line[-4] = 'n0'
                    elif line[-5] == 'GND':
                        line[-5] = 'n0'

                # instantiation for each element
                if ELEMENTS[(line[-1][0])] == "Resistor":
                    r = resistor(line[-1], (line[-3]), line[-2], float(line[0]))
                    resistors.append(r)
                    print("Element Type: {}".format(ELEMENTS[(line[-1][0])]))
                    print("Element name: {}".format(r.name))
                    print("From node: {}".format(r.fromNode))
                    print("To node: {}".format(r.toNode))
                    print("Value: {}\n".format(r.value))

                elif ELEMENTS[line[-1][0]] == "Inductor":
                    l = inductor(line[-1], (line[-2]), line[-3], (line[-4]))
                    inductors.append(l)
                    print("Element Type: {}".format(ELEMENTS[(line[-1][0])]))
                    print("Element name: {}".format(l.name))
                    print("From node: {}".format(l.fromNode))
                    print("To node: {}".format(l.toNode))
                    print("Impedance: {}\n".format(l.impedance))

                elif ELEMENTS[line[-1][0]] == "Capacitor":
                    c = capacitor(line[-1], (line[-2]), line[-3], (line[-4]))
                    capacitors.append(c)
                    print("Element Type: {}".format(ELEMENTS[(line[-1][0])]))
                    print("Element name: {}".format(c.name))
                    print("From node: {}".format(c.fromNode))
                    print("To node: {}".format(c.toNode))
                    print("Impedance: {}\n".format(c.impedance))

                elif ELEMENTS[(line[-1][0])] == "Ind Voltage Source":
                    if line[-4] == 'dc':
                        vs = voltage_source(line[-1], (line[-2]), line[-3], line[-4], line[-5], 0, acfrequency)
                    elif line[-4] == 'ac':
                        vs = voltage_source(line[-1], (line[-2]), line[-3], line[-4], line[-5], line[-6], acfrequency)
                    voltage_sources.append(vs)
                    print("Element Type: {}".format(ELEMENTS[(line[-1][0])]))
                    print("Element name: {}".format(vs.name))
                    print("From (-ve) node: {}".format(vs.fromNode))
                    print("To (+ve) node: {}".format(vs.toNode))
                    print("Type: {}".format(vs.type))
                    print("Value/Amplitude: {}".format(vs.value))
                    print("Phase in radians (for AC): {}".format(vs.phase))
                    if acfrequency:
                        print("Frequency: {}\n".format(vs.freq))

                elif ELEMENTS[(line[-1][0])] == "Ind Current Source":
                    if line[-4] == 'dc':
                        cs = current_source(line[-1], (line[-2]), line[-3], line[-4], line[-5], 0, acfrequency)
                    elif line[-4] == 'ac':
                        cs = current_source(line[-1], (line[-2]), line[-3], line[-4], line[-5], line[-6], acfrequency)

                    current_sources.append(cs)
                    print("Element Type: {}".format(ELEMENTS[(line[-1][0])]))
                    print("Element name: {}".format(cs.name))
                    print("From node: {}".format(cs.fromNode))
                    print("To node: {}".format(cs.toNode))
                    print("Type: {}".format(cs.type))
                    print("Value/Amplitude: {}".format(cs.value))
                    print("Phase in radians (for AC): {}".format(cs.phase))
                    if acfrequency:
                        print("Frequency: {}\n".format(cs.freq))

                elif ELEMENTS[(line[-1][0])] == "VCVS":
                    VolcVolsource = vcvs(line[-1], line[-2], line[-3], line[-4], line[-5], line[-6])
                    vol_ctrl_vol_sources.append(VolcVolsource)
                    print("Element Type: {}".format(ELEMENTS[(line[-1][0])]))
                    print("Element name: {}".format(VolcVolsource.name))
                    print("From (+ve) node: {}".format(VolcVolsource.fromNode))
                    print("To (+ve) node: {}".format(VolcVolsource.toNode))
                    print("Controlling Voltage nodes: {}, {}".format(VolcVolsource.ctrlVolPlusNode,
                                                                     VolcVolsource.ctrlVolMinNode))
                    print("Value: {}\n".format(VolcVolsource.value))


                elif "CCCS" == ELEMENTS[(line[-1][0])]:
                    cccsource = cccs(line[-1], line[-2], line[-3], line[-4], line[-5])
                    cur_ctrl_cur_sources.append(cccsource)
                    print("Element Type: {}".format(ELEMENTS[(line[-1][0])]))
                    print("Element name: {}".format(cccsource.name))
                    print("From node: {}".format(cccsource.fromNode))
                    print("To node: {}".format(cccsource.toNode))
                    print("Controlling I is the I through : {}".format(cccsource.currentThroughWhichVolSource))
                    print("Value/Gain: {}\n".format(cccsource.value))

                elif ELEMENTS[(line[-1][0])] == 'CCVS':
                    ccvsource = ccvs(line[-1], line[-2], line[-3], line[-4], line[-5])
                    cur_ctrl_vol_sources.append(ccvsource)
                    print("Element Type: {}".format(ELEMENTS[(line[-1][0])]))
                    print("Element name: {}".format(ccvsource.name))
                    print("From node: {}".format(ccvsource.fromNode))
                    print("To node: {}".format(ccvsource.toNode))
                    print("Controlling I is the I through : {}".format(ccvsource.currentThroughWhichVolSource))
                    print("Value/Gain: {}\n".format(ccvsource.value))

                elif ELEMENTS[line[-1][0]] == 'VCCS':
                    vccsource = vccs(line[-1], line[-2], line[-3], line[-4], line[-5], line[-6])
                    vol_ctrl_cur_sources.append(vccsource)
                    print("Element Type: {}".format(ELEMENTS[(line[-1][0])]))
                    print("Element name: {}".format(vccsource.name))
                    print("From node: {}".format(vccsource.fromNode))
                    print("To node: {}".format(vccsource.toNode))
                    print("Controlling Voltage nodes: {}, {}".format(vccsource.ctrlVolPlusNode,
                                                                     vccsource.ctrlVolMinNode))
                    print("Value: {}\n".format(vccsource.value))

                # finding the number of nodes
                if int(line[-3][-1]) > number_of_nodes:
                    number_of_nodes = int(line[-3][-1])
                elif int(line[-2][-1]) > number_of_nodes:
                    number_of_nodes = int(line[-2][-1])

            except KeyError:
                # if element name is not in ELEMENTS
                print("KeyError! Invalid Element Type!")

        try:
            # I will need a dictionary that will assign a number to each of the node names
            # that are essentially, strings.
            number_of_nodes += 1  # count GND too
            nodes = {"n{}".format(i): i for i in range(number_of_nodes)}

            # now construct M and b matrices for Mx = b
            # Step 1: construct M first
            # Approach used: Build rach row of M, one by one, considering all elements
            # that can contribute to it, equivalently
            # This means writing each nodal equation and then adding it to M
            k = len(voltage_sources) + len(vol_ctrl_vol_sources) + len(cur_ctrl_vol_sources)
            n = len(nodes)

            M_as_a_list = [[1] + [0 for i in range(n + k - 1)]]
            # the first equation is V0 = 0 which makes the first row of M as follows
            # node rows from node 1 to node(n-1)
            for i in range(1, n):
                row = [0 for j in range(n + k)]

                # for L, C, R the stamps are similar
                for r in resistors:
                    if nodes[r.toNode] == i:
                        row[i] += 1 / r.value
                        row[nodes[r.fromNode]] += -1 / r.value
                    elif nodes[r.fromNode] == i:
                        row[i] += 1 / r.value
                        row[nodes[r.toNode]] += -1 / r.value

                for l in inductors:
                    if nodes[l.toNode] == i:
                        row[i] += 1 / l.impedance
                        row[nodes[l.fromNode]] += -1 / l.impedance
                    elif nodes[l.fromNode] == i:
                        row[i] += 1 / l.impedance
                        row[nodes[l.toNode]] += -1 / l.impedance

                for c in capacitors:
                    if nodes[c.toNode] == i:
                        row[i] += 1 / c.impedance
                        row[nodes[c.fromNode]] += -1 / c.impedance
                    elif nodes[c.fromNode] == i:
                        row[i] += 1 / c.impedance
                        row[nodes[c.toNode]] += -1 / c.impedance

                # add the contribution of each element to the nodal equation
                for vol in voltage_sources:
                    if nodes[vol.fromNode] == i:
                        row[n - 1 + voltage_sources.index(vol) + 1] += 1
                    elif nodes[vol.toNode] == i:
                        row[n - 1 + voltage_sources.index(vol) + 1] += -1

                for v in vol_ctrl_vol_sources:
                    if nodes[v.toNode] == i:
                        row[n + len(voltage_sources) + vol_ctrl_vol_sources.index(v)] = -1
                    elif nodes[v.fromNode] == i:
                        row[n + len(voltage_sources) + vol_ctrl_vol_sources.index(v)] = 1

                for v in vol_ctrl_cur_sources:
                    if nodes[v.fromNode] == i:
                        row[nodes[v.ctrlVolPlusNode]] += v.value
                        row[nodes[v.ctrlVolMinNode]] += -v.value
                    elif nodes[v.toNode] == i:
                        row[nodes[v.ctrlVolPlusNode]] += -v.value
                        row[nodes[v.ctrlVolMinNode]] += v.value

                for c in cur_ctrl_cur_sources:
                    # I need to find the Vol Source the current through which
                    # is the controlling current
                    index_of_V = 0
                    for v in voltage_sources:
                        if v.name == c.currentThroughWhichVolSource:
                            index_of_V = voltage_sources.index(v)
                    if nodes[c.fromNode] == i:
                        row[n + index_of_V] += +c.value
                    elif nodes[c.toNode] == i:
                        row[n + index_of_V] += -c.value

                for v in cur_ctrl_vol_sources:
                    if nodes[v.toNode] == i:
                        row[n + len(voltage_sources) + cur_ctrl_vol_sources.index(v)] += -1
                    elif nodes[v.fromNode] == i:
                        row[n + len(voltage_sources) + cur_ctrl_vol_sources.index(v)] = 1

                M_as_a_list.append(row)

            # for k voltage sources, next k rows of M
            # current through voltage sources
            # rows corresponding to indep vol sources
            for vol in voltage_sources:
                row = [0 for j in range(n + k)]
                row[nodes[vol.fromNode]] += -1
                row[nodes[vol.toNode]] += 1
                M_as_a_list.append(row)

            # rows corresponding to vcvs
            for v in vol_ctrl_vol_sources:
                row = [0 for j in range(n + k)]
                row[nodes[v.toNode]] += 1
                row[nodes[v.fromNode]] += -1
                row[nodes[v.ctrlVolPlusNode]] += -v.value
                row[nodes[v.ctrlVolMinNode]] += (v.value)
                M_as_a_list.append(row)

            # rows corresponding to CCVS
            for v in cur_ctrl_vol_sources:
                row = [0 for j in range(n + k)]
                row[nodes[v.toNode]] += 1
                row[nodes[v.fromNode]] += -1
                for x in voltage_sources:
                    if x.name == v.currentThroughWhichVolSource:
                        row[n + voltage_sources.index(x)] += -v.value
                M_as_a_list.append(row)

            # convert list to a matrix
            M_matrix = np.array(M_as_a_list)
            print("Matrix M:")
            print(M_matrix)

            # Step 2: Build the b matrix for Mx = b
            # these are the RHS terms of nodal equations
            b_as_a_list = [0 for j in range(n + k)]
            for i in range(1, n):
                for cursoc in current_sources:
                    if nodes[cursoc.toNode] == i:
                        b_as_a_list[i] += +cursoc.value * complex(math.cos(cursoc.phase), math.sin(cursoc.phase))
                        # value given is peak-peak
                    elif nodes[cursoc.fromNode] == i:
                        b_as_a_list[i] += -cursoc.value * complex(math.cos(cursoc.phase), math.sin(cursoc.phase))

            for i in range(1, len(voltage_sources) + 1):
                b_as_a_list[n - 1 + i] = voltage_sources[i - 1].value * complex(math.cos(voltage_sources[i - 1].phase),
                                                                                math.sin(voltage_sources[i - 1].phase))

            b_Matrix = np.array([[i] for i in b_as_a_list])
            print("\nMatrix b:")
            print(b_Matrix)

            # solve matrix equations
            x = np.linalg.solve(M_matrix, b_Matrix)

            # I will store the result in a dict result
            result = {}
            for i in range(n):
                result["Voltage at node {}".format(i)] = \
                    str(round(float(x[i].real), DIGITS_AFTER_DECIMAL)) + '+' + str(
                        round(float(x[i].imag), DIGITS_AFTER_DECIMAL)) + 'j'

            for i in range(len(voltage_sources)):
                result["Current through VS {}".format(voltage_sources[i].name)] = \
                    str(round(float(x[n + i].real), DIGITS_AFTER_DECIMAL)) + '+' + str(
                        round(float(x[n + i].imag), DIGITS_AFTER_DECIMAL)) + 'j'

            for i in range(len(vol_ctrl_vol_sources)):
                result["Current through VCVS {}".format(vol_ctrl_vol_sources[i].name)] = \
                    str(round((x[n + i + len(voltage_sources)].real), DIGITS_AFTER_DECIMAL)) + '+' \
                    + str(round(float(x[n + len(voltage_sources) + i].imag), DIGITS_AFTER_DECIMAL)) + 'j'

            for i in range(len(cur_ctrl_vol_sources)):
                result["Current through CCVS {}".format(cur_ctrl_vol_sources[i].name)] = \
                    str(round(float(x[n + len(voltage_sources) + len(vol_ctrl_vol_sources) + i].real),
                              DIGITS_AFTER_DECIMAL)) + '+' \
                    + str(round(float(x[n + len(vol_ctrl_vol_sources) + len(voltage_sources) + i].imag),
                                DIGITS_AFTER_DECIMAL)) + 'j'

            print("\nSolution x:")
            print("Node 0: GND")
            for key, value in result.items():
                if acfrequency == 0:
                    # means that it has all dc sources
                    value = value.split("+")[0]
                print(key + " : " + str(value))

        except KeyError:
            print("KeyError\nCheck character case and nodes.")

# the below block runs if an error occured while opening the file
except IOError:
    print('Invalid file')
    exit()

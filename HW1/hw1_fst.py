from fst import *

# here are some predefined character sets that might come in handy.
# you can define your own
AZ = set("abcdefghijklmnopqrstuvwxyz")
VOWS = set("aeiou")
CONS = set("bcdfghjklmnprstvwxyz")
NPTR = set("nptr")
PT = set("pt")
AO = set("ao")
E = set("e")
I = set("i")
U = set("u")
# Implement your solution here
def buildFST():
    #print "Your task is to implement a better FST in the buildFST() function, using the methods described here"
    #print "You may define additional methods in this module (hw2_fst.py) as desired"
    #
    # The states (you need to add more)
    # ---------------------------------------
    # 
    f = FST("q0") # q0 is the initial (non-accepting) state

    f.addState("q1") # a non-accepting state
    f.addState("q2") # a non-accepting state
    f.addState("q3") # a non-accepting state
    f.addState("q4") # a non-accepting state
    f.addState("q5") # a non-accepting state
    f.addState("q6") # a non-accepting state
    f.addState("q7") # a non-accepting state
    f.addState("q8") # a non-accepting state
    f.addState("q9") # a non-accepting state
    f.addState("q10") # a non-accepting state
    f.addState("q11") # a non-accepting state
    f.addState("q12") # a non-accepting state
    f.addState("q13") # a non-accepting state
    f.addState("q14") # a non-accepting state
    f.addState("q15") # a non-accepting state
    f.addState("q16") # a non-accepting state
    #f.addState("q12") # a non-accepting state
    #f.addState("q13") # a non-accepting state
    #f.addState("q14") # a non-accepting state
    f.addState("q_ing") # a non-accepting state

    f.addState("q_EOW", True) # an accepting state (you shouldn't need any additional accepting states)

    #
    # The transitions (you need to add more):
    # ---------------------------------------
    # transduce every element in this set to itself: 
    f.addSetTransition("q0", CONS, "q1")
    f.addSetTransition("q0", VOWS, "q2")
    # AZ-E =  the set AZ without the elements in the set E
    f.addSetTransition("q1", CONS, "q1")
    f.addSetTransition("q1", AO, "q2")
    f.addTransition("q1", "e", "", "q3")
    f.addTransition("q1", "i", "", "q4")
    f.addSetTransition("q1", U, "q5")
    f.addTransition("q1", "", "", "q_ing")

    f.addSetTransition("q2", CONS-NPTR, "q1")
    f.addTransition("q2", "n", "n", "q13")
    f.addTransition("q2", "p", "p", "q14")
    f.addTransition("q2", "t", "t", "q15")
    f.addTransition("q2", "r", "r", "q16")
    f.addSetTransition("q2", VOWS, "q10")
    f.addTransition("q2", "","", "q_ing")

    f.addTransition("q3", "", "e", "q6")
    f.addTransition("q3", "","", "q_ing")

    f.addTransition("q4", "", "i", "q7")
    f.addTransition("q4", "e", "", "q11")

    f.addTransition("q5", "e", "", "q8")
    f.addSetTransition("q5", CONS-NPTR, "q1")
    f.addSetTransition("q5", VOWS-E, "q10")
    f.addTransition("q5", "n", "n", "q13")
    f.addTransition("q5", "p", "p", "q14")
    f.addTransition("q5", "t", "t", "q15")
    f.addTransition("q5", "r", "r", "q16")
    f.addTransition("q5", "","", "q_ing")

    f.addSetTransition("q6", CONS-PT, "q1")
    f.addSetTransition("q6", VOWS, "q10")
    f.addTransition("q6", "p", "p", "q14")
    f.addTransition("q6", "t", "t", "q15")

    f.addSetTransition("q7", CONS-NPTR, "q1")
    f.addSetTransition("q7", VOWS-E, "q10")
    f.addTransition("q7", "n", "n", "q13")
    f.addTransition("q7", "p", "p", "q14")
    f.addTransition("q7", "t", "t", "q15")
    f.addTransition("q7", "r", "r", "q16")
    f.addTransition("q7", "","", "q_ing")

    f.addTransition("q8", "", "", "q_ing")
    f.addTransition("q8", "", "e", "q9")

    f.addSetTransition("q9", CONS, "q1")
    f.addSetTransition("q9", VOWS, "q10")

    f.addSetTransition("q10", CONS, "q1")
    f.addSetTransition("q10", VOWS, "q10")
    f.addTransition("q10", "", "", "q_ing")

    f.addTransition("q11", "", "y", "q_ing")
    f.addTransition("q11", "", "ie", "q12")

    f.addSetTransition("q12", VOWS, "q10")
    f.addSetTransition("q12", CONS, "q1")

    f.addTransition("q13", "", "n", "q_ing")
    f.addSetTransition("q13", CONS, "q1")
    f.addSetTransition("q13", AO, "q2")
    f.addTransition("q13", "e", "", "q3")
    f.addTransition("q13", "i", "", "q4")
    f.addTransition("q13", "u", "u", "q5")

    f.addTransition("q14", "", "p", "q_ing")
    f.addSetTransition("q14", CONS, "q1")
    f.addSetTransition("q14", AO, "q2")
    f.addTransition("q14", "e", "", "q3")
    f.addTransition("q14", "i", "", "q4")
    f.addTransition("q14", "u", "u", "q5")

    f.addTransition("q15", "", "t", "q_ing")
    f.addSetTransition("q15", CONS, "q1")
    f.addSetTransition("q15", AO, "q2")
    f.addTransition("q15", "e", "", "q3")
    f.addTransition("q15", "i", "", "q4")
    f.addTransition("q15", "u", "u", "q5")

    f.addTransition("q16", "", "r", "q_ing")
    f.addSetTransition("q16", CONS, "q1")
    f.addSetTransition("q16", AO, "q2")
    f.addTransition("q16", "e", "", "q3")
    f.addTransition("q16", "i", "", "q4")
    f.addTransition("q16", "u", "u", "q5")

    # map the empty string to ing: 
    f.addTransition("q_ing", "", "ing", "q_EOW")

    # Return your completed FST
    return f
    

if __name__ == "__main__":
    # Pass in the input file as an argument
    file = sys.argv[1]
    # Construct an FST for translating verb forms 
    # (Currently constructs a rudimentary, buggy FST; your task is to implement a better one.
    f = buildFST()
    # Print out the FST translations of the input file
    f.parseInputFile(file)


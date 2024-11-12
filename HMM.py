import random
import argparse
import codecs
import os
import numpy

# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}
              
        trans{    
            {'happy': {'happy': '0.5', 'grumpy': '0.1', 'hungry': '0.4'},
            {'grumpy': {'happy': '0.6', 'grumpy': '0.3', 'hungry': '0.1'},
            {'hungry': {'happy': '0.1', 'grumpy': '0.6', 'hungry': '0.3'}}"""
              
        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        emitFile = basename + ".emit"
        transFile = basename + ".trans"

        
        with open(emitFile) as f:
            for line in f:
                line = line.split()
                if line[0] not in self.emissions:
                    self.emissions[line[0]] = {}
                self.emissions[line[0]][line[1]] = float(line[2])
                
        with open(transFile) as f:
            for line in f:
                line = line.split()
                if line[0] not in self.transitions:
                    self.transitions[line[0]] = {}
                self.transitions[line[0]][line[1]] = float(line[2])
                    
        return self.transitions, self.emissions
        

   ## you do this.
    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""
        stateSeq = []
        outputSeq = []
        current_state = '#' # start state
                            # note: not displaying the initial state in the state sequence
        
        for i in range(n):
            current_state = numpy.random.choice(list(self.transitions[current_state].keys()), p = list(self.transitions[current_state].values()))
            stateSeq.append(current_state)
            
            output = numpy.random.choice(list(self.emissions[current_state].keys()), p = list(self.emissions[current_state].values()))
            outputSeq.append(output)
        
        return Sequence(stateSeq, outputSeq)

    def forward(self, sequence):
        """Implements the forward algorithm. Given a Sequence with a list of emissions,
        determine the probability of the sequence."""
        observations = sequence.outputseq
        num_states = len(self.transitions)  # Number of rows (states)
        num_observations = len(observations)  # Number of columns (time steps)

        # Initialize the forward matrix with zeros
        forwardMatrix = numpy.zeros((num_states, num_observations + 1))
        forwardMatrix[0][0] = 1  # Start state probability (index 0, column 0)

        # Convert the state dictionary keys to a list to access indices easily
        states = list(self.transitions.keys())  # List of states

        for i in range(1, num_observations + 1):  # Loop through columns
            current_observation = observations[i - 1] # Get the current observation
            for s in states:  
                if s == '#':
                    continue
                sum = 0
                for s2 in states:  # Loop through each previous state `s2`
                    if s2 == '#':
                        if i == 1:  # Only allow transitions from `#` at the first time step
                            s_idx = states.index(s)
                            s2_idx = states.index(s2)
                            sum += forwardMatrix[s2_idx][i - 1] * self.transitions[s2][s] * self.emissions[s][current_observation]
                    else:
                        s_idx = states.index(s)
                        s2_idx = states.index(s2)
                        sum += forwardMatrix[s2_idx][i - 1] * self.transitions[s2][s] * self.emissions[s][current_observation]
                forwardMatrix[s_idx][i] = round(sum, 4)
                    
        # Return the state with the highest possible value in the last column
        highest_probability = numpy.argmax(forwardMatrix[:, num_observations])
        most_probable_state = states[highest_probability]
        return most_probable_state 

    def viterbi(self, sequence):
        pass
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.

# main class
if __name__ == "__main__":
    hmm = HMM()
    hmm.load("cat")
    testSequence = hmm.generate(5)
    print("Generated sequence:", testSequence.outputseq)
    print("Most probable state:", hmm.forward(testSequence))



    
    
    
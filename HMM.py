import random
import argparse
import codecs
import os
import numpy
import argparse

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
        num_observations = len(observations)  # Number of columns (number of observations)

        # Fill the initial matrix with zeros
        forwardMatrix = numpy.zeros((num_states, num_observations + 1))
        forwardMatrix[0][0] = 1  # Set the start state probability

        # Convert the state dictionary keys to a list to access indices easily
        states = list(self.transitions.keys()) 

        for i in range(1, num_observations + 1):  # Loop through columns
            current_observation = observations[i - 1] # Get the current observation
            for s in states:
                if s == '#':
                    continue
                sum = 0
                for s2 in states:  # Loop through each previous state
                    s_idx = states.index(s)
                    s2_idx = states.index(s2) 
                    # check if [s2] is in the transitions dictionary and if the current observation is in the emissions dictionary
                    if s in self.transitions[s2] and current_observation in self.emissions[s]:
                        sum += forwardMatrix[s2_idx][i - 1] * self.transitions[s2][s] * self.emissions[s][current_observation]
                forwardMatrix[s_idx][i] = sum
                    
        # Return the state with the highest possible value in the last column
        highest_probability = numpy.argmax(forwardMatrix[:, num_observations])
        most_probable_state = states[highest_probability]
        return most_probable_state 

    def viterbi(self, sequence):
        ## You do this. Given a sequence with a list of emissions, fill in the most likely
        ## hidden states using the Viterbi algorithm.
        observations = sequence.outputseq
        num_states = len(self.transitions)
        num_observations = len(observations)
        
        matrix = numpy.zeros((num_states, num_observations + 1))
        matrix[0][0] = 1 # start state probability
        backpointers = numpy.zeros((num_states, num_observations + 1))
        
        states = list(self.transitions.keys())
        
        for i in range (1, num_observations + 1):
            current_observation = observations[i - 1]
            for s in states:
                if s == '#':
                    continue
                max_state = 0
                max_prob = 0
                for s2 in states:
                    s_idx = states.index(s)
                    s2_idx = states.index(s2)
                    if s in self.transitions[s2] and current_observation in self.emissions[s]:
                        prob = matrix[s2_idx][i - 1] * self.transitions[s2][s] * self.emissions[s][current_observation]
                        if prob > max_prob:
                            max_state = s2_idx
                            max_prob = prob
                matrix[s_idx][i] = max_prob
                backpointers[s_idx][i] = max_state
        
        most_likely_sequence = []
        most_likely_sequence.append(numpy.argmax(matrix[:, num_observations])) # get the most likely last state
        for i in range(num_observations, 0, -1): # traverse the backpointers backwards
            most_likely_sequence.append(int(backpointers[most_likely_sequence[-1]][i])) # get the most likely previous state
        
        most_likely_sequence.reverse() # reverse the list to show the most likely sequence in the correct order

        for i in range(len(most_likely_sequence)):
            most_likely_sequence[i] = states[most_likely_sequence[i]] # assign the state indices to the actual state names
        
        return most_likely_sequence
        
        

# main class
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HMM algorithms on input data.")
    parser.add_argument("basename", help="The base name for the .trans and .emit files (e.g., 'cat').")
    parser.add_argument("--forward", help="Run the forward algorithm on the given file.", type=str)
    parser.add_argument("--viterbi", help="Run the Viterbi algorithm on the given file.", type=str)
    
    args = parser.parse_args()
    
    hmm = HMM()
    hmm.load(args.basename)
    
    lander_successful_states = ['2,5', '3,4', '4,3', '4,4', '5,5']
    
    if args.forward:
        with open(args.forward) as file:
            sequences = [line.strip().split() for line in file]
            for sequence in sequences:
                if len(sequence) != 0:
                    print("Sequence: ", sequence)
                    print("Most likely final state:", hmm.forward(Sequence(list(hmm.transitions.keys()), sequence)))
                    if args.basename == "lander":
                        if hmm.forward(Sequence(list(hmm.transitions.keys()), sequence)) in lander_successful_states:
                            print("Lander successfully landed.")
                        else:
                            print("Lander crashed")
                    
    
    if args.viterbi:
        with open(args.viterbi) as file:
            sequences = [line.strip().split() for line in file]
            for sequence in sequences:
                if len(sequence) != 0:
                    print("Sequence: ", sequence)
                    print("Most likely sequence of states:", hmm.viterbi(Sequence(list(hmm.transitions.keys()), sequence)))
    
    



    
    
    
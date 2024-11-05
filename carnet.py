from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination


car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves"),
        ("KeyPresent","Starts")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

# P(starts | gas, ignition, keyPresent) = 0.99
# P(starts | gas, !ignition, keyPresent) = 0.01
# P(starts | !gas, ignition, keyPresent) = 0.01
# P(starts | gas, ignition, !keyPresent) = 0.01
# P(starts | !gas, !ignition, keyPresent) = 0.01
# P(starts | !gas, ignition, !keyPresent) = 0.01
# P(starts | gas, !ignition, !keyPresent) = 0.01 
# P(starts | !gas, !ignition, !keyPresent) = 0.01
cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[
        [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
    ],
    evidence=["Ignition", "Gas", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"], "KeyPresent":['yes','no']},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)

cpd_key_present = TabularCPD(
    variable="KeyPresent", variable_card=2, values=[[0.7], [0.3]],
    state_names={"KeyPresent":['yes','no']},
)


# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_key_present)

car_infer = VariableElimination(car_model)

# print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))

if __name__ == "__main__":
    # Given that the car will not move, what is the probability that the battery is not working?
    q1 = car_infer.query(variables=["Battery"],evidence={"Moves":"no"})
    print("Chances that the battery is not working given that the car will not move: \n", q1)

    # Given that the radio is not working, what is the probability that the car will not start?
    q2 = car_infer.query(variables=["Starts"],evidence={"Radio":"Doesn't turn on"})
    print("Chances that the car will not start given that the radio is not working: \n", q2)

    # Probability that the radio works given that the battery works (and we don't know if the car has gas in it)
    q3a = car_infer.query(variables=["Radio"],evidence={"Battery":"Works"})
    print("Chances that the radio works given that the battery is working: \n", q3a)
    
    # Probability that the radio works given that the battery is working and (we know) the car has gas in it
    q3b = car_infer.query(variables=["Radio"],evidence={"Battery":"Works","Gas":"Full"})
    print("Chances that the radio works given that the battery is working and the car has gas: \n", q3b)

    # Probability that the car doesn't start given that the car doesn't move (and we don't know if the car has gas in it)
    q4a = car_infer.query(variables=["Ignition"],evidence={"Moves":"no"})
    print("Chances that the car doesn't start given that the car doesn't move: \n", q4a)
    
    # Probability that the car doesn't start given that the car doesn't move and (we know) the car does not have gas in it
    q4b = car_infer.query(variables=["Ignition"],evidence={"Moves":"no","Gas":"Empty"})
    print("Chances that the car doesn't start given that the car doesn't move and the car has gas: \n", q4b)

    # What is the probability that the car starts if the radio works and it has gas in it?
    q5 = car_infer.query(variables=["Starts"],evidence={"Radio":"turns on","Gas":"Full"})
    print("Chances that the car starts given that the radio works and the car has gas: \n", q5)
    
    # Probability that the key is not present given that the car does not move
    q6 = car_infer.query(variables=["KeyPresent"],evidence={"Moves":"no"})
    print("Chances that the key is not present given that the car does not move: \n", q6)
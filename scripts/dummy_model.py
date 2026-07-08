from docplex.mp.model import Model
import random

def test_cplex_limits():
    # Initialize the model
    mdl = Model(name="Limit_Tester")
    
    # The Community Edition limit is exactly 1,000 for both variables and constraints.
    # We set this to 1,500 to guarantee we trigger the restriction if it exists.
    N = 1500 

    print(f"Building model with {N} variables and {N} constraints...")
    
    # Generate continuous variables
    x = mdl.continuous_var_list(N, name="x")
    
    # Add dummy constraints to inflate the model size
    for i in range(N):
        # Creates a basic chained constraint: x_i + (random_weight * x_{i+1}) <= 10
        next_idx = (i + 1) % N
        weight = random.uniform(0.5, 1.5)
        mdl.add_constraint(x[i] + weight * x[next_idx] <= 10)
        
    # Set a basic objective to maximize the sum of all variables
    mdl.maximize(mdl.sum(x))
    
    print("Model built successfully.")
    print(f"Total Variables: {mdl.number_of_variables}")
    print(f"Total Constraints: {mdl.number_of_constraints}")
    print("-" * 40)
    print("Attempting to solve...")
    
    try:
        # Attempt to solve the model
        solution = mdl.solve()
        
        if solution:
            print("\n✅ SUCCESS: Your CPLEX engine is upgraded!")
            print("The model solved successfully past the 1,000 limit threshold.")
            print(f"Objective Value: {solution.get_objective_value():.2f}")
        else:
            print("\n❌ FAILED: The solver returned no solution, but did not throw a limit error. Check your model formulation.")
            
    except Exception as e:
        print("\n❌ FAILED: Limit Error Encountered.")
        print("If you see 'CPLEX Error 1016: Community Edition', the docplex upgrade command did not link correctly.")
        print(f"Error Details: {e}")

if __name__ == "__main__":
    test_cplex_limits()
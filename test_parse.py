import pandas as pd
import re
import sys

def extract_results(input_file):
    # Read the file line by line
    with open(input_file, 'r',encoding='latin-1') as file:
        lines = file.readlines()

    # Initialize lists to store the extracted information
    graphs = []
    out_degrees = []
    action_sequences = []
    number_of_interactions = []
    rewards = []

    # Iterate through the lines and extract the relevant information
    for line in lines:
        if "GRAPH" in line:
            graph = re.findall(r"GRAPH: (.+)", line)[0]
            graphs.append(graph)
        elif "out_degree" in line:
            out_degree = re.findall(r"out_degree (.+)", line)[0]
            out_degrees.append(out_degree)
        elif "Action Sequence" in line:
            action_sequence = re.findall(r"Action Sequence \(First 10, Last 10\): (.+)", line)[0]
            action_sequences.append(action_sequence)
        elif "Number of Interaction" in line:
            interaction = int(re.findall(r"Number of Interaction: (\d+)", line)[0])
            number_of_interactions.append(interaction)
        elif "Reward" in line:
            reward = float(re.findall(r"Reward: ([\d.]+)", line)[0])
            rewards.append(reward)

    # Create a pandas DataFrame to store the results
    results_df = pd.DataFrame({
        'Graph': graphs,
        'Out_degree': out_degrees,
        'Action_Sequence': action_sequences,
        'Number_of_Interactions': number_of_interactions,
        'Reward': rewards
    })

    # Generate output file name based on input file name
    output_file = input_file.split('.')[0] + '_results.csv'

    # Save the DataFrame to a CSV file
    results_df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_results.py <input_file>")
    else:
        input_file = sys.argv[1]
        extract_results(input_file)

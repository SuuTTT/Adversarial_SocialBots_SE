#!/bin/bash

# Set the number of runs
num_runs=10

# Initialize variables for mean and variance calculations
total_reward=0
total_reward_squared=0

# Get the current timestamp
timestamp=$(date +%Y-%m-%d@%H-%M-%S)

# Create a directory based on the timestamp
results_dir="results_${timestamp}"
mkdir $results_dir

# Set the ADVBOT_LOG_FOLDER environment variable
export ADVBOT_LOG_FOLDER="/home/sudingli/ray_results/hierachical_synthetic6_${timestamp}/"

# Create a file to store the mean and variance
results_file="${results_dir}/mean_variance_results_${timestamp}.txt"
touch $results_file

for i in $(seq 1 $num_runs); do
    # Run the training command
    python ./ppo_single_large_hiar.py train > ${results_dir}/train_output_${i}_${timestamp}.out

    # Get the random name for the folder inside hierachical_synthetic6/
    folder_name=$(ls ${ADVBOT_LOG_FOLDER} | tail -n 1)

    # Run the test command
    output=$(python ./ppo_single_large_hiar.py test ${ADVBOT_LOG_FOLDER}${folder_name}/checkpoint_000200/checkpoint-200 2>&1 > ${results_dir}/test_output_${i}_${timestamp}.out)

    # Extract the mean reward and number of interactions
    mean_reward=$(echo "$output" | grep -oP '\[\K[-+]?\d*\.\d+(?=[,\]])' | head -n 1)
    num_interactions=$(echo "$output" | grep -oP '\[\K\d+(?=[,\]])' | tail -n 1)

    # Accumulate the rewards for mean and variance calculations
    total_reward=$(echo "$total_reward + $mean_reward" | bc)
    total_reward_squared=$(echo "$total_reward_squared + ($mean_reward * $mean_reward)" | bc)
done

# Calculate the mean and variance
mean=$(echo "$total_reward / $num_runs" | bc -l)
variance=$(echo "($total_reward_squared / $num_runs) - ($mean * $mean)" | bc -l)

# Save the mean and variance in the results file
echo "Mean: $mean" >> $results_file
echo "Variance: $variance" >> $results_file

# Print the mean and variance to the console
cat $results_file

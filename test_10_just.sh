#!/bin/bash

# Get the current timestamp
timestamp=$(date +%Y-%m-%d@%H-%M-%S)

# Create a directory based on the timestamp
results_dir="results_${timestamp}"
mkdir $results_dir

# Set the ADVBOT_LOG_FOLDER environment variable
export ADVBOT_LOG_FOLDER="/home/sudingli/ray_results/hierachical_synthetic6_2023-04-26@01-03-47/hierachical_synthetic6/"

# Iterate through each checkpoint folder
for folder_name in $(ls ${ADVBOT_LOG_FOLDER}); do
    # Extract the node percentage from the folder name
    node_percentage=$(echo "$folder_name" | grep -oP 'node_percentage=\K\d+\.\d+' )

    # Run the test command
    output=$(python ./ppo_single_large_hiar.py test ${ADVBOT_LOG_FOLDER}${folder_name}/checkpoint_000200/checkpoint-200 2>&1 > ${results_dir}/test_output_${folder_name}_${timestamp}.out)

    # Extract the mean reward and number of interactions
    mean_reward=$(echo "$output" | grep -oP '\[\K[-+]?\d*\.\d+(?=[,\]])' | head -n 1)
    num_interactions=$(echo "$output" | grep -oP '\[\K\d+(?=[,\]])' | tail -n 1)

    # Save the mean reward to a file for each node percentage
    echo "$mean_reward" >> "${results_dir}/rewards_${node_percentage}.txt"
done

# Calculate the mean and variance for each node percentage
for rewards_file in ${results_dir}/rewards_*.txt; do
    node_percentage=$(basename "$rewards_file" | grep -oP 'rewards_\K\d+\.\d+(?=.txt)')

    num_trials=$(wc -l < "$rewards_file")
    total_reward=$(awk '{sum+=$1} END {print sum}' "$rewards_file")
    total_reward_squared=$(awk '{sum+=$1*$1} END {print sum}' "$rewards_file")

    mean=$(echo "$total_reward / $num_trials" | awk '{print $1/$2}')
    variance=$(echo "$total_reward_squared / $num_trials - ($mean * $mean)" | awk '{print $1/$2 - ($3 * $3)}')

    # Save the mean and variance in the results file
    echo "Node Percentage: $node_percentage" >> "${results_dir}/mean_variance_results_${timestamp}.txt"
    echo "Mean: $mean" >> "${results_dir}/mean_variance_results_${timestamp}.txt"
    echo "Variance: $variance" >> "${results_dir}/mean_variance_results_${timestamp}.txt"
    echo "---------------------" >> "${results_dir}/mean_variance_results_${timestamp}.txt"
done

# Print the mean and variance to the console
cat "${results_dir}/mean_variance_results_${timestamp}.txt"

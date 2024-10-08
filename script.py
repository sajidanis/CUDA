import os
import subprocess

# Define the paths for the input and output directories
input_dir = '../graphs'  # Directory containing the .mtx files
output_dir = './results/pagerank'  # Directory to store the results

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get a list of all .mtx files in the input directory
mtx_files = [f for f in os.listdir(input_dir) if f.endswith('.mtx')]

# Iterate over each .mtx file and run the command
for mtx_file in mtx_files:
    input_file_path = os.path.join(input_dir, mtx_file)
    output_file_path = os.path.join(output_dir, f"{os.path.splitext(mtx_file)[0]}_result.txt")

    print(f"Running the graph file {mtx_file}")

    # Build the command
    command = f'./build/ParallelGraph "{input_file_path}" -1'

    try:
        # Run the command and capture the output
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # Write the output to a file in the results directory
        with open(output_file_path, 'w') as output_file:
            output_file.write(result.stdout)
        
        # Optionally, you can write the error output as well
        if result.stderr:
            with open(output_file_path, 'a') as output_file:
                output_file.write("\n\n[Error Output]\n")
                output_file.write(result.stderr)
        
        print(f"Processed {mtx_file}, results saved to {output_file_path}")

    except Exception as e:
        print(f"Failed to process {mtx_file}: {str(e)}")

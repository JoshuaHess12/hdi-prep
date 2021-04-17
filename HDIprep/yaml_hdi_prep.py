# YAML parsing and implementation of HDIprep module
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external modules
import yaml
from pathlib import Path

# Import custom modules
import hdi_prep


# Define parsing function
def RunHDIprepYAML(path_to_yaml, out_dir):
    """Parsing YAML file to feed into creation of intramodality dataset. Subsequent
    processing of files based on input parameters

    path_to_yaml: Path to .yaml file to parse that includes steps for processing
    """

    # Ensure the path is a pathlib object
    path_to_yaml = Path(path_to_yaml)

    # Open the yaml file
    with open(path_to_yaml, "r") as stream:
        # Try to load the yaml
        try:
            # Load the yaml file
            yml = yaml.full_load(stream)
            print(yml)
        # Throw exception if it fails
        except yaml.YAMLError as exc:
            # Print error
            print(exc)

    # Iterate through each step to check for Running UMAP optimal dimension
    #for s in range(len(yml["ProcessingSteps"])):

        # Get the step -- either a string (if no extra input arguments, or a dictionary with key and value)
        #step = yml["ProcessingSteps"][s]
        # Check if the step has input arguments
        #if isinstance(step, dict):
            # Get the key value
        #    step = list(yml["ProcessingSteps"][s].keys())[0]

        # Check to see if running optimal umap with different parameters
        #if step == "RunOptimalUMAP":
            # Add the output directory to the dictionary
            #yml["ProcessingSteps"][s][step]["output_dir"] = Path(out_dir)
            # Check to see if custom input
            #if not "custom_input" in yml["ProcessingSteps"][s][step]:
                # break the loop and proceed with processing
            #    break

            # Otherwise continue
            #else:
                # create a copy of the input arguments for creating dataset
                #opt_data = yml["ImportOptions"].copy()
                # Iterate through each of the import options and see if changed
                #for k in opt_data.keys():
                    # Check to see if the corresponding key exists in the optimal UMAP options
                #    if k in yml["ProcessingSteps"][s][step]:
                        # Update the dictionary
                #        opt_data[k] = yml["ProcessingSteps"][s][step][k]

                # Use the import options in the yml object to import all datasets
                #opt_set = intramodality_dataset.CreateDataset(**opt_data)

                # Get common keys between import options and run optimal umap
                # common = list(
                #    set(opt_data.keys()) & set(yml["ProcessingSteps"][s][step].keys())
                #)
                # Add the custom input option to the common keys to remove
                #common = common + ["custom_input"]

                # Remove the common keys from the processing option of optimal umap
                #for i in common:
                    # Update the dictionary
                #    yml["ProcessingSteps"][s][step].pop(i, None)

                # Apply the processing step -- Run the optimal umap embedding processing step
                # getattr(opt_set, step)(**yml["ProcessingSteps"][s][step])

                # Get the optimal dimension number
                # opt_dim = opt_set.umap_optimal_dim

                # Remove the opt_set for memory clear
                # opt_set = None

                # Copy input options for optimal umap embedding and run umap with desired number
                # umap_args = yml["ProcessingSteps"][s][step].copy()
                # Iterate through options and remove given parameters
                # for i in ["dim_range", "export_diagnostics", "output_dir", "n_jobs"]:
                    # Update the dictionary
                #    umap_args.pop(i, None)
                # Add parameters for n_components
                # umap_args.update({"n_components": opt_dim})

                # Replace the Runoptimal umap option with RunUMAP for the next step
                # yml["ProcessingSteps"][s] = {"RunUMAP": umap_args}

    # Use the import options in the yml object to import all datasets
    intramod_set = hdi_prep.CreateDataset(**yml["ImportOptions"])

    # Iterate through each step
    for s in range(len(yml["ProcessingSteps"])):
        # Get the step -- either a string (if no extra input arguments, or a dictionary with key and value)
        step = yml["ProcessingSteps"][s]

        # Check to see the type is a string (no input arguments besides function call)
        if isinstance(step, str):
            # Apply the processing step
            getattr(intramod_set, step)()

        # Check if the step has input arguments
        elif isinstance(step, dict):
            # Get the key value
            step = list(yml["ProcessingSteps"][s].keys())[0]
            # If this is a dictionary and is export nifti, add output dir
            if step == "ExportNifti1":
                # Add output
                yml["ProcessingSteps"][s][step]["output_dir"] = Path(out_dir)
            # Apply the processing step
            getattr(intramod_set, step)(**yml["ProcessingSteps"][s][step])

        # Otherwise raise an exception
        else:
            raise (Exception("Encountered an invalid processing step!"))

# YAML parsing and implementation of HDIprep module
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external modules
import yaml
from pathlib import Path

# Import custom modules
from HDIprep import hdi_prep


# Define parsing function
def RunHDIprepYAML(im, pars, out_dir):
    """Parsing YAML file to feed into creation of intramodality dataset. Subsequent
    processing of files based on input parameters

    path_to_yaml: Path to .yaml file to parse that includes steps for processing
    """

    # Ensure the path is a pathlib object
    path_to_yaml = Path(pars)

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

    # Use the import options in the yml object to import all datasets
    intramod_set = hdi_prep.CreateDataset(im,**yml["ImportOptions"])

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
            # If this is a dictionary and is export nifti, add output dir
            if (step == "RunOptimalUMAP" or step == "RunOptimalParametricUMAP"):
                # Add output
                yml["ProcessingSteps"][s][step]["output_dir"] = Path(out_dir)
            # Apply the processing step
            getattr(intramod_set, step)(**yml["ProcessingSteps"][s][step])

        # Otherwise raise an exception
        else:
            raise (Exception("Encountered an invalid processing step!"))

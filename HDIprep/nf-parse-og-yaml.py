# YAML parsing and implementation of HDIprep module
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external modules
import yaml
from pathlib import Path

# Import external modules
import argparse


# Define parsing function
def ParseYAML(path_to_yaml):
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
        # Throw exception if it fails
        except yaml.YAMLError as exc:
            # Print error
            print(exc)

    # Get the processing options
    return yml["ImportOptions"]["list_of_paths"][0]


def ParseCommandYAML():
    """Function for parsing command line arguments for input to YAML HDIprep"""

    # if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_yaml")
    args = parser.parse_args()
    # Create a dictionary object to pass to the next function
    dict = {"path_to_yaml": args.path_to_yaml}
    # Return the dictionary
    return dict


# Parse the command line arguments
args = ParseCommandYAML()

# Run the function
vals = ParseYAML(**args)
print(vals)

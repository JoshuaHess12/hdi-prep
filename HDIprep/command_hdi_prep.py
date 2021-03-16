# Command line implementation for HDIprep module using YAML files
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import custom modules
import parse_input
import yaml_hdi_prep

# Parse the command line arguments
args = parse_input.ParseCommandYAML()

# Run the MultiExtractSingleCells function
yaml_hdi_prep.RunHDIprepYAML(**args)

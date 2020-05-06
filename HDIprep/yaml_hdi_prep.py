#Morphological operation functions for HDI data preparation
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import external modules
import yaml
from pathlib import Path

#Import custom modules
import intramodality_dataset


path_to_yaml = "/Users/joshuahess/Desktop/example.yaml"

def ParseYAML(path_to_yaml):
    """Parsing YAML file to feed into creation of intramodality dataset. Subsequent
    processing of files based on input parameters

    path_to_yaml: Path to .yaml file to parse
    """

    #Ensure the path is a pathlib object
    path_to_yaml = Path(path_to_yaml)

    #Open the yaml file
    with open(path_to_yaml, 'r') as stream:
        try:
            #Load the yaml file
            yml = yaml.full_load(stream)
        except yaml.YAMLError as exc:
            #Print error
            print(exc)

    #Use the import options in the yml object to import all datasets
    intramod_set = intramodality_dataset.IntraModalityDataset(**yml["ImportOptions"])


result = getattr(test, yml["test"][0])(n_components=2)
test.umap_embeddings

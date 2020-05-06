#Functions for parsing command line arguments (YAML file) for HDIprep
#Developer: Joshua M. Hess, BSc
#Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

#Import external modules
import argparse


def ParseCommandYAML():
   """Function for parsing command line arguments for input to YAML HDIprep
   """

#if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--path_to_yaml')
   args = parser.parse_args()
   #Create a dictionary object to pass to the next function
   dict = {'path_to_yaml': args.path_to_yaml}
   #Print the dictionary object
   print(dict)
   #Return the dictionary
   return dict

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:13:37 2019

Control module. Provides a user interface for common tasks.

The primary purpose of the interface is to demonstrate how functions defined in the associated modules can be used and called in python.

@author: Peter
"""

import Image_Prep
import Matching_NN
import Norm_Calc

from random import shuffle
import os
import time
import csv
import bloscpack

def cross_validation(identifier="MP_", binaries_filepath="Binaries/", results_filepath="Results/", k_groups=5):
    
    file_list = []
    for file in os.listdir(binaries_filepath):
            if file.startswith(identifier):
                file_list.append(file[:-4])
    
    number_of_files = len(file_list)
    if number_of_files % k_groups != 0:
        # if the files do not split evenly into same sized groups then halt the process
        print("Cross validation not performed: number of files is not a multiple of the number of groups that you want to split the data into")
        return()
    
    # remove any cross validation epochs left over from previous evaluations
    for file in os.listdir(binaries_filepath):
            if file.startswith("Cross_"):
                os.remove(binaries_filepath+file)
    
    # shuffle the list of files into non-alphabetical order and bundle them into k-many groups
    shuffle(file_list)
    files_per_group = int(number_of_files / k_groups)
    nested_list = [[x, file_list[x:x+files_per_group]] for x in range(k_groups)]
    results = []
    
    # prepare a data epoch for each group - - notice that this means that groups aren't merged when training is performed, potentially reducing the quality of training
    for group in nested_list:
        Image_Prep.prepare_training_epoch(group[1], epoch_name="Cross_"+str(group[0]))
    # prepare an accuracy testing file for each group
    
    # for each group: reset the network to a blank state, train the network using k-1 epochs, test network with the k-th epoch, record results, save network weights, pause, repeat
    for group in nested_list:
        Matching_NN.revert_blank_state()
        for other_group in nested_list:
            if other_group[0] != group[0]:
                Matching_NN.train_network(num_epochs=1, identifier="Cross_"+str(other_group[0]), batch_size=1000)
        results.append( [Matching_NN.evaluate_network(identifier="Cross_"+str(group[0]))] )
        Matching_NN.save_model(save_name="Cross_"+str(group[0]), save_location="cross_models/")        
        time.sleep(60) # pause for 60 seconds
    
    print(nested_list)
    print(results)            
    # print the results to Excel
    save_name = "Cross_Validation_{}_Sets".format(k_groups) + time.strftime('_%a_%I_%M_%p_%S')
    with open(results_filepath + save_name + ".csv", 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # write groups
        datawriter.writerow(["Groups"])
        for group in nested_list:
            datawriter.writerow([group[0]]+group[1])
        datawriter.writerow(["-"])
        # write headings
        datawriter.writerow(["Group Number", "Total Examples", "Total Incorrect", "Total False Negatives", "Total True Positives"])
        # write results for each group
        for group in nested_list:
            datawriter.writerow([group[0]], results[group[0]])
    print("Stats saved as {}.csv in {}".format(save_name, results_filepath))

def evaluate_pairings(image_name, piece_height=28, piece_width=28, image_type=".jpg", results_filepath="Results/"):
    
    Image_Prep.prepare_jigsaw_epoch(image_name=image_name, piece_height=piece_height, piece_width=piece_width, image_type=image_type)
    
    Matching_NN.use_network(identifier=image_name, target_filepath="Jig_Epochs/")
    
    for file in os.listdir("Jig_Epochs/"):
        if file.startswith(image_name):
            stack = bloscpack.unpack_ndarray_from_file("Jig_Epochs/" + file)
            for thing in stack:
                compatibility = Norm_Calc.dissimilarity(concatenated_pair=thing, piece_height=piece_height, piece_width=piece_width)
            # open file
            # perform dissimilarity tests
    
        
    
    return(1)

def main():
    # this function is run if the module is opened as a standalone module
    print("Main menu. Please choose from the following options:")
    print("1) Train network")
    print("2) Evaluate network")
    print("3) Save current model")
    print("4) Load pretrained model")
    print("5) Reset to blank state model")
    print("6) Process new jigsaw")
    print("7) Convert image into a jigsaw file")
    print("8) Generate new training epoch")
    print("9) Perform cross validation")
    print("10) Quit")
    user_input = input("Enter your choice: ")
    if user_input == str(1):
        Matching_NN.train_network()
    elif user_input == str(2):
        Matching_NN.evaluate_network()
    elif user_input == str(3):
        Matching_NN.save_model()
    elif user_input == str(4):
        # Matching_NN.load_model()
        x=1
    elif user_input == str(5):
        Matching_NN.revert_blank_state()
    elif user_input == str(6):
        x=1
    elif user_input == str(7):
        x=1
    elif user_input == str(8):
        x=1
    elif user_input == str(9):
        print("This will return the network to a blank state. Please ensure that you have saved the current model as any training will be lost.")
        print("Do you wish to proceed?")
        print("1) Yes")
        print("2) No")
        new_input = input("Enter your choice: ")
        if new_input == str(1):
            cross_validation()
        else: print("Returning to main menu")
    elif user_input == str(10):
        return
    else: print("Menu option not recognised. You need to enter an integer value corresponding to the appropriate option.")
    # after completing the menu option we want to return to the main menu, notice that this doesn't happen if the user elected to quit    
    main()
### end main

if __name__ == "__main__":
    # checks if the program is being used as a standalone module (rather than being imported by another module)
    # if in standalone mode then run the main method
    main()   

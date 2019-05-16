# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 10:54:26 2019

Program to calculate various distance measures.

Calculation should be performed on the same data epoch which is used to evaluate the neural network.

@author: Peter
"""

import numpy

def dissimilarity(concatenated_pair, piece_height=28, piece_width=28):
# dissimilarity is defined as the sum of squared colour differences along the selected boundary

    # select the right hand column of A and the left hand column of B
    border_A = concatenated_pair[:,piece_width-1,:]
    border_B = concatenated_pair[:,piece_width,:]
    difference = (border_A - border_B)
    difference_squared = difference **2
    result = sum(difference_squared) **0.5
    return(result)
    
def LHalfNorm2(concatenated_pair, piece_height=28, piece_width=28):

    # select the right hand column of A and the left hand column of B
    border_A = concatenated_pair[:,piece_width-1,:]
    border_B = concatenated_pair[:,piece_width,:]
    abs_difference = numpy.absolute(border_A - border_B)
    difference_rooted = abs_difference **0.5
    result = sum(difference_rooted) **2
    return(result)

def LpqNorm(concatenated_pair, piece_height=28, piece_width=28):

    # select the right hand column of A and the left hand column of B
    border_A = concatenated_pair[:,piece_width-1,:]
    border_B = concatenated_pair[:,piece_width,:]
    
    result = 1
    return(result)

def Pomeranz_Compatibility(concatenated_pair, piece_height=28, piece_width=28):
    
    result = 1
    return(result)

def Mahalanobis_Gradient_Compatibility(concatenated_pair, piece_height=28, piece_width=28):
    
    result = 1
    return(result)


def prepare_test_epoch(list_of_files, filepath="Binaries/", epoch_name="Test", sub_epoch_half_length=40000, piece_height=28, piece_width=28):
    
    # want to prepare a data epoch from a pre-processed list of puzzle files
    # this function assumes that images have already been converted into puzzle files using the partition() method
    
    number_of_puzzles = len(list_of_files)
    # generate list of all piecewise pairings
    # num_pairings = (cols-1)*(rows-1) .... sum across all images
    total_pieces = 0
    total_pairings = 0
    full_list_non_pairs = []
    full_list_of_pairs = []
    compiled_metadata = []
    rejected_files = []
    puzzle_number = -1
    for puzzle in list_of_files:
        #with numpy.load(filepath + puzzle + ".npy") as puzzle_array:
        puzzle_array = bloscpack.unpack_ndarray_from_file(filepath + puzzle + ".blp")
        num_pieces = puzzle_array[0]
        cols, rows = puzzle_array[3], puzzle_array[4]
        if puzzle_array[1]==piece_height and puzzle_array[2]==piece_width:
            puzzle_number += 1
            total_pieces += num_pieces
            num_pairings = (cols-1)*rows + (rows-1)*cols
            total_pairings += num_pairings
            
            random_seed = random.randrange(0, 2)
            # sample all the horizontal pairs
            for j in range(rows):
                for i in range(cols-1):
                    # find a pair
                    piece_A = (j*cols)+i
                    piece_B = piece_A + 1
                    pairing = [puzzle_number, 1, piece_A, 0, piece_B, 0]
                    full_list_of_pairs.append(pairing)
                    # find a non-pair containing piece_A or piece_B
                    # want 50% of non-pairs to be pairs with non-matching edge join
                    if ((j*cols)+i+random_seed) % 2 == 0:
                        non_A = piece_A
                        non_B = piece_B
                        rotations = [random.randrange(0, 4), random.randrange(1, 4)] # non_B is always oriented incorrectly
                    # want 50% of non-pairs to be genuine non-pairs
                    else:
                        non_A = piece_A
                        non_B = sample_random_nonadjacent_piece(i,j,cols,rows)
                        rotations = [random.randrange(0, 4), 0] # no point in randomising the rotation of both pieces
                    non_pair = [puzzle_number, 0, non_A, rotations[0], non_B, rotations[1]]
                    full_list_non_pairs.append(non_pair)
            # sample all the vertical pairs
            for j in range(rows-1):
                for i in range(cols):
                    # find a pair
                    piece_A = (j*cols)+i
                    piece_B = piece_A + cols
                    pairing = [puzzle_number, 1, piece_A, 3, piece_B, 3]
                    full_list_of_pairs.append(pairing)
                    # want 50% of non-pairs to be pairs with non-matching edge join
                    if ((j*cols)+i+random_seed) % 2 == 0:
                        non_A = piece_A
                        non_B = piece_B
                        rotations = [random.randrange(0, 4), random.randrange(0, 3)] # non_B is always oriented incorrectly
                    # want 50% of non-pairs to be genuine non-pairs
                    else:
                        non_A = piece_A
                        non_B = sample_random_nonadjacent_piece(i,j,cols,rows)
                        rotations = [random.randrange(0, 4), 0] # no point in randomising the rotation of both pieces
                    non_pair = [puzzle_number, 0, non_A, rotations[0], non_B, rotations[1]]
                    full_list_non_pairs.append(non_pair)
            compiled_metadata.append([puzzle, num_pieces, num_pairings, cols, rows])
        else:
            rejected_files.append([puzzle, num_pieces, puzzle_array[1], puzzle_array[2]])
    # inform the user if any of the binary files were rejected
    if rejected_files: print("The following puzzle files did not match the height/width requirements {} x {} : ".format(piece_height, piece_width), rejected_files)
    else: print("No puzzle files rejected.")
    # 
    if compiled_metadata: print("Preparing data epoch.")
    else: print("No puzzle files were processed. A data epoch has not been generated.")
        
    """ PAIRING FORMAT: [puzzle_number, pairing=1/0 (true/false), piece_A_number, piece_A_orientation, piece_B_number, piece_B_orientation]"""
    
    # randomise list of pairs
    random.shuffle(full_list_of_pairs)
    # randomise list of non-pairs
    random.shuffle(full_list_non_pairs)
    
    if len(full_list_of_pairs) != len(full_list_non_pairs):
        print("Something has gone wrong. Have {} pairs and {} non-pairs. These two numbers should match but don't.".format(len(full_list_of_pairs), len(full_list_non_pairs)))
        return()
    
    data_length_of_piece = piece_height*piece_width*3
    for span in range(0, total_pairings, sub_epoch_half_length):
        # create a sub epoch which contains 50% pairs and 50% non-pairs
        sub_epoch = full_list_of_pairs[span:span+sub_epoch_half_length]+full_list_non_pairs[span:span+sub_epoch_half_length]
        # it is necessary to sort the sub-epoch in order to allow more efficient file opening, the sub-epoch will be reshuffled before each iteration of training
        sub_epoch.sort()
        sub_epoch_stack = []
        previous_puzzle_number = None
        for pairing in sub_epoch:
            if pairing[0] != previous_puzzle_number:
                puzzle_array = bloscpack.unpack_ndarray_from_file(filepath + compiled_metadata[pairing[0]][0] + ".blp")
                previous_puzzle_number = pairing[0]
            start_point_A = 5 + (data_length_of_piece+2)*pairing[2]
            start_point_B = 5 + (data_length_of_piece+2)*pairing[4]
            piece_A_array = puzzle_array[start_point_A:start_point_A + data_length_of_piece].reshape(piece_height, piece_width, 3)
            piece_B_array = puzzle_array[start_point_B:start_point_B + data_length_of_piece].reshape(piece_height, piece_width, 3)
            # now rotate the arrays as specified, notice that rotation is required if we want to assess vertical pairs with the same function as horizontal pairs
            piece_A_array = numpy.rot90(piece_A_array, k=pairing[3], axes=(1,0))
            piece_B_array = numpy.rot90(piece_B_array, k=pairing[5], axes=(1,0))
            # join the pieces A/B together along their right/left edge respectively
            combined_piece = numpy.concatenate((piece_A_array, piece_B_array), axis=1)
            # add the label then the combined image data
            sub_epoch_stack.append(pairing[1])
            sub_epoch_stack.extend(combined_piece.flatten().tolist())
        
        # numpy.save(binary_filepath + epoch_name + "_{}".format(math.ceil(span/sub_epoch_half_length)) + ".npy", numpy.array(sub_epoch_stack, dtype=numpy.uint8))
        bloscpack.pack_ndarray_to_file(numpy.array(sub_epoch_stack, dtype=numpy.uint8), binary_filepath + epoch_name + "_{}".format(math.ceil(span/sub_epoch_half_length)) + ".blp")  
    print("Epoch saved with filename " + epoch_name)
    
### end prepare_data_batch

def sample_random_nonadjacent_piece(i, j, cols, rows):
    
    # generate a random set of coordinates
    temp_result = ((i+random.randrange(0, cols))%cols, (j+random.randrange(0, rows))%rows)
    # check that the coordinates are not identical or adjacent to the original piece - - - imperfect test since this incorrectly rules out diagonally adjacent pieces
    while (temp_result[0] in range(i-1,i+2)) and (temp_result[1] in range(j-1,j+2)):
        temp_result = ((i+random.randrange(0, cols))%cols, (j+random.randrange(0, rows))%rows)
    result = (temp_result[1]*cols)+temp_result[0]
    return(result)

### end sample_random_nonadjacent_piece

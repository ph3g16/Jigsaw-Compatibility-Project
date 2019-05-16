# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 22:00:53 2019

Program designed to split an image into a checkerboard of sub-images.

The sub-images are stored as a list of integers in the following format:
[num_pieces, cols, rows, piece_height, piece_width, piece1 x_coord, piece1 y_coord, piece1 red_values, piece1 green_values, piece1 blue_values, piece2 x_coord, etc..]

@author: Peter
"""

import numpy
import math
import random
import bloscpack # bloscpack is faster and has much better compression than inbuilt numpy file saving options
from timeit import default_timer as timer # start = timer(), end = timer(), time_elapsed = end - start
from PIL import Image

binary_filepath = "Binaries/"
image_filepath = "Images/"
new_image_filepath = "Reconstructions/"

def partition(image_name, piece_height=28, piece_width=28, image_type=".jpg"):
    
    # open image and calculate parameters
    im = Image.open(image_filepath + image_name + image_type)
    width, height = im.size
    rows = math.floor(height/piece_height)
    cols = math.floor(width/piece_width)
    num_pieces = cols*rows
    vert_offset = math.floor( (height - (piece_height*rows)) /2)
    hori_offset = math.floor( (width - (piece_width*cols)) /2)
    image_as_array = numpy.array(im)
    image_as_puzzle = [num_pieces, piece_height, piece_width, cols, rows] # metadata, precedes the puzzle data
    
    for j in range(rows):
        for i in range(cols):
            # create jigsaw piece (i,j)
            x = hori_offset+(i*piece_width)
            y = vert_offset+(j*piece_height)
            # numpy slicing creates a new view of the array rather than copying the actual data
            # numpy indexing denoted [dimension_x, dimension_y, dimension_z] with slices denoted [start_of_slice:end_of_slice] or [start_of_slice:end_of_slice:stride_length]
            sub_image = image_as_array[y:y+piece_height, x:x+piece_width, :].flatten() # r,g,b per pixel
            ### sub_image_r = image_as_array[x:x+piece_width, y:y+piece_height, 0].flatten() # red
            ### sub_image_g = image_as_array[x:x+piece_width, y:y+piece_height, 1].flatten() # green
            ### sub_image_b = image_as_array[x:x+piece_width, y:y+piece_height, 2].flatten() # blue
            coordinate_pair = [i,j]
            sub_image_as_list = coordinate_pair + list(sub_image) # list(sub_image_r) + list(sub_image_g) + list(sub_image_b)
            # now we have a puzzle piece stored as a list of the Red,Green,Blue values with each sublist stored in row major order
            # the list is prefixed with a pair of coordinates that will allow us to reconstruct the image if desired
            
            # add the new piece to the list of puzzle pieces
            image_as_puzzle.extend(sub_image_as_list)         
    # once we have a list of i*j puzzle pieces we can save it to disk by converting it to an array
    # numpy.save(binary_filepath + image_name + ".npy", numpy.array(image_as_puzzle))
    bloscpack.pack_ndarray_to_file(numpy.array(image_as_puzzle), binary_filepath + image_name + ".blp")
    # this will create a huge file. If you want to cut it down a bit try saving the pixel values as dtype=uint8, make sure to seperate the metadata since this is bigger than uint8
    print("Puzzle-ification complete. File saved as " + image_name + "_{}x{}".format(piece_width, piece_height) + ".blp")

### end Image_Prep.partition
    
def assemble(file_name, new_coordinates=False, new_dimensions=False, save=True, show_im=True):
    
    # stack = numpy.load(new_image_filepath + file_name + ".npy").tolist()
    stack = bloscpack.unpack_ndarray_from_file(new_image_filepath + file_name + ".blp").tolist()

    # need to extract key information about the puzzle before attempting assembly
    num_pieces = stack[0]
    piece_height = stack[1]
    piece_width = stack[2]
    datapoints_per_piece = (piece_height*piece_width*3)+2 # +2 since there are 2 coordinates attached to each piece, *3 since there are 3 colour channels: R,G,B
    
    if new_dimensions:
        cols = new_dimensions[0]
        rows = new_dimensions[1]
    else:
        cols = stack[3]
        rows = stack[4]
    # initialise a target array which the puzzle data will be put into
    output = numpy.zeros((rows*piece_height, cols*piece_width, 3), dtype=numpy.uint8)
    
    for piece in range(num_pieces):
        # set a pointer to show where this particular puzzle piece is located in the list
        location_in_list = 5+(piece*datapoints_per_piece) # skip the first 5 elements as they represent jigaw metadata
        # read the coordinates, these will dictate where the piece is placed in the image
        if new_coordinates:
            i, j = new_coordinates[piece][0], new_coordinates[piece][1]
        else:
            i, j = stack[location_in_list], stack[location_in_list+1]
        # scale the coordinates according to the puzzle piece size
        i, j = i*piece_width, j*piece_height
        # read the image data for the piece
        piece_as_list = stack[location_in_list+2:location_in_list+datapoints_per_piece] # +2 since you want to skip the coordinate datapoints
        # reshape the data from a flat list back into an image array
        piece_as_array = numpy.array(piece_as_list).reshape(piece_height, piece_width, 3)
        # insert the data into the output array in the appropriate location
        output[j:j+piece_height,i:i+piece_width,:] = piece_as_array

    # convert into image format
    im = Image.fromarray(output)
    if save:
        im.save(new_image_filepath + file_name + ".png")
        #    im.save(new_image_filepath + file_name + ".jpg", format='JPEG', subsampling=0, quality=100)    # settings to save a lossless JPEG   
        print("Image saved as " + file_name + ".png in " + new_image_filepath) 

    if show_im:
        im.show() 
    
### end Image_Prep.assemble

def prepare_training_epoch(list_of_files, filepath="Binaries/", epoch_name="Training", sub_epoch_half_length=40000, piece_height=28, piece_width=28):
    
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

def prepare_jigsaw_epoch(image_name, piece_height=28, piece_width=28, image_type=".jpg", image_filepath="Image/", epoch_filepath="Jig_Epochs/", sub_epoch_length=80000):
# function assembles all possible pairings (true matches and false matches) and bundles them as a series of sub-epoch files

    # open image and calculate parameters
    im = Image.open(image_filepath + image_name + image_type)
    image_as_array = numpy.array(im)
    width, height = im.size
    rows = math.floor(height/piece_height)
    cols = math.floor(width/piece_width)
    vert_offset = math.floor( (height - (piece_height*rows)) /2)
    hori_offset = math.floor( (width - (piece_width*cols)) /2)
    counter = 0
    counter_limit = math.ceil(sub_epoch_length/16)
    sub_epoch = 0
    stack = []
    
    # cycle through all pieces in the puzzle
    for j in range(rows):
        for i in range(cols):
            # select piece_A
            x_coord = hori_offset + (i*piece_width)
            y_coord = vert_offset + (j*piece_height)
            piece_A = image_as_array[y_coord:y_coord+piece_height,x_coord:x_coord+piece_width,:]
            # for each piece_A, cycle through all piece_B's which haven't yet featured as a piece_A
            for y in range(j, rows):
                for x in range(cols):
                    if y == j and x <= i: continue # skip the first part of row j as these pieces have already been used as a piece_A
                    # select piece_B
                    x_coord = hori_offset + (x*piece_width)
                    y_coord = vert_offset + (y*piece_height)
                    piece_B = image_as_array[y_coord:y_coord+piece_height,x_coord:x_coord+piece_width,:]
                    counter += 1
                    # iterate through all possible edge joinings of A and B
                    for combination in range(16):
                        truth_value = 0
                        if combination is 0 and y == j and x+1 == i: truth_value = 1
                        if combination is 15 and x == i and y+1 == j: truth_value = 1
                        rotation_A = math.floor(combination/4)
                        rotation_B = combination % 4
                        temp_A = numpy.rot90(piece_A, k=rotation_A, axes=(1,0))
                        temp_B = numpy.rot90(piece_B, k=rotation_B, axes=(1,0))
                        pairing = numpy.concatenate((temp_A, temp_B), axis=1).flatten().tolist()
                        stack.append(truth_value)
                        stack.extend(pairing)
                    if counter > counter_limit:
                        bloscpack.pack_ndarray_to_file(numpy.array(stack, dtype=numpy.uint8), epoch_filepath + image_name + "_{}".format(math.ceil(sub_epoch)) + ".blp")
                        stack = []
                        counter = 0
                        sub_epoch += 1
    if counter > 0:
        bloscpack.pack_ndarray_to_file(numpy.array(stack, dtype=numpy.uint8), epoch_filepath + image_name + "_{}".format(math.ceil(sub_epoch)) + ".blp")
    
### end prepare_jigsaw_epoch

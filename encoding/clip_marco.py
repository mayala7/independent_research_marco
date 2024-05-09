from ntpath import join
from feature_spaces import get_story_wordseqs

import os
import sys
import numpy as np
import h5py
import argparse
import json
import pathlib
from os.path import join, dirname
import logging

from encoding_utils import *
from feature_spaces import _FEATURE_CONFIG, get_feature_space
from encoding import *
from ridge_utils.ridge import bootstrap_ridge
from config import  REPO_DIR, EM_DATA_DIR



#GETTING STIMULUS WORDS  
    #Define get_stimulus_words function 
    #Returns: Returns all the stimulus words for a subject 
    #Parameters:
        #Subject number 
    #Call get_story_wordseqs(all_stories)

def get_all_story_words(): 
	"""
	Description: Gets the word sequences for ALL stories for a particular subject
	
	Params: 
        - subject: Subject number
		
	Output: 
        - word_seq_stories: word sequences for ALL stories for particular subject
	   """
	sessions = list(map(str, [1, 2, 3, 4, 5])) #make elements into strings 
	with open(join(EM_DATA_DIR, "sess_to_story.json"), "r") as f: #opens file as f
		sess_to_story = json.load(f) #declares f as sess_to_story
	train_stories, test_stories = [], []
	for sess in sessions:
		stories, tstory = sess_to_story[sess][0], sess_to_story[sess][1]
		train_stories.extend(stories)
		if tstory not in test_stories:
			test_stories.append(tstory) #makes training and testing
	assert len(set(train_stories) & set(test_stories)) == 0, "Train - Test overlap!" #makes sure no overlap
	allstories = list(set(train_stories) | set(test_stories)) #compiles all stories
	
	down_sampled_words = get_story_wordseqs(allstories)
	
	return down_sampled_words

def save_all_story_object():
	save_location = join(REPO_DIR, "save_marco") #CHANGE THIS WHEN COLLECTING DATA
	to_save = get_all_story_words()
	np.savez("%s/words" % save_location, to_save)
	
def get_all_story_words():
	# Load the .npz file
    data = np.load('path', allow_pickle=True)

    # Access the ndarray under 'arr_0'
    arr_0_contents = data['arr_0'].item()  # Use .item() to extract the dictionary from the scalar array

    # Access a specific DataSequence object
    stagefright_sequence = arr_0_contents['stagefright']
    print(stagefright_sequence)

    # Example operation: Access the chunks of this specific sequence
    chunks = stagefright_sequence.chunks()
    data.close()    # Close the npz file after you are done

    return chunks
    

#WORD EMBEDDING HELPER FUNCTION 
    #Define get_llm_vectors 
        #Takes in word_sequence and outputs a tensor embedding. Takes in words from get_stimulus_words 
    #Parameters: stimulus_sequence (?)
    #Returns: stimulus_embedding (Batch Size, Sequence Length)

#STIMULUS EMBEDDING
    #Initialize STIMULUS tensor (n,m)
    #Iterate through each story, 
        #for each session, 
        #for each subject and call get_llm_vectors 
		

#BOLD EMBEDDING
    #Initialize BOLD tensor (n,m) 
    #Iterate through each story, 
        #for each session, 
        #for each subject by calling get_response and populate BOLD tensor
def get_bold(subject):
    zRresp = get_response(train_stories, subject)
    print("zRresp: ", zRresp.shape)
    zPresp = get_response(test_stories, subject)
    print("zPresp: ", zPresp.shape)
	
    save_location = join(REPO_DIR, "save_marco", feature, subject) #CHANGE THIS WHEN COLLECTING DATA

    print("Saving encoding model & results to:", save_location)
    os.makedirs(save_location, exist_ok=True)
	
    np.savez("%s/weights" % save_location, zRresp)
    np.savez("%s/weights" % save_location, zPresp)
	
    return (zRresp, zPresp)



#CLAFP EMBEDDING SPACE 
    #eval function -->  Load data, Load get_llm_vectors and get_response. 
                        #Calculates loss (cosine similarity) across all examples 
    #train function --> Forward. Flatten. Compute Loss (cosine similarity)
	
save_all_story_object()

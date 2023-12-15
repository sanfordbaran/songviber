# -*- coding: utf-8 -*-
"""
Created on Wednesday Dec 13, 6:24 PM 2023

@author: Sanford Baran
"""

import os
import lyricsgenius as lg
import openai
import numpy as np
import pandas as pd
import json
import math
import re
import logging


def initialize_constants():
    global MODEL
    MODEL = "gpt-3.5-turbo"
    #MODEL = "gpt-4"
    
    global INPUT_FILE_NAME
    INPUT_FILE_NAME = 'pitchfork_100_2023.xlsx'

    # Read tokens/keys from .env file
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv('.env') # read local .env file

    # Set your OPENAI key 
    openai.api_key  = os.environ['OPENAI_API_KEY'] 

    global geniusAccessToken
    geniusAccessToken = os.environ["GENIUS_ACCESS_TOKEN"]


def setup_logging():
    logging.basicConfig(
    filename='song_viber_xlsx_builder.log',  # Specify the log file name
    level=logging.INFO,         # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
    

def get_genius():
    """
    Create and return a Genius object for accessing the Genius API.

    This function initializes a Genius object using the provided Genius API access token.
    It also configures the object to remove section headers when parsing lyrics.

    Returns:
    Genius: A Genius object for interacting with the Genius API.
    """
    genius = lg.Genius(geniusAccessToken)

    # setting to remove section headers
    genius.remove_section_headers = True
    return genius


def get_nrc_vad_lex_dict():
    """
    Load and parse the NRC VAD Lexicon from a text file into a dictionary.

    This function reads the NRC VAD Lexicon from a file called 'NRC-VAD-Lexicon.txt'
    and creates a dictionary where each word is mapped to a dictionary containing
    its valence, arousal, and dominance scores.

    Returns:
    dict: A dictionary mapping words to their valence, arousal, and dominance scores.
    """
    nrc_vad_lex = {}
    with open('NRC-VAD-Lexicon.txt', 'r') as file:
        # Iterate through each line in the file
        for line in file:       
            fields = line.strip().split('\t')
            nrc_vad_lex[fields[0]] = {'valence':float(fields[1]), 'arousal':float(fields[2]), 'dominance':float(fields[3])}
            
    return nrc_vad_lex

            
def get_mood_vad_dict():
    """
    Create a mood dictionary with valence, arousal, and dominance scores based on the NRC VAD Lexicon.

    This function reads a list of moods from a file called 'moods.txt' and creates a dictionary where
    each mood is mapped to its corresponding valence, arousal, and dominance scores from the NRC VAD Lexicon.

    Returns:
    dict: A dictionary mapping moods to their valence, arousal, and dominance scores.
    """
    nrc_vad_lex = get_nrc_vad_lex_dict()
    
    mood_vad_dict = {}   
    with open('moods.txt', 'r') as file:
        for line in file:       
            mood = line.strip() 
            if mood.lower() in nrc_vad_lex:
                if mood not in mood_vad_dict:
                    mood_vad_dict[mood] = nrc_vad_lex[mood.lower()]  
    return mood_vad_dict

                    
def cleanCruftyLyrics(lyrics):
    """
    Clean up cruft and additional content from lyrics text.

    This function takes a string containing lyrics text and removes various cruft and additional content 
    that may be present, including contributor credits, ticket information, and suggestions.

    Parameters:
    lyrics (str): The lyrics text to be cleaned.

    Returns:
    str: The cleaned lyrics text with cruft removed.
    """
    cleanedLyrics = re.sub("\d+ Contributor.+Lyrics","", lyrics)   
    cleanedLyrics = re.sub("See.+tickets as low as \$\d+", "", cleanedLyrics)
    
    cleanedLyrics = re.sub("You might also like\n", "", cleanedLyrics)
    if len(re.findall("You might also like[\w\d]", cleanedLyrics)) > 0:
        cleanedLyrics = re.sub("You might also like", "", cleanedLyrics)
        
    cleanedLyrics = re.sub("\d*Embed$", "", cleanedLyrics)   
    return cleanedLyrics


def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, 
                                 max_tokens=2000):
    """
    Generate a chat completion response based on a list of conversation messages.

    This function uses the OpenAI API to generate a chat completion response based on a list
    of conversation messages. It sends the messages to the specified model and receives a response
    with the degree of randomness controlled by the 'temperature' parameter and a maximum token limit
    set by 'max_tokens'.

    Parameters:
    messages (list): A list of message objects, typically containing user and system messages.
    model (str): The name of the OpenAI model to use (default is 'gpt-3.5-turbo').
    temperature (float): The degree of randomness of the model's output (default is 0, less random).
    max_tokens (int): The maximum number of tokens the model can output (default is 2000).

    Returns:
    str: The generated completion content as a string.
    """
    logging.info("Start ChatCompletion...")
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
        max_tokens=max_tokens, # the maximum number of tokens the model can ouptut 
    )
    logging.info("ChatCompletion returned")
    
    content = response.choices[0].message["content"]
    
    return content


def get_completion_and_token_count(messages, 
                                   model="gpt-3.5-turbo",
                                   temperature=0, 
                                   max_tokens=2000):
    """
    Generate a chat completion response and provide token count information.

    This function uses the OpenAI API to generate a chat completion response based on a list
    of conversation messages. It sends the messages to the specified model and receives a response
    with the degree of randomness controlled by the 'temperature' parameter and a maximum token limit
    set by 'max_tokens'. Additionally, it provides information about the token count used for the completion.

    Parameters:
    messages (list): A list of message objects, typically containing user and system messages.
    model (str): The name of the OpenAI model to use (default is 'gpt-3.5-turbo').
    temperature (float): The degree of randomness of the model's output (default is 0, less random).
    max_tokens (int): The maximum number of tokens the model can output (default is 2000).

    Returns:
    tuple: A tuple containing two elements:
        - str: The generated completion content as a string.
        - dict: A dictionary containing token count information:
            - 'prompt_tokens': The number of tokens used for the prompt.
            - 'completion_tokens': The number of tokens used for the completion.
            - 'total_tokens': The total number of tokens used.
    """
    logging.info("Start ChatCompletion...")
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens,
    )
    logging.info("ChatCompletion returned")

    content = response.choices[0].message["content"]

    token_dict = {'prompt_tokens':response['usage']['prompt_tokens'],
                    'completion_tokens':response['usage']['completion_tokens'],
                    'total_tokens':response['usage']['total_tokens'],
    }

    return content, token_dict

    
def getLyrics(songTitle, artist = None):
    """
    Retrieve and clean the lyrics of a song from Genius.

    This function uses the Genius API to search for the lyrics of a song based on its title and optional artist.
    It then cleans up any cruft or additional content from the lyrics using the 'cleanCruftyLyrics' function.

    Parameters:
    songTitle (str): The title of the song for which you want to retrieve the lyrics.
    artist (str, optional): The artist of the song (optional).

    Returns:
    str: The cleaned lyrics of the song as a string.
    """
    genius = get_genius()
    if artist == None:
        song = genius.search_song(songTitle)
    else:
        song = genius.search_song(songTitle, artist)

    return cleanCruftyLyrics(song.lyrics)
    
    
def analyze_lyrics_using_gpt(lyrics):
    """
    Analyze song lyrics using GPT-3.5-turbo model and return results in JSON format.

    This function takes song lyrics as input, sends them to the GPT-3.5-turbo model for analysis, 
    and retrieves the results in JSON format. The lyrics should be delimited with '####'.

    Parameters:
    lyrics (str): The song lyrics to be analyzed, delimited with '####'.

    Returns:
    str: A JSON-formatted string containing the analysis results, including 'Meaning', 'Mood', 
         and 'Emotional Content' as full sentences.
    """
    delimiter = "####"
    system_message = f"""
    You will be provided with song lyrics. \
    The song lyrics will be delimited with \
    {delimiter} characters.

        
    Output the 'Meaning', 'Mood' and 'Emotional Content' of these lyrics in JSON.\
    Make sure each of these are full sentences.
    """          
    messages =  [  
    {'role':'system', 
     'content': system_message},    
    {'role':'user', 
     'content': f"{delimiter}{lyrics}{delimiter}"}  
    ]
    

    content, token_dict = get_completion_and_token_count(messages, model=MODEL)
    logging.info(token_dict)
    
    return content


def get_the_most_relevant_words_and_their_relevances(mood_and_emotion):
    """
    Extract the most relevant words and their relevances from a given mood and emotion text.

    Parameters:
    mood_and_emotion (str): A string containing mood and emotion text.

    Returns:
    str: JSON-formatted string containing a list of dictionaries with keys 'word' and 'relevance'.
    """

    delimiter = "####"
    system_message = f"""
    You will be provided with text \
    delimited with {delimiter} characters.
  
    From this text create a list of the most important words.
    Make sure this list is no longer than 8 words.
    
    For each word in this list assign a relevance \
    on a scale of 0.0 - 1.0
    
    Using this list of words and their corresponding relevances, \
    create a list of dictionaries with the keys 'word' and 'relevance'.
    Output as JSON
    """          
    messages =  [  
    {'role':'system', 
     'content': system_message},    
    {'role':'user', 
     'content': f"{delimiter}{mood_and_emotion}{delimiter}"}  
    ]
    
    content = get_completion_from_messages(messages, model=MODEL)
    return content


def get_vad_from_text(mood_and_emotion):
    """
    Calculate Valence, Arousal, and Dominance (VAD) scores based on mood and emotion text.

    This function analyzes mood and emotion text, extracts relevant words and their relevances,
    and calculates weighted VAD scores using the NRC VAD Lexicon. It returns the weighted average VAD
    scores along with a dictionary containing mood-word mappings and their weights.

    Args:
        mood_and_emotion (str): Text containing mood and emotion information.

    Returns:
        tuple: A tuple containing two elements:
            - dict: Weighted average VAD scores with keys 'valence', 'arousal', and 'dominance'.
            - dict: A dictionary containing mood-word mappings, VAD triplets, and weights with keys:
                - 'Mood': List of mood words.
                - 'VAD': List of VAD triplets.
                - 'Weight': List of relevance weights.
    """
    content = get_the_most_relevant_words_and_their_relevances(mood_and_emotion)
    if content == None:
        return None
    
    list_of_triplets = []
    relevances = []
    
    l1 = []
    l2 = []
    nrc_vad_lex = get_nrc_vad_lex_dict()
    word_relevace_pairs = json.loads(content)
    for word_rel in word_relevace_pairs:
        mood = word_rel['word']
        if mood in nrc_vad_lex:
            vad = nrc_vad_lex[mood]
            vad_values_list = list(vad.values())
            list_of_triplets.append(vad_values_list)
            
            relevance = word_rel['relevance']
            relevances.append([relevance, relevance, relevance])
            
            l1.append(mood)
            l2.append(relevance)
            
    logging.info(f'Moods: {l1}')
    logging.info(f'Weights: {l2}')
    logging.info(f'List of Triplets: {list_of_triplets}')
    
    mood_dict = {'Mood': l1, 'VAD': list_of_triplets, 'Weight': l2}
              
    data = np.array(list_of_triplets)
    weights = np.array(relevances)
    
    logging.info(data)
    logging.info(weights)
    
    weighted_avg = np.average(data, weights=weights, axis=0)  
    logging.info(weighted_avg)
    
    weighted_average_vad = {'valence': round(weighted_avg[0], 4), 'arousal': round(weighted_avg[1], 4), 'dominance': round(weighted_avg[2], 4)}
    return  (weighted_average_vad, mood_dict)


def get_euclidean_distance(delta_v, delta_a, delta_d):
    """
    Calculate the Euclidean distance in the Valence-Arousal-Dominance (VAD) space.

    This function takes the differences (deltas) in Valence (V), Arousal (A), and Dominance (D)
    and calculates the Euclidean distance in the VAD space using the formula:
    distance = sqrt(delta_v^2 + delta_a^2 + delta_d^2)

    Args:
        delta_v (float): The difference in Valence.
        delta_a (float): The difference in Arousal.
        delta_d (float): The difference in Dominance.

    Returns:
        float: The Euclidean distance in the VAD space.
    """
    return math.sqrt(delta_v**2 + delta_a**2 + delta_d**2)


def find_closest_mood(mvdict, valence, arousal, dominance):
    """
    Find the closest mood(s) in a mood-VAD dictionary to given Valence, Arousal, and Dominance (VAD) values.

    This function calculates the Euclidean distance between the given VAD values and each mood's VAD values
    stored in 'mvdict'. It returns the mood(s) with the closest VAD values and their corresponding distances.

    Args:
        mvdict (dict): A dictionary containing mood-VAD mappings with keys as mood names and values as VAD triplets.
        valence (float): The Valence value to compare.
        arousal (float): The Arousal value to compare.
        dominance (float): The Dominance value to compare.

    Returns:
        list: A list of mood names that have the closest VAD values.
    """
    min_euclidean_distance = 10.0
    mood_name = []
    for key in mvdict.keys():
        delta_v = valence - mvdict[key]['valence']
        delta_a = arousal - mvdict[key]['arousal']
        delta_d = dominance - mvdict[key]['dominance']
        distance = get_euclidean_distance(delta_v, delta_a, delta_d)
        if distance == min_euclidean_distance:
            mood_name.append(key)
        elif distance < min_euclidean_distance:
            min_euclidean_distance = distance
            mood_name = [key]
          
    logging.info(f'Closest VAD to Weighted-Average VAD: {mood_name} : {dict[mood_name[0]]}')
    logging.info(f'Distance: {min_euclidean_distance}')
    return mood_name


def get_mood_from_vad(vad, mvdict):
    """
    Determine the mood(s) based on a weighted average Valence-Arousal-Dominance (VAD) vector.

    This function takes a weighted average VAD vector ('vad') and a mood-VAD dictionary ('mvdict')
    and finds the mood(s) with the closest VAD values to the given VAD vector.

    Args:
        vad (dict): A dictionary containing Valence, Arousal, and Dominance values with keys 'valence', 'arousal', and 'dominance'.
        mvdict (dict): A dictionary containing mood-VAD mappings with keys as mood names and values as VAD triplets.

    Returns:
        list: A list of mood names that have the closest VAD values to the given VAD vector.
    """
    logging.info(f'Weighted-Average VAD: {vad}')
    return find_closest_mood(mvdict, vad['valence'], vad['arousal'], vad['dominance'])
    
    
def lyrics_step(df, row, song_title, artist = ""):
    """
    Retrieve and store song lyrics in a DataFrame.

    This function retrieves the lyrics of a song based on its title and optional artist using the 'getLyrics' function.
    It then updates the 'Lyrics' column of the provided DataFrame ('df') with the retrieved lyrics and returns them.

    Args:
        df (DataFrame): The DataFrame in which to store the lyrics.
        row: A DataFrame row to update with the lyrics.
        song_title (str): The title of the song for which you want to retrieve the lyrics.
        artist (str, optional): The artist of the song (optional).

    Returns:
        str: The retrieved lyrics as a string.
    """
    try:
        if artist == "":
            lyrics = getLyrics(song_title)
        else:
            lyrics = getLyrics(song_title, artist)
            
        df.at[row.Index, 'Lyrics'] = lyrics
        return lyrics
    
    except:
        logging.debug("Genius Couldn't Find Lyrics")
        logging.debug("====================================")
        logging.debug("")
        logging.debug("")
        return None
    
    
def analyze_lyrics_step(df, row, lyrics):
    """
    Analyze song lyrics using (GPT-3.5-turbo or GPT-4)model and update a DataFrame with the results.

    This function analyzes the provided lyrics using the 'analyze_lyrics_using_gpt' function and
    updates the specified DataFrame ('df') with the analysis results, including 'Meaning', 'Moods_Description',
    and 'Emotions_Description' columns. It also returns a string combining the 'Mood' and 'Emotional Content'.

    Args:
        df (DataFrame): The DataFrame to update with the analysis results.
        row: A DataFrame row to update with the analysis results.
        lyrics (str): The lyrics of the song to be analyzed.

    Returns:
        str: A string containing the combined 'Mood' and 'Emotional Content'.
    """
    try:   
        logging.info("Have the LLM analyze the lyrics and give its interpretation...")
        analysis = analyze_lyrics_using_gpt(lyrics)
        logging.info(analysis)
        
        anal_json = json.loads(analysis)
        mood_and_emotion = (f'{anal_json["Mood"]} {anal_json["Emotional Content"]}')
        logging.info(mood_and_emotion)
    
        df.at[row.Index, 'Meaning'] = anal_json['Meaning']
        df.at[row.Index, 'Moods_Description'] = anal_json['Mood']
        df.at[row.Index, 'Emotions_Description'] = anal_json['Emotional Content']
        return mood_and_emotion
    
    except:
        logging.debug("Exception thrown trying to analyze lyrics")
        logging.debug("====================================")
        logging.debug("")
        logging.debug("")
        return None
    

def vads_processing_step(df, row, mood_and_emotion):
    """
    Extract and process VAD information from mood and emotion analysis and update a DataFrame.

    This function extracts Valence-Arousal-Dominance (VAD) information from mood and emotion analysis
    performed using the 'mood_and_emotion' text and updates the specified DataFrame ('df') with the extracted
    VAD information, including 'Moods', 'Moods_JSON', and 'VAD_Centroid' columns. It also returns the weighted
    average VAD values.

    Args:
        df (DataFrame): The DataFrame to update with the VAD information.
        row: A DataFrame row to update with the VAD information.
        mood_and_emotion (str): The text containing mood and emotion analysis results.

    Returns:
        dict: A dictionary containing the weighted average VAD values with keys 'valence', 'arousal', and 'dominance'.
    """
    try:
        logging.info("getting vads from LLM mood and emotion analysis")
        weighted_average_vad, mood_dict = get_vad_from_text(mood_and_emotion)
        logging.info(f'Weighted-Average VAD: {weighted_average_vad}')

        df.at[row.Index, 'Moods'] = " ".join(mood_dict['Mood'])
        df.at[row.Index, 'Moods_JSON'] = json.dumps(mood_dict)
        df.at[row.Index, 'VAD_Centroid'] = json.dumps(weighted_average_vad)
        return weighted_average_vad
    except:
        logging.debug("Exception getting vads from LLM mood and emotion analysis")
        logging.debug("====================================")
        logging.debug("")
        logging.debug("")
        return None
    
    
def get_closest_overall_mood_step(df, row, weighted_average_vad, mood_vad_dict):
    """
    Determine the closest overall mood based on a weighted average Valence-Arousal-Dominance (VAD) vector.

    This function calculates the closest overall mood to the provided weighted average VAD vector
    using the 'get_mood_from_vad' function and updates the specified DataFrame ('df') with the result.

    Args:
        df (DataFrame): The DataFrame to update with the closest overall mood.
        row: A DataFrame row to update with the closest overall mood.
        weighted_average_vad (dict): A dictionary containing Valence, Arousal, and Dominance values with keys 'valence', 'arousal', and 'dominance'.
        mood_vad_dict (dict): A dictionary containing mood-VAD mappings with keys as mood names and values as VAD triplets.
    """
    try:
        logging.info("Get closest mood from weighted-average VAD")
        centroid_mood = get_mood_from_vad(weighted_average_vad, mood_vad_dict)
        #mood = get_mood_from_vad(average_vad, nrc_vad_lex)

        df.at[row.Index, 'Centroid_Mood'] = centroid_mood
        return centroid_mood
    except:
        logging.debug("Exception thrown while trying to get closest mood from weighted-average VAD")
        return None
    
       
def analyze_song(df, row, song_title, artist = ""):
    """
    Analyze a song, retrieve lyrics, perform sentiment analysis, and determine the closest overall mood.

    This function performs a series of steps to analyze a song, including retrieving its lyrics,
    performing sentiment analysis using the 'analyze_lyrics_step' function, processing Valence-Arousal-Dominance (VAD)
    information using the 'vads_processing_step' function, and determining the closest overall mood using the 'get_closest_overall_mood_step' function.

    Args:
        df (DataFrame): The DataFrame to update with analysis results.
        row: A DataFrame row representing the song to analyze.
        song_title (str): The title of the song to analyze.
        artist (str, optional): The artist of the song (optional).
    """
    print()
    print(f'Ranking:  {row.Ranking}')

    logging.info("")
    logging.info(f'Ranking:  {row.Ranking}')

    logging.info("Step1: Get Lyrics from Genius")
    lyrics = lyrics_step(df, row, song_title, artist)
    if lyrics == None:
        return -1
        
    logging.info("Step2: GPT Analyze Lyrics")
    mood_and_emotion = analyze_lyrics_step(df, row, lyrics)
    if mood_and_emotion == None:
        print("Failed Step2: GPT Analyze Lyrics")
        logging.debug("Failed Step2: GPT Analyze Lyrics")
        return -1
        
    logging.info("Step3: VADs Processing Step")
    weighted_average_vad = vads_processing_step(df, row, mood_and_emotion)
    if weighted_average_vad == None:
        print("Failed Step3: VADs Processing Step")
        logging.debug("Failed Step3: VADs Processing Step")
        return -1
        

    logging.info("Step4: Get Closest Overall Mood") 
    if weighted_average_vad != None:
        mood_vad_dict = get_mood_vad_dict() 
        centroid_mood = get_closest_overall_mood_step(df, row, weighted_average_vad, mood_vad_dict)
        if centroid_mood == None:
            print("Failed Step4: Get Closest Overall Mood")
            logging.debug("Failed Step4: Get Closest Overall Mood")
            return -1
    
    return 0

def main():
    """
    Analyze songs from an input Excel file, update the DataFrame with analysis results, and save the results to an output Excel file.

    This function reads an input Excel file containing song information, iterates through the rows to analyze each song using the 'analyze_song' function,
    and updates the DataFrame with analysis results. Finally, it saves the updated DataFrame to an output Excel file based on the selected model.

    Note: Ensure that the 'INPUT_FILE_NAME' and 'MODEL' variables are properly configured before running this function.

    Example:
        To analyze songs using the GPT-3.5-turbo model, set 'MODEL' to "gpt-3.5-turbo" and provide the input file name in 'INPUT_FILE_NAME'.
        The output file will be named based on the input file and the selected model (e.g., 'input_file_gpt35.xlsx').

    """
    initialize_constants()

    setup_logging()

    df = pd.read_excel(INPUT_FILE_NAME)
    logging.info(df.columns)

    for row in df.itertuples():
        return_code = analyze_song(df, row, row.Title, row.Artist)
        if return_code == 0:
            print("success...")
            logging.info("success...")

    file_name_prefix = INPUT_FILE_NAME.split('.')[0]

    if MODEL == "gpt-3.5-turbo":
        out_file_name = file_name_prefix + '_gpt35.xlsx'
    elif MODEL == "gpt-4":
        out_file_name = file_name_prefix + '_gpt4.xlsx'

    df.to_excel(out_file_name, index=False)
    print()
    print("FINISHED!!")

    logging.info("")
    logging.info("Finished!! ...  yea")
 
    
if __name__ == '__main__':
    main()
    
    
    
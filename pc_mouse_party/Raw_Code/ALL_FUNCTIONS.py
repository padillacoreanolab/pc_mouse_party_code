import os
import re
import operator
import traceback
import warnings
import pathlib
import h5py
import math
import numpy as np
import pandas as pd
import moviepy.editor as mpy
from moviepy.editor import *
from medpc2excel.medpc_read import medpc_read
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from collections import defaultdict


def calculate_elo_rating(
    subject_elo_rating,
    agent_elo_rating,
    k_factor=20,
    score=1,
    number_of_decimals=1
):
    """
    Calculates the Elo rating of a given subject given it's original score,
    it's opponent, the K-Factor, and whether or not it has won or not.
    The calculation is based on: https://www.omnicalculator.com/sports/elo

    Args:
        subject_elo_rating(float): The original Elo rating for the subject
        agent_elo_rating(float): The original Elo rating for the agent
        k_factor(int): k-factor, or development coefficient.
            - It usually takes values between 10 and 40,
            depending on player's strength
        score(int): the actual outcome of the game.
            - In chess, a win counts as 1 point, a draw is equal to 0.5,
            and a lose gives 0.
        number_of_decimals(int): Number of decimals to round to

    Returns:
        int: Updated Elo rating of the subject
    """
    # Calculating the Elo rating
    rating_difference = agent_elo_rating - subject_elo_rating
    expected_score = 1 / (1 + 10 ** (rating_difference / 400))
    new_elo_rating = subject_elo_rating + k_factor * (score - expected_score)
    # Rounding to `number_of_decimals`
    return round(new_elo_rating, number_of_decimals)


def update_elo_rating(
    winner_id,
    loser_id,
    id_to_elo_rating=None,
    default_elo_rating=1000,
    winner_score=1,
    loser_score=0,
    **calculate_elo_rating_params
):
    """
    Updates the Elo rating in a dictionary that contains the ID of the subject
    as keys, and the Elo rating as the values.
    You can also adjust how the Elo rating is calculated with
    'calculate_elo_rating_params'.

    Args:
        - winner_id(str): ID of the winner
        - loser_id(str): ID of the loser
        - id_to_elo_rating(dict): Dict that has the ID of the subjects
        as keys to the Elo Score as values.
        - default_elo_rating(int): The default Elo rating to be used if
        there is no elo score for the specified ID.
        - **calculate_elo_rating_params(kwargs): Other params for the
        calculate_elo_rating to change how the Elo rating is calculated

    Returns:
        - Dict: Dictionary that has the ID of the subjects as keys
        to the Elo Score as values.
    """
    if id_to_elo_rating is None:
        id_to_elo_rating = defaultdict(lambda: default_elo_rating)

    # Getting the current Elo Score
    current_winner_rating = id_to_elo_rating[winner_id]
    current_loser_rating = id_to_elo_rating[loser_id]

    # Calculating Elo rating
    id_to_elo_rating[winner_id] = calculate_elo_rating(
        subject_elo_rating=current_winner_rating,
        agent_elo_rating=current_loser_rating,
        score=winner_score,
        **calculate_elo_rating_params
        )
    id_to_elo_rating[loser_id] = calculate_elo_rating(
        subject_elo_rating=current_loser_rating,
        agent_elo_rating=current_winner_rating,
        score=loser_score,
        **calculate_elo_rating_params
        )

    return id_to_elo_rating


def get_ranking_from_elo_rating_dictionary(input_dict, subject_id):
    """
    Orders a dictionary of subject ID keys to ELO score values by ELO score.
    And then gets the rank of the subject with the inputted ID.
    Lower ranks like 1 would represent those subjects with higher
    ELO scores and vice versa.

    Args:
        input_dict(dict):
            Dictionary of subject ID keys to ELO score values
        subject_id(str, int, or any value that's a key in input dict):
            The ID of the subject that you want the ranking of

    Returns:
        int:
            Ranking of the subject with the ID inputted
    """
    # Sorting the subject ID's by ELO score
    sorted_subject_to_elo_rating = sorted(
        input_dict.items(),
        key=operator.itemgetter(1),
        reverse=True
        )
    # Getting the rank of the subject based on ELO score
    return [
        subject_tuple[0] for subject_tuple in sorted_subject_to_elo_rating
        ].index(subject_id) + 1


def iterate_elo_rating_calculation_for_dataframe(
    dataframe,
    winner_id_column,
    loser_id_column,
    tie_column=None,
    additional_columns=None
):
    """
    Iterates through a dataframe that has the ID of winners and losers
    for a given event.
    A dictionary will be created that contains the information of the event,
    which can then be turned into a dataframe.
    Each key is either from winner or loser's perspective.

    Args:
        - dataframe(Pandas DataFrame):
        - winner_id_column(str): The name of the column that has
        the winner's ID
        - loser_id_column(str): The name of the column that has the loser's ID
        - additional_columns(list): Additional columns to take from
        the dataframe.

    Returns:
        - Dict: With a key value pair for each event either from the
        winner or loser's perspective.
        This can be turned into a dataframe with each
        key value pair being a row.
    """
    if additional_columns is None:
        additional_columns = []

    # Dictionary that keeps track of the current Elo rating of the subject
    id_to_elo_rating = defaultdict(lambda: 1000)
    # Dictionary that will be converted to a DataFrame
    index_to_elo_rating_and_meta_data = defaultdict(dict)

    # Indexes that will identify which row the dictionary key
    # value pair will be
    # The number of the index has no significance other than being
    # the number of the row
    all_indexes = iter(range(0, 99999))

    # Keeping track of the number of matches
    total_match_number = 1

    # Making a copy in case there is an error with
    # changing the type of the tie column
    copied_dataframe = dataframe.copy()
    # Changing the tie column type to bool
    # So that we can filter out for booleans including False and 0
    try:
        copied_dataframe[tie_column] = copied_dataframe[
            tie_column
            ].astype(bool)
    except Exception:
        copied_dataframe = dataframe.copy()

    for i, row in copied_dataframe.dropna(subset=winner_id_column).iterrows():
        # Getting the ID of the winner subject
        winner_id = row[winner_id_column]
        # Getting the ID of the loser subject
        loser_id = row[loser_id_column]

        # Getting the current Elo Score
        current_winner_rating = id_to_elo_rating[winner_id]
        current_loser_rating = id_to_elo_rating[loser_id]

        if tie_column:
            # When there is nothing in the tie column
            # Or when there is a false value indicating that it is not a tie
            if pd.isna(copied_dataframe[tie_column][i]) or not (copied_dataframe[tie_column][i]).any():

                winner_score = 1
                loser_score = 0
            # When there is value in the tie column
            else:
                winner_score = 0.5
                loser_score = 0.5
        # When there is no tie column
        else:
            winner_score = 1
            loser_score = 0

        # Updating the dictionary with ID keys and Elo Score values
        update_elo_rating(
            winner_id=winner_id,
            loser_id=loser_id,
            id_to_elo_rating=id_to_elo_rating,
            winner_score=winner_score,
            loser_score=loser_score
            )

        # Saving all the data for the winner
        winner_index = next(all_indexes)
        index_to_elo_rating_and_meta_data[winner_index]["total_match_number"] = total_match_number
        index_to_elo_rating_and_meta_data[winner_index]["subject_id"] = winner_id
        index_to_elo_rating_and_meta_data[winner_index]["agent_id"] = loser_id
        index_to_elo_rating_and_meta_data[winner_index]["original_elo_rating"] = current_winner_rating
        index_to_elo_rating_and_meta_data[winner_index]["updated_elo_rating"] = id_to_elo_rating[winner_id]
        index_to_elo_rating_and_meta_data[winner_index]["win_draw_loss"] = winner_score
        index_to_elo_rating_and_meta_data[winner_index]["subject_ranking"] = get_ranking_from_elo_rating_dictionary(id_to_elo_rating, winner_id)
        index_to_elo_rating_and_meta_data[winner_index]["agent_ranking"] = get_ranking_from_elo_rating_dictionary(id_to_elo_rating, loser_id)
        index_to_elo_rating_and_meta_data[winner_index]["pairing_index"] = 0
        for column in additional_columns:
            index_to_elo_rating_and_meta_data[winner_index][column] = row[column]  

        # Saving all the data for the loser
        loser_index = next(all_indexes)
        index_to_elo_rating_and_meta_data[loser_index]["total_match_number"] = total_match_number
        index_to_elo_rating_and_meta_data[loser_index]["subject_id"] = loser_id
        index_to_elo_rating_and_meta_data[loser_index]["agent_id"] = winner_id
        index_to_elo_rating_and_meta_data[loser_index]["original_elo_rating"] = current_loser_rating
        index_to_elo_rating_and_meta_data[loser_index]["updated_elo_rating"] = id_to_elo_rating[loser_id]
        index_to_elo_rating_and_meta_data[loser_index]["win_draw_loss"] = loser_score
        index_to_elo_rating_and_meta_data[loser_index]["subject_ranking"] = get_ranking_from_elo_rating_dictionary(id_to_elo_rating, loser_id)
        index_to_elo_rating_and_meta_data[loser_index]["agent_ranking"] = get_ranking_from_elo_rating_dictionary(id_to_elo_rating, winner_id)
        index_to_elo_rating_and_meta_data[loser_index]["pairing_index"] = 1
        for column in additional_columns:
            index_to_elo_rating_and_meta_data[loser_index][column] = row[column]  

        # Updating the match number
        total_match_number += 1

    return index_to_elo_rating_and_meta_data


def get_first_key_from_dictionary(input_dictionary):
    """
    Gets the first key from a dictionary.
    Usually used to get the dataframe from the nested dictionary
    created by medpc2excel.medpc_read.

    Args:
        input_dictionary: dict
            - A dictionary that you want to get the first key from

    Returns:
        str (usually)
            - First key to the inputted dictionary
    """
    # Turns the dictionary keys into a list and gets the first item
    return list(input_dictionary.keys())[0]


def get_medpc_dataframe_from_medpc_read_output(medpc_read_dictionary_output):
    """
    Gets the dataframe from the output from medpc2excel.medpc_read,
    that extracts data from a MED-PC file.
    This is done by getting the values of the nested dictionaries.

    Args:
        medpc_read_dictionary_output: Nested defaultdict
            - The output from medpc2excel.medpc_read.
            This contains the dataframe extracted from MED-PC file

    Returns:
        str(usually), str(usually), Pandas DataFrame
            - The data key to the medpc2excel.medpc_read output
            - The subject key to the medpc2excel.medpc_read
            - The dataframe extracted from the MED-PC file
    """
    date = get_first_key_from_dictionary(
        input_dictionary=medpc_read_dictionary_output
        )
    subject = get_first_key_from_dictionary(
        input_dictionary=medpc_read_dictionary_output[date]
        )
    # Dataframe must use both the date and subject key
    # with the inputted dictionary
    return date, subject, medpc_read_dictionary_output[date][subject]


def get_medpc_dataframe_from_list_of_files(medpc_files, stop_with_error=False):
    """
    Gets the dataframe from the output from medpc2excel.
    medpc_read that extracts data from a MED-PC file.
    This is done with multiple files from a list.
    And the date and the subject of the recording session is extracted as well.
    The data and subject metadata are added to the dataframe.
    And then all the dataframes for all the files are combined.

    Args:
        medpc_files: list
            - List of MED-PC recording files. Can be either relative
            or absolute paths.
        stop_with_error: bool
            - Flag to terminate the program when an error is raised.
            - Sometimes MED-PC files have incorrect formatting,
            so can be skipped over.
    Returns:
        Pandas DataFrame
            - Combined MED-PC DataFrame for all the files
            with the corresponding date and subject.
    """
    # List to combine all the Data Frames at the end
    all_medpc_df = []
    for file_path in medpc_files:
        try:
            # Reading in the MED-PC log file
            ts_df, medpc_log = medpc_read(
                file=file_path,
                override=True,
                replace=False
                )
            # Extracting the corresponding MED-PC Dataframe,
            # date, and subject ID
            date, subject, medpc_df = get_medpc_dataframe_from_medpc_read_output(
                medpc_read_dictionary_output=ts_df
                )
            medpc_df["date"] = date
            medpc_df["subject"] = subject
            medpc_df["file_path"] = file_path
            all_medpc_df.append(medpc_df)
        except Exception:
            # Printing out error messages and the corresponding traceback
            print(traceback.format_exc())
            if stop_with_error:
                # Stopping the program all together
                raise ValueError("Invalid Formatting for file: {}".format(file_path))
            else:
                # Continuing with execution
                print("Invalid Formatting for file: {}".format(file_path))
    return pd.concat(all_medpc_df)


def get_all_animal_ids(animal_string):
    """
    Converts a string that contains the ID of animals, and only gets the IDs.
    This usually removes extra characters that were added.
    (i.e. "1.1 v 2.2" to ("1.1", "2.2"))

    Args:
        animal_string(str): This is the first param.

    Returns:
        tuple: Of IDs of animals as strings
    """
    # Splitting by space so that we have a list of just the words
    all_words = animal_string.split()
    # Removing all words that are not numbers
    all_numbers = [num for num in all_words if re.match(r'^-?\d+(?:\.\d+)$', num)]
    return tuple(all_numbers)


def calculate_elo_score(
    subject_elo_score,
    agent_elo_score,
    k_factor=20,
    score=1,
    number_of_decimals=1
):
    """
    Calculates the Elo score of a given subject
    given it's original score, it's opponent,
    the K-Factor, and whether or not it has won or not.
    The calculation is based on: https://www.omnicalculator.com/sports/elo

    Args:
        subject_elo_score(float): The original Elo score for the subject
        agent_elo_score(float): The original Elo score for the agent
        k_factor(int): k-factor, or development coefficient.
            - It usually takes values between 10 and 40,
            depending on player's strength
        score(int): the actual outcome of the game.
            - In chess, a win counts as 1 point, a draw is equal to 0.5,
            and a lose gives 0.
        number_of_decimals(int): Number of decimals to round to

    Returns:
        int: Updated Elo score of the subject
    """
    # Calculating the Elo score
    rating_difference = agent_elo_score - subject_elo_score
    expected_score = 1 / (1 + 10 ** (rating_difference / 400))
    new_elo_score = subject_elo_score + k_factor * (score - expected_score)
    # Rounding to `number_of_decimals`
    return round(new_elo_score, number_of_decimals)


def add_session_number_column(
    dataframe,
    indexes,
    session_number_column="session_number"
):
    """
    Add a column to Pandas DataFrame that contains the session number.
    This will only add session numbers to the rows specified by indexes.
    You can fill in the empty cells with method:
    DataFrame.fillna(method='ffill')

    Args:
        dataframe(Pandas DataFrame): The DataFrame to add the
        session number column
        indexes(list): List of indexes for which rows to
        add the session numbers
        session_number_column(str): Name of the column to add

    Returns:
        Pandas DataFrame: DataFrame with the session numbers added
    """
    copy_dataframe = dataframe.copy()
    session_number = 1
    for index in indexes:
        copy_dataframe.at[index, session_number_column] = session_number
        session_number += 1
    return copy_dataframe


def update_elo_score(
    winner_id,
    loser_id,
    id_to_elo_score=None,
    default_elo_score=1000,
    winner_score=1,
    loser_score=0,
    **calculate_elo_score_params
):
    """
    Updates the Elo score in a dictionary that contains the
    ID of the subject as keys,
    and the Elo score as the values.
    You can also adjust how the Elo score is calculated with
    'calculate_elo_score_params'.

    Args:
        winner_id(str): ID of the winner
        loser_id(str): ID of the loser
        id_to_elo_score(dict): Dict that has the ID of the subjects as
        keys to the Elo Score as values
        default_elo_score(int): The default Elo score to be used if
        there is not elo score for the specified ID
        **calculate_elo_score_params(kwargs): Other params for the
        calculate_elo_score to change how the Elo score is calculated

    Returns:
        Dict: Dict that has the ID of the subjects as
        keys to the Elo Score as values
    """
    if id_to_elo_score is None:
        id_to_elo_score = defaultdict(lambda: default_elo_score)

    # Getting the current Elo Score
    current_winner_rating = id_to_elo_score[winner_id]
    current_loser_rating = id_to_elo_score[loser_id]

    # Calculating Elo score
    id_to_elo_score[winner_id] = calculate_elo_score(
        subject_elo_score=current_winner_rating,
        agent_elo_score=current_loser_rating,
        score=winner_score,
        **calculate_elo_score_params
        )
    id_to_elo_score[loser_id] = calculate_elo_score(
        subject_elo_score=current_loser_rating,
        agent_elo_score=current_winner_rating,
        score=loser_score,
        **calculate_elo_score_params
        )

    return id_to_elo_score


def get_ranking_from_elo_score_dictionary(input_dict, subject_id):
    """
    Orders a dictionary of subject ID keys to ELO score values by ELO score.
    And then gets the rank of the subject with the inputted ID.
    Lower ranks like 1 would represent those subjects with higher
    ELO scores and vice versa.

    Args:
        input_dict(dict):
            Dictionary of subject ID keys to ELO score values
        subject_id(str, int, or any value that's a key in input dict):
            The ID of the subject that you want the ranking of

    Returns:
        int:
            Ranking of the subject with the ID inputted
    """
    # Sorting the subject ID's by ELO score
    sorted_subject_to_elo_score = sorted(
        input_dict.items(),
        key=operator.itemgetter(1),
        reverse=True
        )
    # Getting the rank of the subject based on ELO score
    return [
        subject_tuple[0] for subject_tuple in sorted_subject_to_elo_score
        ].index(subject_id) + 1


def iterate_elo_score_calculation_for_dataframe(
    dataframe,
    winner_column,
    loser_column,
    tie_column=None,
    additional_columns=None
):
    """
    Iterates through a dataframe that has the ID of
    winners and losers for a given event.
    A dictionary will be created that
    contains the information of the event,
    which can then be turned into a dataframe.
    Each key is either from winner or loser's perspective.

    Args:
        dataframe(Pandas DataFrame):
        winner_column(str): The name of the column that has the winner's ID
        loser_column(str): The name of the column that has the loser's ID
        additional_columns(list): Additional columns to take from the

    Returns:
        Dict: With a key value pair for each event either
        from the winner or loser's perspective.
        This can be turned into a dataframe with
        each key value pair being a row.
    """
    if additional_columns is None:
        additional_columns = []

    # Dictionary that keeps track of the current Elo score of the subject
    id_to_elo_score = defaultdict(lambda: 1000)
    # Dictionary that will be converted to a DataFrame
    index_to_elo_score_and_meta_data = defaultdict(dict)

    # Indexes that will identify which row the dictionary
    # key value pair will be
    # The number of the index has no significance other than
    # being the number of the row
    all_indexes = iter(range(0, 99999))

    # Keeping track of the number of matches
    total_match_number = 1

    for index, row in dataframe.dropna(subset=winner_column).iterrows():
        # Getting the ID of the winner subject
        winner_id = row[winner_column]
        # Getting the ID of the loser subject
        loser_id = row[loser_column]

        # Getting the current Elo Score
        current_winner_rating = id_to_elo_score[winner_id]
        current_loser_rating = id_to_elo_score[loser_id]

        if tie_column:
            # When there is nothing in the tie column
            if pd.isna(dataframe[tie_column][index]):
                winner_score = 1
                loser_score = 0
            # When there is value in the tie column
            else:
                winner_score = 0.5
                loser_score = 0.5
        # When there is no tie column
        else:
            winner_score = 1
            loser_score = 0

        # Updating the dictionary with ID keys and Elo Score values
        update_elo_score(
            winner_id=winner_id,
            loser_id=loser_id,
            id_to_elo_score=id_to_elo_score,
            winner_score=winner_score,
            loser_score=loser_score
            )

        # Saving all the data for the winner
        winner_index = next(all_indexes)
        index_to_elo_score_and_meta_data[winner_index]["total_match_number"] = total_match_number
        index_to_elo_score_and_meta_data[winner_index]["subject_id"] = winner_id
        index_to_elo_score_and_meta_data[winner_index]["agent_id"] = loser_id
        index_to_elo_score_and_meta_data[winner_index]["original_elo_score"] = current_winner_rating
        index_to_elo_score_and_meta_data[winner_index]["updated_elo_score"] = id_to_elo_score[winner_id]
        index_to_elo_score_and_meta_data[winner_index]["win_draw_loss"] = winner_score
        index_to_elo_score_and_meta_data[winner_index]["subject_ranking"] = get_ranking_from_elo_score_dictionary(id_to_elo_score, winner_id)
        index_to_elo_score_and_meta_data[winner_index]["agent_ranking"] = get_ranking_from_elo_score_dictionary(id_to_elo_score, loser_id)

        for column in additional_columns:
            index_to_elo_score_and_meta_data[winner_index][column] = row[column]  

        # Saving all the data for the loser
        loser_index = next(all_indexes)
        index_to_elo_score_and_meta_data[loser_index]["total_match_number"] = total_match_number
        index_to_elo_score_and_meta_data[loser_index]["subject_id"] = loser_id
        index_to_elo_score_and_meta_data[loser_index]["agent_id"] = winner_id
        index_to_elo_score_and_meta_data[loser_index]["original_elo_score"] = current_loser_rating
        index_to_elo_score_and_meta_data[loser_index]["updated_elo_score"] = id_to_elo_score[loser_id]
        index_to_elo_score_and_meta_data[loser_index]["win_draw_loss"] = loser_score
        index_to_elo_score_and_meta_data[loser_index]["subject_ranking"] = get_ranking_from_elo_score_dictionary(id_to_elo_score, loser_id)
        index_to_elo_score_and_meta_data[loser_index]["agent_ranking"] = get_ranking_from_elo_score_dictionary(id_to_elo_score, winner_id)
        for column in additional_columns:
            index_to_elo_score_and_meta_data[loser_index][column] = row[column]

        # Updating the match number
        total_match_number += 1

    return index_to_elo_score_and_meta_data


def get_med_pc_meta_data(
    file_path,
    meta_data_headers=None,
    file_path_to_meta_data=None
):
    """
    Parses out the metadata from output of a MED-PC data file.
    The output file looks something like:
        Start Date: 05/04/22
        End Date: 05/04/22
        Subject: 4.4 (4)
        Experiment: Pilot of Pilot
        Group: Cage 4
        Box: 1
        Start Time: 13:06:15
        End Time: 14:10:05
        MSN: levelNP_CS_reward_laserepochON1st_noshock
    The metadata will be saved into a nested default dictionary.
    With the file path as the key, and the meta data headers as the values.
    And then the meta data headers are the nested keys,
    and the meta data as the values.

    The dictionary would look something like:
    defaultdict(dict,
            {'./data/2022-05-04_13h06m_Subject 4.4 (4).txt':
                {'File': 'C:\\MED-PC\\Data\\2022-05-04_13h06m_Subject4.4.txt',
              'Start Date': '05/04/22',
              'End Date': '05/04/22',
              'Subject': '4.4 (4)',
              'Experiment': 'Pilot of Pilot',
              'Group': 'Cage 4',
              'Box': '1',
              'Start Time': '13:06:15',
              'End Time': '14:10:05',
              'MSN': 'levelNP_CS_reward_laserepochON1st_noshock'}})

    Args:
        file_path: str
            - The path to the MED-PC data file
        meta_data_headers: list
            - List of the types of metadata to be parsed out for
            - Default metadata includes: "File", "Start Date",
            "End Date", "Subject", "Experiment", "Group",
            "Box", "Start Time", "End Time", "MSN"
        file_path_to_meta_data: Nested Default Dictionary
            - Any dictionary that has already been produced by this function
            that more metadata is chosen to be added to.
            The dictionary will have the file path as the key,
            and the meta data headers as the values.
            And then the meta data headers are the nested keys,
            and the meta data as the values.

    Returns:
        Nested Default Dictionary:
            - With the file path as the key,
            and the meta data headers as the values.
            And then the meta data headers are the nested keys,
            and the meta data as the values.
    """
    # The default metadata found in MED-PC files
    if meta_data_headers is None:
        meta_data_headers = [
            "File",
            "Start Date",
            "End Date",
            "Subject",
            "Experiment",
            "Group",
            "Box",
            "Start Time",
            "End Time",
            "MSN"
            ]
    # Creating a new dictionary if none is inputted
    if file_path_to_meta_data is None:
        file_path_to_meta_data = defaultdict(dict)

    # List of all the headers that we've gone through
    used_headers = []
    # Going through each line of the MED-PC data file
    with open(file_path, 'r') as file:
        for line in file.readlines():
            # Checking to see if we've gone through all the headers or not
            if set(meta_data_headers) == set(used_headers):
                break
            # Going through each header to see which line
            # starts with the header
            for header in meta_data_headers:
                if line.strip().startswith(header):
                    # Removing all unnecessary characters
                    file_path_to_meta_data[file_path][header] = line.strip().replace(header, '').strip(":").strip()
                    used_headers.append(header)
                    # Move onto next line if header is found
                    break
    return file_path_to_meta_data


def get_all_med_pc_meta_data_from_files(
    list_of_files,
    meta_data_headers=None,
    file_path_to_meta_data=None
):
    """
    Iterates through a list of MED-PC files to extract
    all the metadata from those files

    Args:
        list_of_files: list
            - A list of file paths
            (not names, must be relative or absolute path)
            of MED-PC output files
            - We recommend using glob.glob("./path_to_files/*txt")
            to get list of files
        meta_data_headers: list
            - List of the types of metadata to be parsed out for
            - Default metadata includes: "File", "Start Date", "End Date",
            "Subject", "Experiment", "Group", "Box",
            "Start Time", "End Time", "MSN"
        file_path_to_meta_data: Nested Default Dictionary
            - Any dictionary that has already been produced by
            this function that more metadata is chosen to be added to.
            The dictionary will have the file path as the key,
            and the meta data headers as the values.
            And then the meta data headers are the nested keys,
            and the meta data as the values.

    Returns:
        Nested Default Dictionary:
            - With the file path as the key,
            and the meta data headers as the values.
            And then the meta data headers are the nested keys,
            and the meta data as the values.
    """
    # Creating a new dictionary if none is inputted
    if file_path_to_meta_data is None:
        file_path_to_meta_data = defaultdict(dict)

    for file_path in list_of_files:
        # Parsing out the metadata from MED-PC files
        try:
            file_path_to_meta_data = get_med_pc_meta_data(
                file_path=file_path,
                meta_data_headers=meta_data_headers,
                file_path_to_meta_data=file_path_to_meta_data
                )
        # Except in case file can not be read or is missing
        except Exception:
            print("Please review contents of {}".format(file_path))
    return file_path_to_meta_data


def scale_time_to_whole_number(time, multiplier=100):
    """
    Function used to convert times that are floats into whole numbers
    by scaling it. i.e. from 71.36 to 7136
    This is used with pandas.DataFrame.apply/pandas.Series.apply
    to convert a column of float times to integer times.

    Args:
        time: float
            - The time in seconds that something is happening
    Returns:
        int:
            - Converted whole number time
    """
    try:
        if np.isnan(time):
            return 0
        else:
            return int(time * multiplier)
    except Exception:
        return 0


def get_all_port_entry_increments(port_entry_scaled, port_exit_scaled):
    """
    Gets all the numbers that are in the duration
    of the port entry and port exit times.
    i.e. If the port entry was 7136 and port exit was 7142, we'd get:
    [7136, 7137, 7138, 7139, 7140, 7141, 7142]
    This is done for all port entry and port exit times
    pairs between two Pandas Series

    Args:
        port_entry_scaled: Pandas Series
            - A column from a MED-PC Dataframe that has all
            the port entry times scaled
            (usually with the scale_time_to_whole_number function)
        port_exit_scaled: Pandas Series
            - A column from a MED-PC Dataframe that has all
            the port exit times scaled
            (usually with the scale_time_to_whole_number function)
    Returns:
        Numpy array:
            - 1D Numpy Array of all the numbers that are in the
            duration of all the port entry and port exit times
    """
    all_port_entry_ranges = [
        np.arange(port_entry, port_exit+1) for port_entry, port_exit in zip(port_entry_scaled, port_exit_scaled)
        ]
    return np.concatenate(all_port_entry_ranges)


def get_inside_port_mask(inside_port_numbers, max_time):
    """
    Gets a mask of all the times that the subject is inside the port.
    First a range of number from 1 to the number for the max time is created.
    Then, a mask is created by seeing which numbers
    are within the inside port duration

    Args:
        max_time: int
            - The number that represents the largest number for the time.
                - Usually this will be the number for the last tone played.
            - We recommend adding 2001 if you are just
            using the number for the last tone played
                - This is because we are looking 20 seconds before and after.
                - And 20 seconds becomes 2000 when scaled with our method.
        inside_port_numbers: Numpy Array
            - All the increments of of the duration that the
            subject is within the port
    Returns:
        session_time_increments: Numpy Array
            - Range of number from 1 to max time
        inside_port_mask: Numpy Array
            - The mask of True or False if the subject is in
            the port during the time of that index
    """
    if max_time is None:
        max_time = inside_port_numbers.max()
    session_time_increments = np.arange(1, max_time+1)
    inside_port_mask = np.isin(session_time_increments, inside_port_numbers)
    return session_time_increments, inside_port_mask


def get_inside_port_probability_averages_for_all_increments(tone_times, inside_port_mask, before_tone_duration=2000, after_tone_duration=2000):
    """
    Calculates the average probability that a
    subject is in the port between sessions.
    This is calculated by seeing the ratio that a
    subject is in the port at a given time increment
    that's the same time difference to the tone with all the other sessions.
    i.e. The time increment of 10.01 seconds after the tone for all sessions.

    Args:
        tone_times: list or Pandas Series
            - An array of the times that the tone has played
        inside_port_mask: Numpy Array
            - The mask where the subject is in the port based on the
            index being the time increment
        before_tone_duration: int
            - The number of increments before the tone to be analyzed
        after_tone_duration: int
            - The number of increments after the tone to be analyzed
    Returns:
        Numpy Array
            - The averages of the probabilities that the subject
            is inside the port for all increments
    """
    result = []
    for tone_start in tone_times:
        tone_start_int = int(tone_start)
        result.append(inside_port_mask[tone_start_int - before_tone_duration: tone_start_int + after_tone_duration])
    return np.stack(result).mean(axis=0)


def parse_exported_file(file_path):
    """
    """
    with open(file_path, 'rb') as f:
        # Check if first line is start of settings block
        if f.readline().decode('ascii').strip() != '<Start settings>':
            raise Exception("Settings format not supported")
        fields = True
        fields_text = {}
        for line in f:
            # Read through block of settings
            if fields:
                line = line.decode('ascii').strip()
                # filling in fields dict
                if line != '<End settings>':
                    vals = line.split(': ')
                    fields_text.update({vals[0].lower(): vals[1]})
                # End of settings block, signal end of fields
                else:
                    fields = False
                    dt = parse_fields(fields_text['fields'])
                    fields_text['data'] = np.zeros([1], dtype=dt)
                    break
        # Reads rest of file at once, using dtype format
        # generated by parse_fields()
        dt = parse_fields(fields_text['fields'])
        data = np.fromfile(f, dt)
        fields_text.update({'data': data})
        return fields_text


# Parses last fields parameter (<time uint32><...>) as a single string
# Assumes it is formatted as <name number * type> or <name type>
# Returns: np.dtype
def parse_fields(field_str):
    """
    """
    # Returns np.dtype from field string
    sep = re.split('\s', re.sub(r"\>\<|\>|\<", ' ', field_str).strip())
    # print(sep)
    typearr = []
    # Every two elmts is fieldname followed by datatype
    for i in range(0, sep.__len__(), 2):
        fieldname = sep[i]
        repeats = 1
        ftype = 'uint32'
        # Finds if a <num>* is included in datatype
        if sep[i+1].__contains__('*'):
            temptypes = re.split('\*', sep[i+1])
            # Results in the correct assignment,
            # whether str is num*dtype or dtype*num
            ftype = temptypes[temptypes[0].isdigit()]
            repeats = int(temptypes[temptypes[1].isdigit()])
        else:
            ftype = sep[i+1]
        try:
            fieldtype = getattr(np, ftype)
        except AttributeError:
            print(ftype + " is not a valid field type.\n")
            exit(1)
        else:
            typearr.append((str(fieldname), fieldtype, repeats))
    return np.dtype(typearr)


def get_key_with_substring(input_dict, substring="", return_first=True):
    """
    """
    keys_with_substring = []
    for key in input_dict.keys():
        if substring in key:
            keys_with_substring.append(key)
    if substring in keys_with_substring:
        return substring
    elif return_first:
        return keys_with_substring[0]
    else:
        return keys_with_substring


def get_all_file_suffixes(file_name):
    """
    Creates a string of the suffixes of a file
    name that's joined together by "."
    Suffixes will be all the parts of the file name that follows the first "."
    Example: "file.txt.zip.asc" >> "txt.zip.asc"

    Args:
        file_name(str): Name of the file

    Returns:
        String of all the suffixes joined by "."
    """
    # Getting all the suffixes in the file name
    # And removing any periods before and after
    stripped_suffixes = [
        suffix.strip(".") for suffix in pathlib.Path(file_name).suffixes
        ]

    if stripped_suffixes:
        return ".".join(stripped_suffixes)
    # When the file name is just a ".", the stripped suffix is blank
    else:
        return "."


def update_trodes_file_to_data(file_path, file_to_data=None):
    """
    Get the data/metadata froma a Trodes recording file.
    Save it to a dictionary with the file name as the key.
    And the name of the data/metadata(sub-key)
    and the data/metadata point(sub-value) as a subdictionary for the value.

    Args:
        file_path(str): Path of the Trodes recording file.
        Can be relative or absolute path.
        file_to_data(dict): Dictionary that had the
        trodes file name as the key and the data/metadata as the value.

    Returns:
        Dictionary that has file name keys with a subdictionary of
        all the different data/metadata from the Trodes recording file.
    """
    # Creating a new dictionary if none is inputted
    if file_to_data is None:
        file_to_data = defaultdict(dict)
    # Getting just the file name to use as the key
    file_name = os.path.basename(file_path)
    # Getting the absolute file path as metadata
    absolute_file_path = os.path.abspath(file_path)
    try:
        # Reading in the Trodes recording file with the function
        trodes_recording = parse_exported_file(absolute_file_path)

        file_prefix = get_all_file_suffixes(file_name)
        print("file prefix: {}".format(file_prefix))
        file_to_data[file_prefix] = trodes_recording
        file_to_data[file_prefix]["absolute_file_path"] = absolute_file_path
        return file_to_data
    except Exception:
        # TODO: Fix format so that file path is included in warning
        warnings.warn("Can not process {}".format(absolute_file_path))
        return None


def get_all_trodes_data_from_directory(parent_directory_path="."):
    """
    Goes through all the files in a directory created by Trodes.
    Each file is organized into a dictionary that is directory name to
    the file name to associated data/metadata of the file.
    The structure would look something like:
    result[current_directory_name][file_name][data_type]

    Args:
        parent_directory_path(str): Path of the directory that contains the
        Trodes recording files. Can be relative or absolute path.

    Returns:
        Dictionary that has the Trodes directory name as the key
        and a subdictionary as the values.
        This subdictionary has all the files as keys with the corresponding
        data/metadata from the Trodes recording file as values.
    """
    directory_to_file_to_data = defaultdict(dict)
    # Going through each directory
    for item in os.listdir(parent_directory_path):
        item_path = os.path.join(parent_directory_path, item)
        # Getting the directory name to save as the key
        if os.path.isdir(item_path):
            current_directory_name = os.path.basename(item_path)
        # If the item is a file instead of a directory
        else:
            current_directory_name = "."
        directory_prefix = get_all_file_suffixes(current_directory_name)

        current_directory_path = os.path.join(
            parent_directory_path,
            current_directory_name
            )
        # Going through each file in the directory
        for file_name in os.listdir(current_directory_path):
            file_path = os.path.join(current_directory_path, file_name)
            if os.path.isfile(file_path):
                # Creating a sub dictionary that has file keys and a sub-sub dictionary of data type to data value 
                current_directory_to_file_to_data = update_trodes_file_to_data(file_path=file_path, file_to_data=directory_to_file_to_data[current_directory_name])
                # None will be returned if the file can not be processed
                if current_directory_to_file_to_data is not None:
                    print("directory prefix: {}".format(directory_prefix))
                    directory_to_file_to_data[directory_prefix] = current_directory_to_file_to_data
    return directory_to_file_to_data


def get_max_tone_number(tone_pd_series):
    """
    Gets the index, and the number for valid tones in MED-PC's outputted data.
    The recorded tones produce numbers that are divisible by 1000 after the
    recorded data.
    You can use the index to remove these unnecessary numbers by indexing
    until that number.

    Args:
        tone_pd_series: Pandas Series
            - A column from the dataframe that contains the data from MED-PC's
            output file
            - Usually created with dataframe_variable["(S)CSpresentation"]
    Returns:
        int, float
            - The index of the max tone number. This number can be used to
            index the tone_pd_series to remove unnecessary numbers.
            - The max tone number. This number can be used to verify whether
            or not the tone_pd_series had unnecessary numbers.
    """
    for index, num in enumerate(tone_pd_series):
        if num % 1000 == 0:
            return index, num
    return index, num


def get_valid_tones(tone_pd_series, drop_1000s=True, dropna=True):
    """
    Removes all unnecessary numbers from a Pandas Series of tone times
    extracted from MED-PC's dataframe.
    The unnecessary numbers are added after recorded tone times. These numbers
    are usually divisible by 1000.
    NaNs are also added after that. So we will remove all tone times entries
    that meet either of these criterias.

    Args:
        tone_pd_series: Pandas Series
        dropna: bool
            - Whether or not you want to remove NaNs from tone_pd_series.
            - Usually a good idea because MED-PC adds NaNs to the tone time
            column.
    Returns:
        Pandas series
            - The tone times with unnecessary numbers and NaNs removed
    """
    if dropna:
        tone_pd_series = tone_pd_series.dropna()
    if drop_1000s:
        # Getting the index of the tone time that is divisible by 1000
        max_tone_index, max_tone_number = get_max_tone_number(tone_pd_series=tone_pd_series)
        tone_pd_series = tone_pd_series[:max_tone_index]
    # Removing all numbers that are after the max tone
    return tone_pd_series


def get_first_port_entries_after_tone(
    tone_pd_series,
    port_entries_pd_series,
    port_exits_pd_series
):
    """
    From an array of times of tones being played and subject's entries to a
    port,
    finds the first entry immediately after every tone.
    Makes a dataframe of tone times to first port entry times

    Args:
        tone_pd_series: Pandas Series
            - All the times the tone is being played
        port_entries_pd_series: Pandas Series
            - All the times that the port is being entered
    Returns:
        Pandas DataFrame
            - A dataframe of tone times to first port entry times
    """
    # Creating a dictionary of index(current row number we're on) to
    # current/next tone time and first port entry
    first_port_entry_dict = defaultdict(dict)
    for index, current_tone_time in tone_pd_series.items():
        # Using a counter so that we don't go through all the rows that
        # include NaNs
        try:
            first_port_entry_dict[index]["current_tone_time"] = current_tone_time
            # Getting all the port entries that happened after the tone started
            # And then getting the first one of those port entries
            first_port_entry_after_tone = port_entries_pd_series[port_entries_pd_series >= current_tone_time].min()
            first_port_entry_dict[index]["first_port_entry_after_tone"] = first_port_entry_after_tone
            # Getting all the port exits that happened after the entery
            # And then getting the first one of those port exits
            port_exit_after_first_port_entry_after_tone = port_exits_pd_series[port_exits_pd_series > first_port_entry_after_tone].min()
            first_port_entry_dict[index]["port_exit_after_first_port_entry_after_tone"] = port_exit_after_first_port_entry_after_tone
        except Exception:
            print("Look over value {} at index {}".format(current_tone_time, index))
    return pd.DataFrame.from_dict(first_port_entry_dict, orient="index")


def get_last_port_entries_before_tone(tone_pd_series, port_entries_pd_series, port_exits_pd_series):
    """
    From an array of times of tones being played and subject's entries to a port,
    finds the first entry immediately after every tone.
    Makes a dataframe of tone times to first port entry times

    Args:
        tone_pd_series: Pandas Series
            - All the times the tone is being played
        port_entries_pd_series: Pandas Series
            - All the times that the port is being entered
    Returns:
        Pandas DataFrame
            - A dataframe of tone times to first port entry times
    """
    # Creating a dictionary of index(current row number we're on) to current/next tone time and first port entry
    last_port_entry_dict = defaultdict(dict)
    for index, current_tone_time in tone_pd_series.items():
        # Using a counter so that we don't go through all the rows that include NaNs
        try:
            last_port_entry_dict[index]["current_tone_time"] = current_tone_time
            # Getting all the port entries that happened after the tone started
            # And then getting the first one of those port entries
            last_port_entry_before_tone = port_entries_pd_series[port_entries_pd_series <= current_tone_time].max()
            last_port_entry_dict[index]["last_port_entry_before_tone"] = last_port_entry_before_tone
            # Getting all the port exits that happened after the entery
            # And then getting the first one of those port exits
            port_exit_after_last_port_entry_before_tone = port_exits_pd_series[port_exits_pd_series > last_port_entry_before_tone].min()
            last_port_entry_dict[index]["port_exit_after_last_port_entry_before_tone"] = port_exit_after_last_port_entry_before_tone
        except Exception:
            print("Look over value {} at index {}".format(current_tone_time, index))
    return pd.DataFrame.from_dict(last_port_entry_dict, orient="index")


def get_concatted_first_porty_entry_after_tone_dataframe(
    concatted_medpc_df,
    tone_time_column="(S)CSpresentation",
    port_entry_column="(P)Portentry",
    port_exit_column="(N)Portexit",
    subject_column="subject",
    date_column="date",
    stop_with_error=False
):
    """
    Creates dataframes of the time of the tone, and the first port entry after
    that tone.
    Along with the corresponding metadata of the path of the file, the date,
    and the subject.
    This is created from a dataframe that contains tone times, port entry
    times, and associated metadata.
    Which is usually from the extract.dataframe.
    get_medpc_dataframe_from_list_of_files function

    Args:
        concatted_medpc_df: Pandas Dataframe
            - Output of
            extract.dataframe.get_medpc_dataframe_from_list_of_files
            - Includes tone playing time, port entry time, subject,
            and date for each recording session
        tone_time_column: str
            - Name of the column of concatted_medpc_df that has the array port
            entry times
        port_entry_column: str
            - Name of the column of concatted_medpc_df that has the array port
            entry times
        subject_column: str
            - Name of the column of concatted_medpc_df that has the subject's
            ID
        date_column: str
            - Name of the column of concatted_medpc_df that has the date of
            the recording
        stop_with_error: bool
            - Flag to terminate the program when an error is raised.
            - Sometimes recordings can be for testing and don't include any
            valid tone times

    Returns:
        Pandas Dataframe
            -
    """
    # List to combine all the Data Frames at the end
    all_first_port_entry_df = []
    for file_path in concatted_medpc_df["file_path"].unique():
        current_file_df = concatted_medpc_df[concatted_medpc_df["file_path"] == file_path]
        valid_tones = get_valid_tones(
            tone_pd_series=current_file_df[tone_time_column]
            )
        # Sometimes the valid tones do not exist because it was a
        # test recording
        if not valid_tones.empty:
            # All the first port entries for each tone
            first_port_entry_df = get_first_port_entries_after_tone(
                tone_pd_series=valid_tones,
                port_entries_pd_series=current_file_df[port_entry_column],
                port_exits_pd_series=current_file_df[port_exit_column]
                )
            # Adding the metadata as columns
            first_port_entry_df["file_path"] = file_path
            # Making sure that there is only one date and
            # subject for all the rows
            if len(current_file_df[date_column].unique()) == 1 and len(current_file_df[subject_column].unique()) == 1:
                # This assumes that all the date and subject keys are the same for the file
                first_port_entry_df[date_column] = current_file_df[date_column].unique()[0]
                first_port_entry_df[subject_column] = current_file_df[subject_column].unique()[0]
            elif stop_with_error:
                raise ValueError("More then one date or subject in {}".format(file_path))
            else:
                print("More then one date or subject in {}".format(file_path))
            all_first_port_entry_df.append(first_port_entry_df)
        elif valid_tones.empty and stop_with_error:
            raise ValueError("No valid tones for {}".format(file_path))
        else:
            print("No valid tones for {}".format(file_path))
    # Index repeats itself because it is concatenated with multiple dataframes
    return pd.concat(all_first_port_entry_df).reset_index(drop="True")


def get_concatted_last_porty_entry_before_tone_dataframe(
    concatted_medpc_df,
    tone_time_column="(S)CSpresentation",
    port_entry_column="(P)Portentry",
    port_exit_column="(N)Portexit",
    subject_column="subject",
    date_column="date",
    stop_with_error=False
):
    """
    Creates dataframes of the time of the tone, and the first port entry after
    that tone.
    Along with the corresponding metadata of the path of the file, the date,
    and the subject.
    This is created from a dataframe that contains tone times,
    port entry times,
    and associated metadata.
    Which is usually from the
    extract.dataframe.get_medpc_dataframe_from_list_of_files function

    Args:
        concatted_medpc_df: Pandas Dataframe
            - Output of
            extract.dataframe.get_medpc_dataframe_from_list_of_files
            - Includes tone playing time, port entry time, subject,
            and date for each recording session
        tone_time_column: str
            - Name of the column of concatted_medpc_df that has the array port
            entry times
        port_entry_column: str
            - Name of the column of concatted_medpc_df that has the array port
            entry times
        subject_column: str
            - Name of the column of concatted_medpc_df that has the subject's
            ID
        date_column: str
            - Name of the column of concatted_medpc_df that has the date of
            the recording
        stop_with_error: bool
            - Flag to terminate the program when an error is raised.
            - Sometimes recordings can be for testing and don't include any
            valid tone times

    Returns:
        Pandas Dataframe
            -
    """
    # List to combine all the Data Frames at the end
    all_last_port_entry_df = []
    for file_path in concatted_medpc_df["file_path"].unique():
        current_file_df = concatted_medpc_df[concatted_medpc_df["file_path"] == file_path]
        valid_tones = get_valid_tones(
            tone_pd_series=current_file_df[tone_time_column]
            )
        # Sometimes the valid tones do not exist because it was a
        # test recording
        if not valid_tones.empty:
            # All the first port entries for each tone
            last_port_entry_df = get_last_port_entries_before_tone(
                tone_pd_series=valid_tones,
                port_entries_pd_series=current_file_df[port_entry_column],
                port_exits_pd_series=current_file_df[port_exit_column]
                )
            # Adding the metadata as columns
            last_port_entry_df["file_path"] = file_path
            # Making sure that there is only one date and subject
            # for all the rows
            if len(current_file_df[date_column].unique()) == 1 and len(current_file_df[subject_column].unique()) == 1:
                # This assumes that all the date and subject keys are the same for the file
                last_port_entry_df[date_column] = current_file_df[date_column].unique()[0]
                last_port_entry_df[subject_column] = current_file_df[subject_column].unique()[0]
            elif stop_with_error:
                raise ValueError("More then one date or subject in {}".format(file_path))
            else:
                print("More then one date or subject in {}".format(file_path))
            all_last_port_entry_df.append(last_port_entry_df)
        elif valid_tones.empty and stop_with_error:
            raise ValueError("No valid tones for {}".format(file_path))
        else:
            print("No valid tones for {}".format(file_path))
    # Index repeats itself because it is concatenated with multiple dataframes
    return pd.concat(all_last_port_entry_df).reset_index(drop="True")


def get_info(filename):
    with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]
        track_names = [n.decode() for n in f["track_names"][:]]
    return dset_names, locations, node_names, track_names


def fill_missing(Y, kind="linear"):
    """
    Fills missing values independently along each dimension
    after the first.
    """

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(
            np.flatnonzero(mask),
            np.flatnonzero(~mask),
            y[~mask]
            )

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y


def get_mouse_sides(locations):
    '''
    takes in locations
    returns (left mouse index, right mouse index)
    '''
    if locations[0,1,0,0] > locations[0,1,0,1]:
        # if mouse 0 has a greater starting x value for its nose,
        # then it is the right mouse
        return (1, 0)
    else:
        return (0, 1)


def smooth_diff(node_loc, win=25, poly=3):
    """
    node_loc is a [frames, 2] array

    win defines the window to smooth over

    poly defines the order of the polynomial
    to fit with
    """
    node_loc_vel = np.zeros_like(node_loc)

    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)

    node_vel = np.linalg.norm(node_loc_vel,axis=1)

    return node_vel


def smooth_diff_one_dim(node_loc, win=25, poly=3):
    """
    node_loc is a [frames] array

    win defines the window to smooth over

    poly defines the order of the polynomial
    to fit with

    """
    node_loc_vel = np.zeros_like(node_loc)

    node_loc_vel[:] = savgol_filter(node_loc[:], win, poly, deriv=1)

    return node_loc_vel


def get_distances_between_mice(locations, node_index):
    # confirmed does what i want it to
    """
    takes in locations and node index
    returns a list of distances between the nodes of the two mice
    """
    c_list = []
    left_mouse, right_mouse = get_mouse_sides(locations)
    for i in range(len(locations)):

        (x1, y1) = (locations[i, node_index, 0, left_mouse],
                    locations[i, node_index, 1, left_mouse])
        # x , y coordinate of nose for mouse 1
        (x2, y2) = (locations[i, node_index, 0, right_mouse],
                    locations[i, node_index, 1, right_mouse])
        # x and y coordinate of nose of mouse 2
        # solve for c using pythagroean theory
        a2 = (x1 - x2) ** 2
        b2 = (y1 - y2) ** 2
        c = math.sqrt(a2 + b2)
        if x1 > x2:
            c_list.append(-1*c)
        else:
            c_list.append(c)
    return c_list


def get_distances_between_nodes(locations, node_index1, node_index2):
    # CONFIRMED THAT IT WORKS in terms of doing the math by hand
    """
    takes in locations and node indexes of the two body parts you want
    within mice distances for

    returns nested lists, list[0] is the distances within track1
    list[1] is the distances within track2

    """
    c_list = []
    m1_c_list = []
    m2_c_list = []
    left_mouse, right_mouse = get_mouse_sides(locations)
    for i in range(len(locations)):
        x1, y1 = locations[i, node_index1, 0, 0], locations[i, node_index1, 1, left_mouse]
        # x , y coordinate of node 1 for mouse 1
        x2, y2 = locations[i, node_index2, 0, 0], locations[i, node_index2, 1, left_mouse]
        # x, y coordiantes of node 2 for mouse 1
        x3, y3 = locations[i, node_index1, 0, 1], locations[i, node_index1, 1, right_mouse]
        # x and y coordinate of node 1 of mouse 2
        x4, y4 = locations[i, node_index2, 0, 1], locations[i, node_index2, 1, right_mouse]
        # solve for c using pythagroean theory
        a2 = (x1 - x2) ** 2
        b2 = (y1 - y2) ** 2
        a2_m2 = (x3 - x4) ** 2
        b2_m2 = (y3 - y4) ** 2
        c2 = math.sqrt(a2_m2 + b2_m2)
        c1 = math.sqrt(a2 + b2)
        m1_c_list.append(c1)
        m2_c_list.append(c2)
    c_list.append(m1_c_list)
    c_list.append(m2_c_list)
    return c_list


def get_speeds(locations, node_index):

    node_loc_1 = locations[:,node_index,:,0]
    # node loc (x,y) of node of mouse 1
    node_loc_2 = locations[:,node_index,:,1]
    # x,y's of node of mouse 2
    m1_vel = smooth_diff(node_loc_1)
    m2_vel = smooth_diff(node_loc_2)
    velocities = [m1_vel,m2_vel]
    return velocities


def get_velocities(locations, node_index):

    left_mouse, right_mouse = get_mouse_sides(locations)
    node_loc_left = locations[:, node_index, 0, left_mouse]
    # node loc (x,y) of node of mouse 1
    node_loc_right = (locations[:, node_index, 0, right_mouse]) * (-1)
    # x,y's of node of mouse 2

    m1_vel = smooth_diff_one_dim(node_loc_left)
    m2_vel = smooth_diff_one_dim(node_loc_right)
    velocities = [m1_vel, m2_vel]
    return velocities


def get_angles(locations, node_index_1, node_index_2, node_index_3):
    """
    takes in locations and three nodes, calculates angle between the
    three points
    with the second node being the center point
    i.e. node_1 = nose , node_2 = ear , node_3 = thorax
    returns [[list of angles for mouse 1][list of angles for mouse 2]]
    """
    angles_all_mice = []
    frame, nodes, axes, mice = locations.shape

    for mouse in range(mice):
        angles = []
        for i in range(len(locations)):
            a = np.array([locations[i, node_index_1, 0, mouse], locations[i, node_index_1, 1, mouse]])
            b = np.array([locations[i, node_index_2, 0, mouse], locations[i, node_index_2, 1, mouse]])
            c = np.array([locations[i, node_index_3, 0, mouse], locations[i, node_index_3, 1, mouse]])
            ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
            if ang < 0:
                angles.append(ang + 360)
            else:
                angles.append(ang)
        angles_all_mice.append(angles)
    return angles_all_mice


def cluster_dic(labels):
    """
    takes in a list of labels (hdbscan labels)
    and returns a dictionary {cluster:[list of frames]}
    """
    clusters = {}
    print(labels)
    # if a cluster key already exists in the dictionary, append its value
    # (list) with the new frame (i)
    for i in range(len(labels)):
        if labels[i] in clusters:
            temp_val = clusters[labels[i]]
            temp_val.append(i)
            clusters[labels[i]] = temp_val
        # if the cluser does not have a unique key yet, create one, who
        else:
            clusters[labels[i]] = [i]
    return clusters


# create temp list of frames for the range you are on so you only have to open
# each video once
def make_clip(list_of_frames, framedic):
    vid_to_frames_dict = {}
    frames = []
    # video name , value list of frame from that video in this cluster
    # string : list of integers
    for frame in list_of_frames:
        for key, value in framedic.items():
            start = value[0]
            stop = value[1]
            if frame in range(start, stop):
                if key in vid_to_frames_dict:
                    vid_to_frames_dict[key].append(frame)
                else:
                    vid_to_frames_dict[key] = [frame]
                break
    for key in vid_to_frames_dict:
        vid = VideoFileClip(key)
        for frame in vid_to_frames_dict[key]:
            start_of_vid_frame = framedic[key][0]
            frames.append(vid.get_frame((frame - start_of_vid_frame)/30))
        vid.close()

    clip = mpy.ImageSequenceClip(frames, fps=30)
    return clip


def contact(node_array_m1, node_array_m2, epsilon):
    """
    given two node location arrays for mouse 1 and mouse 2 in one dimension,
    nose nodes recommended for tube test in the x dimension,
    returns a boolean of the number of frames of the trial/video
    true = contact, false = no contact
    epsilon = threshold for closeness that defines a contact
    """
    contact_array = [0] * len(node_array_m1)
    # get left mouse
    # left of screen in 0 on x axis
    if node_array_m1[0] > node_array_m2[0]:
        left_array = node_array_m2
        right_array = node_array_m1
        print('LEFT MOUSE IS MOUSE 2')
    else:
        left_array = node_array_m1
        right_array = node_array_m2
        print('LEFT MOUSE IS MOUSE 1')
    for i in range(len(node_array_m1)):
        if abs(node_array_m1[i] - node_array_m2[i]) < epsilon:
            contact_array[i] = True
        if left_array[i] > right_array[i]:
            contact_array[i] = True
        else:
            contact_array[i] = False
    return contact_array


# def spikesort():
#     pwd = os.getcwd() + "/spikesort"
#     print(pwd)
#     prb_file_path = Path(f"{pwd}/data/nancyprobe_linearprobelargespace.prb")
#     probe_object = read_prb(prb_file_path)
#     probe_df = probe_object.to_dataframe()
#     print(probe_df)
#     recording_filepath_glob = str(Path(f"{pwd}/data/**/*merged.rec"))
#     all_recording_files = glob.glob(recording_filepath_glob, recursive=True)

#     for recording_file in all_recording_files:
#         trodes_recording = se.read_spikegadgets(recording_file, stream_id="trodes")       
#         trodes_recording = trodes_recording.set_probes(probe_object)
#         recording_basename = os.path.basename(recording_file)
#         recording_output_directory = str(Path(f"{pwd}/proc/{recording_basename}"))
#         os.makedirs(recording_output_directory, exist_ok=True)
#         child_spikesorting_output_directory = os.path.join(recording_output_directory,"ss_output")
#         child_recording_output_directory = os.path.join(recording_output_directory,"preprocessed_recording_output")
#         child_lfp_output_directory = os.path.join(recording_output_directory,"lfp_preprocessing_output")

#         print("Calculating LFP...")
#         # Make sure the recording is preprocessed appropriately
#         # lazy preprocessing
#         recording_filtered = sp.bandpass_filter(trodes_recording, freq_min=300, freq_max=6000)
#         # Do LFP
#         # Notch Filtering, keeping all the points that are within a certain frequency range
#         recording_notch = sp.notch_filter(recording_filtered, freq=60)
#         # We are not going to run the resampling step because it causes issues with saving to file?
#         # Resampling
#         # recording_resample = sp.resample(recording_notch, resample_rate=1000)
#         print("Saving LFP result...")
#         recording_notch.save_to_folder(name="lfp_preprocessing", folder=child_lfp_output_directory, n_jobs=8)
#         print("Spikesorting preprocessing...")
#         recording_preprocessed: si.BaseRecording = sp.whiten(recording_filtered, dtype='float32')
#         spike_sorted_object = ms5.sorting_scheme2(
#         recording=recording_preprocessed,
#         sorting_parameters=ms5.Scheme2SortingParameters(
#             detect_sign=0,
#             phase1_detect_channel_radius=700,
#             detect_channel_radius=700,
#             # other parameters...
#             )
#                 )
#         print("Saving variables...")
#         spike_sorted_object_disk = spike_sorted_object.save(folder=child_spikesorting_output_directory)
#         recording_preprocessed_disk = recording_preprocessed.save(folder=child_recording_output_directory)

#         sw.plot_rasters(spike_sorted_object)
#         plt.title(recording_basename)
#         plt.ylabel("Unit IDs")

#         plt.savefig(os.path.join(recording_output_directory, f"{recording_basename}_raster_plot.png"))
#         plt.close()

#         waveform_output_directory = os.path.join(recording_output_directory, "waveforms")

#         print("Extracting Waveforms...")
#         we_spike_sorted = si.extract_waveforms(recording=recording_preprocessed_disk, 
#                                        sorting=spike_sorted_object_disk, folder=waveform_output_directory,
#                                       ms_before=1, ms_after=1, progress_bar=True,
#                                       n_jobs=8, total_memory="1G", overwrite=True,
#                                        max_spikes_per_unit=2000)

#         phy_output_directory = os.path.join(recording_output_directory, "phy")
#         print("Saving PHY2 output...")
#         export_to_phy(we_spike_sorted, phy_output_directory,
#               compute_pc_features=True, compute_amplitudes=True, remove_if_exists=False)
#         print("PHY2 output Saved!")

#     return "SPIKES ARE SORTED & LFP DONE! :)"
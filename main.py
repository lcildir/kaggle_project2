# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import csv

import nltk as nltk
import pandas
import pandas as pd
import numpy as np
from sklearn import linear_model
from datetime import datetime

easy_words_list = []
awl_list = []
l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
l6 = []
l7 = []
l8 = []
l9 = []
l10 = []


def open_files():
    """
    This function open train dataset file, invoke clean data,write_report_to_file, prepare_target_for_test_data
    and prepare_submission_file functions.
    """
    print("Open Train dataset file")
    columns_list = ["id", "excerpt", "target", "standard_error"]
    file_name = "train.csv"
    dataset_parsed = parse_data_from_dataset(columns_list, file_name)

    awl_file_name = "AWL.csv"
    awl_columns_list = ["l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8", "l9", "l10"]
    #parse_list_from_awl(awl_columns_list, awl_file_name)
    clean_data = check_missing_values(dataset_parsed)
    report_list = prepare_report_list(clean_data)
    write_report_to_file(report_list)
    prepare_target_for_test_data()
    prepare_submission_file()


def check_missing_values(pandas_dataframe):
    """
    This function checks if the dataset/dataframe contains any missing values and prints the result
    :param pandas_dataframe: This is the dataframe already parsed from the CSV file by pandas
    """
    # Check if dataframe contains null value
    print("Check missing values")
    dataframe_checked = np.where(pd.isnull(pandas_dataframe))
    # If dataframe contains any element with null value, print a message
    if len(dataframe_checked) == 0:
        print("No missing value found in the dataset.")
    else:
        count = 0
        count_value = 0
        missing_value_row = list()
        missing_value_column = list()
        # Loop through the tuple with containing the rows and columns number with missing values
        for missing_values in dataframe_checked:
            # If missing_values size is zero, it means there is no missing values found
            if len(missing_values) == 0:
                # Print a message and break the loop.
                print("No missing value found in the dataset.")
                return pandas_dataframe
            # Else, save the missing values row and columns into two different lists to print later
            else:
                if count == 0:
                    print(f"We found {len(missing_values)} missing values in the dataset:\n")
                for value in missing_values:
                    if count == 0 or (count / 2) == 0:
                        missing_value_row.append(value + 2)
                        # print(f"Row of missing value {count_value + 1}: {value + 2}")
                        count_value += 1
                    else:
                        missing_value_column.append(value + 1)
                        # print(f"Column missing value {count_value + 1}: {value + 1}")
                        count_value += 1
            count_value = 0
            count += 1

        # Loop through each row and columns lists and print the missing values
        count = 1
        for row in missing_value_row:
            for column in missing_value_column:
                print(f"Missing value {count}: Row {row}, Column {column}")
                break
            count += 1
        print("\nClean dataset.")
        return clean_dataset(pandas_dataframe, missing_value_row)


def clean_dataset(dataset, row_numbers):
    """
    This function clean the dataset.
    :param dataset: This is the dataframe already parsed from the CSV file by pandas
    :param row_numbers: Number of rows which has missing data.
    :return: clean dataset.
    """
    print("Clean dataset")
    for row_number in row_numbers:
        dataset.drop(row_number - 2, axis=0, inplace=True)
    return dataset


def prepare_submission_file():
    """
    This function prepare submission file for Kaggle competition.
    """
    print("Prepare submission file")
    test_file_name = "test.csv"
    submission_file = "submission.csv"
    report_file = "cleanreport.csv"

    # Read from report.csv to calculate Multiple Regression Model
    report_file = pandas.read_csv(report_file)
    X = report_file[["total_words", "total_sentences", "total_syllables", "ave_syllables", "ave_sentence_length",
                     "num_verb_noun", "num_adj_noun", "per_diff_words",
                     "flesch_score", "dale_score", "sumner_level", "coleman_index", "readability_index",
                     "gunning_fog_score", "awl_index"]]
    y = report_file['target']

    # Print out the statistics
    # print(model.summary())

    regr_model = linear_model.LinearRegression()
    regr_model.fit(X, y)
    # print(regr_model.coef_)
    # print("intercept")
    # print(regr_model.__dict__)

    test_file = pandas.read_csv(test_file_name)
    test_file_list = test_file[["id", "excerpt"]]
    submission_report_list = []

    for n in range(len(test_file_list)):
        id = test_file_list.values[n][0]
        text = test_file_list.values[n][1]
        sentences = tokenize_by_sentence(text)
        total_sentences = len(sentences)
        words_list = extract_words_from_text(text)
        total_words = len(words_list)
        awl_index = calculate_awl_index(words_list)
        num_verb_noun = chunk_verb_noun(text)
        num_adj_noun = chunk_adj_noun(text)

        total_letters = get_total_letters_from_text(text)
        total_syllables = syllables_in_text(words_list)
        ave_syllables = total_syllables / total_words
        average_sentence_length_in_words = total_words / total_sentences
        per_diff_words = get_percentage_of_difficult_words(words_list, easy_words_list)
        flesch_score = flesch_reading_ease(total_words, total_sentences, total_syllables)
        dale_chall_score = dale_chall_readability_score(per_diff_words, average_sentence_length_in_words)
        sumner_level = powers_sumner_kearl_grade_level(total_words, total_sentences, total_syllables)
        coleman_index = coleman_liau_index(total_words, total_sentences, total_letters)
        readability_score = automated_readability_index(total_words, total_sentences, total_letters)
        gunning_fog_score = gunning_fog(per_diff_words, average_sentence_length_in_words)

        target = regr_model.predict([[total_words, total_sentences, total_syllables, ave_syllables,
                                      average_sentence_length_in_words, num_verb_noun, num_adj_noun,
                                      per_diff_words, flesch_score, dale_chall_score, sumner_level, coleman_index,
                                      readability_score, gunning_fog_score, awl_index]])
        # print(target)
        submission_report_list.append([id, target[0]])

    with open(submission_file, 'w', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "target"])
        for report in submission_report_list:
            writer.writerow([report[0], report[1]])


def write_report_to_file(report_list):
    """
    This function write the report list to report.csv file.
    :param report_list: List of the report for the text file.
    """
    print("Write calculated data to report file")
    with open('report.csv', 'w', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "target", "standard_error", "total_words", "total_sentences", "total_syllables",
                         "ave_syllables", "ave_sentence_length", "num_verb_noun", "num_adj_noun", "per_diff_words",
                         "flesch_score", "dale_score", "sumner_level",
                         "coleman_index", "readability_index", "gunning_fog_score", "awl_index"])
        for report in report_list:
            writer.writerow([report.id, report.target, report.std_error, report.total_words, report.total_sentences,
                             report.total_syllables, report.ave_syllables, report.ave_sentence_length,
                             report.num_verb_noun, report.num_adj_noun, report.per_diff_words,
                             report.flesch_score, report.dale_score, report.sumner_level, report.coleman_index,
                             report.readability_index, report.gunning_fog_score, report.awl_index])


def prepare_target_for_test_data():
    """
    This function calculate target score for train dataset to clean dataset.
    """
    print("Calculate target score to clean outlineers.")
    clean_report_file = "cleanreport.csv"
    report_file = "report.csv"

    # Read from report.csv to calculate Multiple Regression Model
    report_file = pandas.read_csv(report_file)
    X = report_file[["total_words", "total_sentences", "total_syllables", "ave_syllables", "ave_sentence_length",
                     "num_verb_noun", "num_adj_noun", "per_diff_words",
                     "flesch_score", "dale_score", "sumner_level", "coleman_index", "readability_index",
                     "gunning_fog_score", "awl_index"]]
    y = report_file['target']

    regr_model = linear_model.LinearRegression()
    regr_model.fit(X, y)

    # old_file = pandas.read_csv(report_file)
    old_file_list = report_file[["id", "target", "standard_error", "total_words", "total_sentences", "total_syllables",
                                 "ave_syllables", "ave_sentence_length", "num_verb_noun", "num_adj_noun",
                                 "per_diff_words", "flesch_score", "dale_score", "sumner_level",
                                 "coleman_index", "readability_index", "gunning_fog_score", "awl_index"]]
    new_file_list = []
    t = 0;
    print("Clean train dataset.")
    with open(clean_report_file, 'w', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "target", "standard_error", "total_words", "total_sentences", "total_syllables",
                         "ave_syllables", "ave_sentence_length", "num_verb_noun", "num_adj_noun", "per_diff_words",
                         "flesch_score", "dale_score", "sumner_level",
                         "coleman_index", "readability_index", "gunning_fog_score", "awl_index", "new_target"])

        for n in range(len(old_file_list)):
            target = regr_model.predict(
                [[old_file_list.values[n][3], old_file_list.values[n][4], old_file_list.values[n][5],
                  old_file_list.values[n][6], old_file_list.values[n][7], old_file_list.values[n][8],
                  old_file_list.values[n][9], old_file_list.values[n][10], old_file_list.values[n][11],
                  old_file_list.values[n][12], old_file_list.values[n][13], old_file_list.values[n][14],
                  old_file_list.values[n][15], old_file_list.values[n][16], old_file_list.values[n][17]]])
            if (old_file_list.values[n][1] - old_file_list.values[n][2]) < target[0] < (
                    old_file_list.values[n][1] + old_file_list.values[n][2]):
                writer.writerow([old_file_list.values[n][0], old_file_list.values[n][1], old_file_list.values[n][2],
                                 old_file_list.values[n][3],
                                 old_file_list.values[n][4], old_file_list.values[n][5], old_file_list.values[n][6],
                                 old_file_list.values[n][7],
                                 old_file_list.values[n][8], old_file_list.values[n][9], old_file_list.values[n][10],
                                 old_file_list.values[n][11],
                                 old_file_list.values[n][12], old_file_list.values[n][13], old_file_list.values[n][13],
                                 old_file_list.values[n][15], old_file_list.values[n][16], old_file_list.values[n][17],
                                 target[0]])
            # print(target)


def parse_data_from_dataset(columns, dataset_path):
    """ This function parse data from a csv dataset
    It uses pandas package to parse and read data from the csv dataset using only the columns from columns_list.
    :param columns: This argument receives a list of columns that will be used in the dataframe
    :param dataset_path: Thi argument receives the dataset name and path to be loaded by pandas
    :return: Returns the tabulated parsed data
    """
    # This try and except block tries to read from the dataset using pandas and throws an exception
    # if file does not exists
    try:
        dataset = pd.read_csv(dataset_path, index_col=False, usecols=columns)
        return dataset
    # If file does not exists, it will show a message to the user
    except FileNotFoundError:
        print("File does not exists! Please certify if the file name or path is correct.")


def parse_list_from_awl(columns, dataset_path):
    """ This function parse data from a csv dataset
    It uses pandas package to parse and read data from the csv dataset using only the columns from columns_list.
    :param columns: This argument receives a list of columns that will be used in the dataframe
    :param dataset_path: Thi argument receives the dataset name and path to be loaded by pandas
    :return: Returns the tabulated parsed data
    """
    # This try and except block tries to read from the dataset using pandas and throws an exception
    # if file does not exists
    try:
        dataset = pd.read_csv(dataset_path, index_col=False, usecols=columns)
        for count in range(len(dataset)):
            l1.append(dataset.iloc[count].l1)
            l2.append(dataset.iloc[count].l2)
            l3.append(dataset.iloc[count].l3)
            l4.append(dataset.iloc[count].l4)
            l5.append(dataset.iloc[count].l5)
            l6.append(dataset.iloc[count].l6)
            l7.append(dataset.iloc[count].l7)
            l8.append(dataset.iloc[count].l8)
            l9.append(dataset.iloc[count].l9)
            l10.append(dataset.iloc[count].l10)

    # If file does not exists, it will show a message to the user
    except FileNotFoundError:
        print("File does not exists! Please certify if the file name or path is correct.")


def create_easy_word_list():
    """
    This function creates a easy words list from a file.
    :return: list of easy words
    """
    my_file = open("dale-chall-word-list.txt", "r")
    content = my_file.read()
    easy_word_list = []
    for easy_word in content.split():
        easy_word_list.append(easy_word)
    return easy_word_list


def get_percentage_of_difficult_words(words_list, easy_words_list):
    """
    This function calculate percentage of the difficult words in a text by using easy words list from a file.
    :return: percentage of the difficult words
    """
    count = 0
    for word in words_list:
        if (not word.isdigit()) and (word not in easy_words_list) and (syllables_in_word(word) > 2):
            count += 1
    return 100 * count / len(words_list)


def calculate_awl_index(words_list):
    awl_index = 0
    for word in words_list:
        if word in l1:
            awl_index += 1
        if word in l2:
            awl_index += 2
        if word in l3:
            awl_index += 3
        if word in l4:
            awl_index += 4
        if word in l5:
            awl_index += 5
        if word in l6:
            awl_index += 6
        if word in l7:
            awl_index += 7
        if word in l8:
            awl_index += 8
        if word in l9:
            awl_index += 9
        if word in l10:
            awl_index += 10
    return awl_index


def prepare_report_list(dataset):
    """
    This function prepares a list of readability score and grade level for a text file.
    :param dataset: Text for the analysing.
    :return: List of reports.
    """
    print("Prepare report list for train dataset.")
    report_list = []
    number_of_text = len(dataset)
    # number_of_text = 20
    global easy_words_list
    easy_words_list = create_easy_word_list()

    for count in range(number_of_text):
        id = dataset.iloc[count].id
        target = dataset.iloc[count].target
        std_error = dataset.iloc[count].standard_error
        text = dataset.iloc[count].excerpt
        sentences = tokenize_by_sentence(text)
        total_sentences = len(sentences)
        words_list = extract_words_from_text(text)
        total_words = len(words_list)

        num_verb_noun = chunk_verb_noun(text)
        num_adj_noun = chunk_adj_noun(text)
        awl_index = calculate_awl_index(words_list)

        total_letters = get_total_letters_from_text(text)
        total_syllables = syllables_in_text(words_list)
        ave_syllables = total_syllables / total_words
        average_sentence_length_in_words = total_words / total_sentences
        per_diff_words = get_percentage_of_difficult_words(words_list, easy_words_list)
        flesch_score = flesch_reading_ease(total_words, total_sentences, total_syllables)
        dale_chall_score = dale_chall_readability_score(per_diff_words, average_sentence_length_in_words)
        sumner_level = powers_sumner_kearl_grade_level(total_words, total_sentences, total_syllables)
        coleman_index = coleman_liau_index(total_words, total_sentences, total_letters)
        readability_score = automated_readability_index(total_words, total_sentences, total_letters)
        gunning_fog_score = gunning_fog(per_diff_words, average_sentence_length_in_words)
        new_report = Reports(id, target, std_error, total_words, total_sentences, total_syllables, ave_syllables,
                             average_sentence_length_in_words, num_verb_noun, num_adj_noun, per_diff_words,
                             flesch_score, dale_chall_score,
                             sumner_level, coleman_index, readability_score, gunning_fog_score, awl_index)
        report_list.append(new_report)
    return report_list


def tokenize_by_sentence(text_to_tokenize):
    """
    This function tokenizes text by sentences, it creates a python list by
    breaking text into individual sentences.
    :param text_to_tokenize: Text that need to be tokenized
    """
    # Tokenize the text in sentences and save as a list in the variable
    sentences_tokenized = nltk.sent_tokenize(text_to_tokenize)
    # Return the text tokenized by sentence
    return sentences_tokenized


def get_total_letters_from_text(text):
    """
    This function calculates total digit and letters in a text.
    :param text: text
    """
    total_letters = 0
    for letter in text:
        if letter.isdigit() or letter.isalpha():
            total_letters += 1
    return total_letters


def extract_words_from_text(text_to_tokenize):
    """
    This function extracts words from a text by using split method.
    :param text_to_tokenize: Text which will be extracted.
    """
    words_list = []
    split_words = text_to_tokenize.split()
    for word in split_words:
        words_list.append(word)
    return words_list


def syllables_in_text(word_list):
    """
    This function calculate total syllables in a text.
    :return: Total number of syllables in the text.
    """
    total_syllables = 0
    vowels = 'aeiouy'
    for word in word_list:
        count = 0
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
        if count == 0:
            count += 1
        total_syllables += count
    return total_syllables


def syllables_in_word(word):
    """
    This function calculate total syllables in a text.
    :return: Total number of syllables in the text.
    """
    vowels = 'aeiouy'
    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1
    if count == 0:
        count += 1

    return count


def chunk_verb_noun(text_to_chunk):
    """
    This function creates a chunk of words by using special regexp syntax to find pattern in the text.
    In this case the function uses a regexp rule to find a VB (Verb) followed by a NN (Noun), it it
    finds the patters, it saves each chunk in a list and returns it.
    :param text_to_chunk: Text used to find words + noun in it.
    """
    words = extract_words_from_text(text_to_chunk)
    # Tags each token word with from the list with its own respective type: (Verb, noun, adjective, etc)
    chunk_tags = nltk.pos_tag(words)
    # Create the regexp pattern to find anything starting with a Verb and followed by a Noun
    chunk_rule = """Chunk:{<VB.?>+<NN.?>}"""
    # Creates the regex parser
    regex_parser = nltk.RegexpParser(chunk_rule)
    # Parse all tagged words and saves into a variable
    parsed_sentence = regex_parser.parse(chunk_tags)

    # Code to filter the chunk words from the nltk.Tree
    # (https://stackoverflow.com/questions/58617920/how-print-only-the-string-result-of-the-chunking-with-nltk)
    # Filter only the Chunk matched by the chunk_rules an save into a list tu return only the chunks
    count = 0
    for chunk in parsed_sentence:
        if isinstance(chunk, nltk.tree.Tree):
            if chunk.label() == "Chunk":
                count += 1
    return count


def chunk_adj_noun(text_to_chunk):
    """
    This function creates a chunk of words by using special regexp syntax to find pattern in the text.
    In this case the function uses a regexp rule to find a VB (Verb) followed by a NN (Noun), it it
    finds the patters, it saves each chunk in a list and returns it.
    :param text_to_chunk: Text used to find words + noun in it.
    """
    words = extract_words_from_text(text_to_chunk)
    chunk_tags = nltk.pos_tag(words)
    chunk_rule = """Chunk:{<DT>?<JJ.*>+<NN.*>+}"""
    regex_parser = nltk.RegexpParser(chunk_rule)
    parsed_sentence = regex_parser.parse(chunk_tags)
    count = 0
    for chunk in parsed_sentence:
        if isinstance(chunk, nltk.tree.Tree):
            if chunk.label() == "Chunk":
                count += 1
    return count


# https://readable.com/blog/the-flesch-reading-ease-and-flesch-kincaid-grade-level/
def flesch_reading_ease(total_words, total_sentences, total_syllables):
    """
    This function calculate The Flesch Reading Ease score for a given text.
    The score is between 1 and 100 and 100 is the highest readability score.
    :param total_words: number of total words in the text.
    :param total_sentences: number of total sentences in the text.
    :param total_syllables: number of total syllables in the text.
    :return: return the The Flesch Reading Ease score
    """
    return 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)


def dale_chall_readability_score(percentage_of_difficult_words, average_sentence_length_in_words):
    """
    This function calculates the Dale Chall Readability Score.
    :param percentage_of_difficult_words: percentage of the difficult words in the text.
    :param average_sentence_length_in_words: average length of a sentence in words.
    :return: Dale Chall Readability Score
    """
    if percentage_of_difficult_words <= 5:
        return 0.1579 * percentage_of_difficult_words + 0.0496 * average_sentence_length_in_words
    else:
        return 0.1579 * percentage_of_difficult_words + 0.0496 * average_sentence_length_in_words + 3.6365


def gunning_fog(per_diff_words, average_sentence_length_in_words):
    """
    This function calculate Gunning Fog readability score for a text.
    """
    return 0.4 * (average_sentence_length_in_words + per_diff_words)


def powers_sumner_kearl_grade_level(total_words, total_sentences, total_syllables):
    """
    This function calculates Powers Sumner Kearl grade level of a text by using parameters.
    :param total_words: number of total words in the text
    :param total_sentences: number of total sentences in the text.
    :param total_syllables: number of total syllables in the text.
    :return: Powers Sumner Kearl grade level
    """
    average_sentence_length = total_words / total_sentences
    average_syllables_length = 100 * total_syllables / total_words
    return 0.0778 * average_sentence_length + .0455 * average_syllables_length - 2.2029


def coleman_liau_index(total_words, total_sentences, total_letters):
    """
    This function calculates the Coleman Liau Index of a text.
    :param total_words: number of total words in the text
    :param total_sentences: number of total sentences in the text.
    :param total_letters: number of total letters and digits in the text.
    :return: the Coleman Liau Index
    """
    average_number_of_letters = 100 * total_letters / total_words
    average_number_of_sentences = 100 * total_sentences / total_words
    return 0.0588 * average_number_of_letters - 0.296 * average_number_of_sentences - 15.8


def automated_readability_index(total_words, total_sentences, total_letters):
    """
    This function calculates the Automated Readability Index of a text.
    :param total_words: number of total words in the text
    :param total_sentences: number of total sentences in the text.
    :param total_letters: number of total letters and digits in the text.
    :return: the Automated Readability Index
    """
    return 4.71 * (total_letters / total_words) + 0.5 * (total_words / total_sentences) - 21.43


class Reports:
    """
    A class to represent Report object.
    Report class has one constructor (__init__) to create Report object a
    """

    def __init__(self, id, target, std_error, total_words, total_sentences, total_syllables, ave_syllables,
                 ave_sentence_length, num_verb_noun, num_adj_noun,
                 per_diff_words, flesch_score, dale_score, sumner_level, coleman_index, readability_index,
                 gunning_fog_score, awl_index):
        """Constructor to create Report object by using 13 parameters."""
        self.id = id
        self.target = target
        self.std_error = std_error
        self.total_words = total_words
        self.total_sentences = total_sentences
        self.total_syllables = total_syllables
        self.ave_syllables = ave_syllables
        self.ave_sentence_length = ave_sentence_length
        self.per_diff_words = per_diff_words
        self.flesch_score = flesch_score
        self.dale_score = dale_score
        self.sumner_level = sumner_level
        self.coleman_index = coleman_index
        self.readability_index = readability_index
        self.gunning_fog_score = gunning_fog_score
        self.num_verb_noun = num_verb_noun
        self.num_adj_noun = num_adj_noun
        self.awl_index = awl_index


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    Main function start the program.
    """
    print(datetime.now())
    open_files()
    print(datetime.now())

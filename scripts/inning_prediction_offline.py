"""
Script to be run on the summaries file. It will write to output file if an ordinal adjective is an inning or not.
"""
import codecs
import argparse
from inning_classifier import load_model, get_next_token
from nltk.corpus import stopwords
import logging
logging.basicConfig(level=logging.INFO)

inning_identifier = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                     "7th", "8th", "9th", "10th", "11th", "12th", "13th", "14th", "15th"}
inning_identifier_map = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
                         "eighth": 8, "ninth": 9, "tenth": 10, "7th": 7, "8th": 8, "9th": 9, "10th": 10, "11th": 11,
                         "12th": 12, "13th": 13, "14th": 14, "15th": 15}
additional_check = {"16th", "17th", "18th", "19th", "20th", "21st", "22nd", "23rd", "24th", "25th", "26th", "27th",
                    "28th", "29th", "30th"}

def get_inning(device, model, tokenizer, sent, prev_sent_context, output):
    """

    :param device:
    :param model:
    :param tokenizer:
    :param entry:
    :param sent:
    :param prev_sent_context:
    :param inning_prev_sent:
    :return:
    """
    stops = stopwords.words('english')
    intersected = set(sent).intersection(inning_identifier)
    if len(intersected) > 0:
        # candidate present
        for i in range(len(sent)):
            if sent[i] in inning_identifier and i+1 < len(sent) and sent[i+1] in ["inning", "innings"]:
                pass
            elif "-" in sent[i] and sent[i].split("-")[0] in inning_identifier and sent[i].split("-")[1] == "inning":
                pass
            elif sent[i] in inning_identifier and ((i+1 < len(sent) and (sent[i+1] in [".", ","] or sent[i+1] in stops)) or i+1 == len(sent)):
                # i+1 == len(sent) handles the case such as "Kapler also doubled in a run in the first "; no full stop at the end
                expanded_context = prev_sent_context + sent[:i+1]
                expanded_context = " ".join(expanded_context)
                next_tokens = get_next_token(device, model, tokenizer, expanded_context)
                if "inning" in set(next_tokens) or "innings" in set(next_tokens):
                    output.append((expanded_context, True))
                else:
                    output.append((expanded_context, False))


def process(filename, output_file_name):
    """

    :param filename:
    :param output_file_name:
    :return:
    """
    output = []
    device, model, tokenizer = load_model()
    output_file = codecs.open(output_file_name, mode="w",
                              encoding="utf-8")
    logging.info("filename %s", filename)
    summary_file = codecs.open(filename, mode="r", encoding="utf-8")
    data = summary_file.readlines()
    data = [x.strip() for x in data]
    for entry_index, entry in enumerate(data):
        segments = entry.split(" <segment> ")
        for j, segment in enumerate(segments):
            prev_segment = [] if j == 0 else segments[j - 1].split()
            get_inning(device, model, tokenizer, segment.split(), prev_segment, output)
        for context_result in output:
            output_file.write("\t".join([context_result[0], str(context_result[1])]))
            output_file.write("\n")
        output = []  # reset
        if entry_index % 50 == 0:
            logging.info("entry_index %s", entry_index)
    output_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing to map ordinal adjective to inning or not')
    parser.add_argument('-input_file', type=str,
                        help='path of input valid file')
    parser.add_argument('-output_file', type=str,
                        help='path of map of inning sentence')
    args = parser.parse_args()
    input_file = args.input_file
    process(input_file, args.output_file)

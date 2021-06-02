import os
import json
import argparse
from tokenizer import detokenize, word_tokenize


def process(input_folder, type, output_folder):
    updated_json = open(os.path.join(output_folder, type + ".json"), mode="w", encoding="utf-8")
    file_list = os.listdir(input_folder)
    for filename in file_list:
        if type in filename:
            print("filename", filename)
            json_file = open(os.path.join(input_folder, filename), mode="r", encoding="utf-8")
            data = json.load(json_file)
            upd_trdata = []
            for entry_index, entry in enumerate(data):
                summary = entry['summary']
                summary = detokenize(summary)
                summary = " ".join(word_tokenize(summary))
                upd_entry = entry
                upd_entry['summary'] = summary
                upd_trdata.append(upd_entry)
                if entry_index % 50 == 0:
                    print(entry_index)
            json.dump(upd_trdata, updated_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating retokenized rotowire dataset')
    parser.add_argument('-json_root', type=str,
                        help='path of json root', default=None)
    parser.add_argument('-output_folder', type=str,
                        help='path of output file', default=None)
    parser.add_argument('-dataset_type', type=str,
                        help='type of dataset', default=None)
    args = parser.parse_args()

    process(args.json_root, args.dataset_type, args.output_folder)

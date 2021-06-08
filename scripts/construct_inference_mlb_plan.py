import os
import json
from collections import OrderedDict
from mlb_utils import get_all_paragraph_plans
import argparse


def process(input_folder, type, suffix, output_folder, for_macroplanning=False):
    output_file = open(os.path.join(output_folder, type + "." + suffix + ".pp"), mode="w", encoding="utf-8")
    for filename in os.listdir(input_folder):
        if type in filename:
            print("filename", filename)
            json_file = open(os.path.join(input_folder, filename), mode="r", encoding="utf-8")
            data = json.load(json_file)
            for entry_index, entry in enumerate(data):
                output = get_all_paragraph_plans(entry, for_macroplanning)

                descs_list = list(OrderedDict.fromkeys(output))
                prefix_tokens_=  ["<unk>", "<blank>", "<s>", "</s>"]
                input_template = " <segment> " + " <segment> ".join(descs_list)
                input_template = " ".join(prefix_tokens_) + " " + input_template
                input_template = input_template.replace("  ",
                                                        " ")
                output_file.write(input_template)
                output_file.write("\n")

                if entry_index % 50 == 0:
                    print("entry_index", entry_index)
    output_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating inference plans for mlb dataset')
    parser.add_argument('-json_root', type=str,
                        help='path of json root', default=None)
    parser.add_argument('-suffix', type=str,
                        help='suffix', default=None)
    parser.add_argument('-output_folder', type=str,
                        help='path of output file', default=None)
    parser.add_argument('-dataset_type', type=str,
                        help='type of dataset', default=None)
    parser.add_argument('-for_macroplanning', action="store_true",
                        help='Create the plan in the format for macro planning')
    args = parser.parse_args()

    process(args.json_root, args.dataset_type, args.suffix, args.output_folder, args.for_macroplanning)

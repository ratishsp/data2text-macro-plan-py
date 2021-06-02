import os
import json
from collections import OrderedDict
from roto_utils import get_all_paragraph_plans
import argparse


def process(input_folder, type, output_folder, for_macroplanning, suffix):
    output_file = open(os.path.join(output_folder, type + "."+ suffix + ".pp"), mode="w", encoding="utf-8")
    for filename in os.listdir(input_folder):
        if type in filename:
            print("filename", filename)
            json_file = open(os.path.join(input_folder, filename), mode="r", encoding="utf-8")
            data = json.load(json_file)
            for entry_index, entry in enumerate(data):
                descs = get_all_paragraph_plans(entry, for_macroplanning=for_macroplanning)

                descs_list = list(OrderedDict.fromkeys(descs))
                prefix_tokens_ = ["<unk>", "<blank>", "<s>", "</s>"]
                input_template = "<segment>" + " <segment> ".join(descs_list)
                input_template = " ".join(prefix_tokens_) + " " + input_template
                output_file.write(input_template)
                output_file.write("\n")

                #print("descs", descs)
                if entry_index % 50 == 0:
                    print("entry_index", entry_index)
    output_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating RotoWire inference plans')
    parser.add_argument('-json_root', type=str,
                        help='path of json root', default=None)
    parser.add_argument('-output_folder', type=str,
                        help='path of output file', default=None)
    parser.add_argument('-dataset_type', type=str,
                        help='type of dataset', default=None)
    parser.add_argument('-for_macroplanning', action="store_true",
                        help='Create the plan in the format for macro planning')
    parser.add_argument('-suffix', type=str,
                        help='file name suffix', default=None)
    args = parser.parse_args()

    process(args.json_root, args.dataset_type, args.output_folder, args.for_macroplanning, args.suffix)

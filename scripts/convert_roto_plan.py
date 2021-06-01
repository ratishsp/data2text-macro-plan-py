"""
Script to convert inferred roto plan to input to plan to summary generation model
"""
import argparse


def convert_lines(src_file):
    lines = []
    for line in src_file:
        lines.append(number_segment(line.split()))
    return lines


def number_segment(toks):
    outer_list = []
    sublist = []
    segment_index = 0
    for tok in toks:
        if tok == "<segment>":
            if sublist:
                outer_list.append(" ".join(sublist))
            sublist = ["<segment" + str(segment_index) + ">"]  # add segment id
            segment_index += 1
        elif tok == "</s>":
            continue
        else:
            sublist.append(tok)
    outer_list.append(" ".join(sublist))
    return " ".join(outer_list)


def process(roto_plan_filename, output_filename):
    roto_plan = open(roto_plan_filename, mode='r', encoding='utf-8')
    output_file = open(output_filename, mode='w', encoding='utf-8')
    roto_plan_conv = convert_lines(roto_plan)
    output_file.write("\n".join(roto_plan_conv))
    output_file.write("\n")
    output_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converting inferred macro plan to format used by plan to summary model')
    parser.add_argument('-roto_plan', type=str,
                        help='path of roto plan', default=None)
    parser.add_argument('-output_file', type=str,
                        help='path of output file', default=None)
    args = parser.parse_args()

    process(args.roto_plan, args.output_file)

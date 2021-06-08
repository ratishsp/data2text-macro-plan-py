import argparse


def convert_lines(src_file):
    lines = []
    for index, line in enumerate(src_file):
        line = line.strip()
        if line.startswith("<segment> "):
            line = line[len("<segment> "):]
        if line.endswith(' </s>'):
            line = line[:-len(' </s>')]
        lines.append(line)
    return lines


def process(mlb_plan_filename, output_filename):
    mlb_plan = open(mlb_plan_filename, mode='r', encoding='utf-8')
    output_file = open(output_filename, mode='w', encoding='utf-8')
    mlb_plan_conv = convert_lines(mlb_plan)
    output_file.write("\n".join(mlb_plan_conv))
    output_file.write("\n")
    output_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converting inferred macro plan to format used by plan to summary model')
    parser.add_argument('-mlb_plan', type=str,
                        help='path of mlb plan', default=None)
    parser.add_argument('-output_file', type=str,
                        help='path of output file', default=None)
    args = parser.parse_args()

    process(args.mlb_plan, args.output_file)

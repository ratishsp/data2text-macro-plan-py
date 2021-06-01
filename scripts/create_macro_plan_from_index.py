import argparse


def read_file(file_name):
    src_file = open(file_name, mode='r', encoding='utf-8')
    content = src_file.readlines()
    content = [x.strip() for x in content]
    return content

def split_src_file(src_file):
    lines = []
    for line in src_file:
        segments = line.split()[:4]
        segments = segments + split_segment(line.split()[4:])
        lines.append(segments)
    return lines

def split_segment(toks):
    outer_list = []
    sublist= []
    for tok in toks:
        if tok == "<segment>":
            if sublist:
                outer_list.append(" ".join(sublist))
            sublist = [tok]
        else:
            sublist.append(tok)
    outer_list.append(" ".join(sublist))
    return outer_list

def split_macro_plan_indices(macro_plan_indices):
    return [x.split() for x in macro_plan_indices]

def process(src_file_name, macro_plan_indices_name, output_file_name):
    src_file = read_file(src_file_name)
    src_file = split_src_file(src_file)
    macro_plan_indices = read_file(macro_plan_indices_name)
    macro_plan_indices = split_macro_plan_indices(macro_plan_indices)
    output_file = open(output_file_name, mode="w", encoding="utf-8")

    outputs = []
    for src_file_entry, macro_plan_indices_entry in zip(src_file, macro_plan_indices):
        output = [src_file_entry[int(record)] for record in macro_plan_indices_entry]
        output = " ".join(output)
        outputs.append(output)
    output_file.write("\n".join(outputs))
    output_file.write("\n")
    output_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating Inferred MacroPlan from Indices')
    parser.add_argument('-src_file', type=str,
                        help='path of src file', default=None)
    parser.add_argument('-macro_plan_indices', type=str,
                        help='path of macro plan indices file', default=None)
    parser.add_argument('-output_file', type=str,
                        help='path of output file', default=None)
    args = parser.parse_args()

    process(args.src_file, args.macro_plan_indices, args.output_file)

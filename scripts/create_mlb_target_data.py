import os
import json
from mlb_utils import get_play_by_play_all_entities_inning
from mlb_utils import get_player_line, get_team_line_attributes, match_in_candidate_innings
from mlb_utils import get_inning_side_entities
from mlb_utils import get_all_paragraph_plans
from collections import OrderedDict
import itertools
import logging
import argparse
from mlb_utils import get_ents, get_ordinal_adjective_map, get_inning, sort_files_key, filter_summaries, \
    extract_entities, chunks, get_players_with_map

logging.basicConfig(level=logging.INFO)


def process(input_folder, type, ordinal_adjective_map_file, output_folder):
    paragraph_plans_file = open(os.path.join(output_folder, type + ".pp"), mode="w", encoding="utf-8")
    chosen_paragraph_plan_file = open(os.path.join(output_folder, type + ".macroplan"), mode="w", encoding="utf-8")
    ordinal_adjective_map = get_ordinal_adjective_map(ordinal_adjective_map_file)

    file_list = os.listdir(input_folder)
    sorted_file_list = sorted(file_list, key=sort_files_key)
    seen_output = set()
    test_seen_output = set()
    for filename in sorted_file_list:
        if "valid" in filename or "test" in filename:
            print("test filename", filename)
            json_file = open(os.path.join(input_folder, filename), mode="r", encoding="utf-8")
            data = json.load(json_file)
            for entry_index, entry in enumerate(data):
                test_seen_output.add("_".join(entry["summary"][:50]))

    for filename in sorted_file_list:
        if type in filename:
            print("filename", filename)
            json_file = open(os.path.join(input_folder, filename), mode="r", encoding="utf-8")
            data = json.load(json_file)
            for entry_index, entry in enumerate(data):
                logging.debug("instance %s", entry_index)
                if type == "train":
                    if filter_summaries(entry, seen_output, test_seen_output):
                        continue
                seen_output.add("_".join(entry["summary"][:50]))

                summary = entry["summary"]
                all_ents, players, teams, cities = get_ents(entry)
                players_list, player_team_map = get_players_with_map(entry)
                summ = " ".join(summary)
                segments = summ.split(" *NEWPARAGRAPH* ")
                names_map = {}
                candidate_innings = [
                    len(entry["play_by_play"])]  # initialize with the last inning as it often occurs in the game
                descs = []
                entities_in_game = set()

                for segment in segments:
                    _, sequential_entities = extract_entities(entry, segment.split(), all_ents, players, teams,
                                                                 cities, players_list, names_map)
                    entities_in_game.update([ent[0] for ent in sequential_entities if ent[1] is None])  # adding player entities

                logging.debug("entities_in_game %s", entities_in_game)
                for j, segment in enumerate(segments):
                    logging.debug("segment %s", segment)
                    prev_segment = [] if j == 0 else segments[j - 1].split()
                    innings = get_inning(segment.split(), prev_segment, ordinal_adjective_map)
                    ents, sequential_entities = extract_entities(entry, segment.split(), all_ents, players, teams, cities,
                                                                 players_list, names_map)
                    logging.debug("ents, sequential_entities  %s  %s", ents, sequential_entities)
                    logging.debug("innings %s", innings)
                    candidate_innings.extend([inn[0] for inn in innings])
                    desc = []
                    inning_match = match_in_candidate_innings(entry, candidate_innings[::-1],
                                                              [inn[0] for inn in innings],
                                                              set([ent[0] for ent in sequential_entities
                                                                   if ent[1] is None]))
                    inning_found = False
                    if len(innings) > 0 or inning_match != -1:
                        p_by_p_desc = []
                        total_entities = [ent[0] for ent in sequential_entities]
                        innings_non_repeating = list(OrderedDict.fromkeys([x[0] for x in innings]))
                        for inning in innings_non_repeating:
                            side, _ = get_inning_side_entities(entry, inning, total_entities)
                            if side == "both":
                                inning_found = True
                                for each_side in ["top", "bottom"]:
                                    run_pbyp(entry, inning, p_by_p_desc, each_side)
                            elif side in ["top", "bottom"]:
                                inning_found = True
                                run_pbyp(entry, inning, p_by_p_desc, side)
                        if inning_match != -1:
                            logging.debug("inning_match %s", inning_match)
                            side, _ = get_inning_side_entities(entry, inning_match, total_entities)
                            if side in ["top", "bottom"]:  # ignore both side for inning_match
                                inning_found = True
                                run_pbyp(entry, inning_match, p_by_p_desc, side)

                        p_by_p_desc_in_two = list(chunks(p_by_p_desc, 2))
                        p_by_p_desc_in_two = [" ".join(x) for x in p_by_p_desc_in_two]
                        desc.extend(p_by_p_desc_in_two)
                        descs.extend(desc)

                    if not inning_found:
                        if ents:
                            sequential_entities_upd = list(OrderedDict.fromkeys(sequential_entities))  # get non-repeating list
                            for entity in sequential_entities_upd:
                                if entity[1] is None:  # type is player
                                    desc.append(get_player_line(entry["box_score"], entity[0]))
                                else:  # type is team
                                    desc.append(get_team_line_attributes(entry, entity[0]))

                            descs.append(" ".join(desc))
                    logging.debug("desc %s", desc)
                    logging.debug("=========================")
                descs_non_duplicates = [x for x, _ in itertools.groupby(descs)]
                augmented_paragraph_plans = get_all_paragraph_plans(entry)
                upd_source_sents_list = augmented_paragraph_plans + descs
                descs_list = list(OrderedDict.fromkeys(upd_source_sents_list))
                prefix_tokens_ = ["<unk>", "<blank>", "<s>", "</s>"]
                new_descs_list = prefix_tokens_ + descs_list
                input_template = "<segment> " + " <segment> ".join(descs_list)
                input_template = " ".join(prefix_tokens_) + " " + input_template
                input_template = input_template.replace("  ", " ")
                paragraph_plans_file.write(input_template)
                paragraph_plans_file.write("\n")
                chosen_paragraph_plans = [str(new_descs_list.index(descs_non_duplicates[_paragraph])) for _paragraph in
                                          range(len(descs_non_duplicates))]
                chosen_paragraph_plan_file.write(" ".join(chosen_paragraph_plans))
                chosen_paragraph_plan_file.write("\n")
                if entry_index % 50 == 0:
                    print("entry_index", entry_index)
    chosen_paragraph_plan_file.close()
    paragraph_plans_file.close()


def run_pbyp(entry, inning, p_by_p_desc, side):
    play_by_play_desc, _ = get_play_by_play_all_entities_inning(entry,
                                                             entry[
                                                                 "home_line"][
                                                                 "team_name"],
                                                             entry[
                                                                 "vis_line"][
                                                                 "team_name"],
                                                             inning,
                                                             side)
    p_by_p_desc.append(" ".join(play_by_play_desc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to create dataset for macro planning for mlb')
    parser.add_argument('-ordinal_adjective_map_file', type=str,
                        help='path of ordinal_adjective_map_file', default=None)
    parser.add_argument('-json_root', type=str,
                        help='path of json root', default=None)
    parser.add_argument('-output_folder', type=str,
                        help='path of output file', default=None)
    parser.add_argument('-dataset_type', type=str,
                        help='type of dataset', default=None)
    args = parser.parse_args()
    process(args.json_root, args.dataset_type, args.ordinal_adjective_map_file, args.output_folder)

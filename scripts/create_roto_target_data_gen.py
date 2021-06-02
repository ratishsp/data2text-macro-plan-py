import os
import json
from roto_utils import get_ents, get_players, extract_entities, get_player_line, get_team_line, sort_points
from tokenizer import sent_tokenize
import logging
import numpy as np
from collections import OrderedDict
import argparse
logging.basicConfig(level=logging.INFO)


def process(input_folder, type, output_folder):
    output_file = open(os.path.join(output_folder, type + ".te"), mode="w", encoding="utf-8")
    summary_file = open(os.path.join(output_folder, type + ".su"), mode="w", encoding="utf-8")
    file_list = os.listdir(input_folder)
    for filename in file_list:
        if type in filename:
            print("filename", filename)
            json_file = open(os.path.join(input_folder, filename), mode="r", encoding="utf-8")
            data = json.load(json_file)
            for entry_index, entry in enumerate(data):
                summary = entry['summary']
                all_ents, players, teams, cities = get_ents(entry)
                players_list, player_team_map = get_players(entry)
                home_team_map, vis_team_map, home_seq, vis_seq = sort_points(entry)
                segments = sent_tokenize(summary)

                names_map = {}
                entities_in_game = set()
                candidate_rels = []
                total_entities_in_game = set()
                for sentence in segments:
                    ents, entity_bitarray, sequential_entities = extract_entities(entry, sentence.split(), all_ents, players, teams,
                                                              cities, players_list, names_map)
                    candidate_rels.append((ents, entity_bitarray, sentence, sequential_entities))
                    entities_in_game.update([ent[0] for ent in sequential_entities if ent[1] is None])  # adding player entities
                    total_entities_in_game.update([ent[0] for ent in sequential_entities])  # adding all entities

                if type == "train" and not total_entities_in_game:
                    print("summary", summary)  # omit example where no entity is obtained in the training data
                    continue

                logging.debug("entities_in_game %s", entities_in_game)
                if int(entry["home_line"]["TEAM-PTS"]) > int(entry["vis_line"]["TEAM-PTS"]):
                    home_won = True
                else:
                    home_won = False
                i = 0
                j = 1
                entity_bitarray_i = candidate_rels[i][1]
                updated_sentences = []
                updated_sentences.append([candidate_rels[i][2]])
                sequential_entities = []
                sequential_entities.append(candidate_rels[i][3])
                while i < len(candidate_rels) - 1 and j < len(candidate_rels):
                    entity_bitarray_j = candidate_rels[j][1]
                    if entity_bitarray_j.any() and np.array_equal(np.logical_or(entity_bitarray_i, entity_bitarray_j),
                                                                  entity_bitarray_i):
                        updated_sentences[-1].append(candidate_rels[j][2])
                        sequential_entities[-1].extend(candidate_rels[j][3])
                        j += 1
                    else:
                        updated_sentences.append([candidate_rels[j][2]])
                        sequential_entities.append(candidate_rels[j][3])
                        i = j
                        entity_bitarray_i = candidate_rels[i][1]
                        j = i + 1

                source_sents_list = []
                summary_segments = []
                for entities, sentences in zip(sequential_entities, updated_sentences):
                    source_sents = []
                    entities_upd = list(OrderedDict.fromkeys(entities))
                    for entity in entities_upd:
                        source_sent = None
                        if entity[1] is None:  # player
                            result = None
                            source_sent = get_player_line(entry["box_score"], entity[0], player_team_map, home_seq,
                                            vis_seq, home_team_map, vis_team_map, result, for_macroplanning=False)
                        elif entity[1] == "home":  # home team
                            source_sent = get_team_line(entry["home_line"], "won" if home_won else "lost", entity[1],
                                                        for_macroplanning=False)
                        elif entity[1] == "vis":  # visiting team
                            source_sent = get_team_line(entry["vis_line"], "lost" if home_won else "won", entity[1],
                                                        for_macroplanning=False)
                        source_sents.append(source_sent)

                    source_sents_list.append(" ".join(source_sents))
                    summary_segments.append(" ".join(sentences))

                    logging.debug("segment %s", " ".join(sentences))
                    logging.debug("desc %s", " ".join(source_sents))
                    logging.debug("=========================")

                updated_source_segments = []
                updated_target_segments = []
                for zip_index, source_target in enumerate(zip(source_sents_list, summary_segments)):
                    prefix = "<segment" + str(zip_index) + ">"
                    source_segment = " ".join([prefix, source_target[0]])
                    target_segment = " ".join([prefix, source_target[1]])
                    updated_source_segments.append(source_segment)
                    updated_target_segments.append(target_segment)

                game_summary = " ".join(updated_target_segments)
                input_template = " ".join(updated_source_segments)
                input_template = input_template.replace("  ", " ")
                input_template = input_template.strip()
                output_file.write(input_template)
                output_file.write("\n")
                summary_file.write(game_summary)
                summary_file.write("\n")
                if entry_index % 50 == 0:
                    print("entry_index", entry_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating RotoWire dataset')
    parser.add_argument('-json_root', type=str,
                        help='path of json root', default=None)
    parser.add_argument('-output_folder', type=str,
                        help='path of output file', default=None)
    parser.add_argument('-dataset_type', type=str,
                        help='type of dataset', default=None)
    args = parser.parse_args()

    process(args.json_root, args.dataset_type, args.output_folder)

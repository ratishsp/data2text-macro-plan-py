import re
from collections import OrderedDict
from nltk.corpus import stopwords

team_line1 = "%s <TEAM> %s <CITY> %s <TEAM-RESULT> %s <TEAM_RUNS> %d <TEAM_HITS> %d <TEAM_ERRORS> %d"

pbyp_verbalization_map = {"o": "<PBYP-OUTS>", "b": "<PBYP-BALLS>", "s": "<PBYP-STRIKES>", "b1": "<PBYP-B1>",
                     "b2": "<PBYP-B2>", "b3": "<PBYP-B3>", "batter": "<PBYP-BATTER>", "pitcher": "<PBYP-PITCHER>",
                     "scorers": "<PBYP-SCORERS>", "event": "<PBYP-EVENT>", "event2": "<PBYP-EVENT2>",
                     "fielder_error": "<PBYP-FIELDER-ERROR>", "runs": "<PBYP-RUNS>", "rbi": "<PBYP-RBI>",
                     "error_runs": "<PBYP-ERROR-RUNS>", "top": "<TOP>", "bottom": "<BOTTOM>"}
pitcher_verbalization_map = {"p_bb": "<PITCH-BASE-ON-BALLS>", "p_er": "<EARNED-RUN>", "p_era": "<EARNED-RUN-AVG>",
                     "p_h": "<PITCH-HITS>", "p_hr": "<PITCH-HOME-RUN>", "p_l": "<PITCH-LOSS>",
                     "p_loss": "<PITCH-LOSING-PITCHER>", "p_s": "<PITCH-STRIKES-THROWN>",
                     "p_np": "<PITCH-COUNT>", "p_r": "<PITCH-RUNS>", "p_save": "<PITCH-SAVING-PITCHER>",
                     "p_so": "<PITCH-STRIKE-OUT>", "p_bf": "<PITCH-BATTERS-FACED>", "p_bs": "<PITCH-BLOWN-SAVE>",
                     "p_sv": "<PITCH-SAVE>", "p_w": "<PITCH-WIN>", "p_ip1": "<INNINGS-PITCHED-1>",
                     "p_ip2": "<INNINGS-PITCHED-2>", "p_win": "<PITCH-WINNING-PITCHER>", "p_out": "<PITCH-OUT>"}
batter_verbalization_map = {"h": "<HITS>", "r": "<RUNS>", "hr": "<HOME-RUN>", "ab": "<ATBAT>", "avg": "<AVG>",
                     "rbi": "<RBI>", "cs": "<CAUGHT-STEAL>", "hbp": "<HIT-BY-PITCH>", "a": "<ASSIST>",
                     "bb": "<BASE-ON-BALL>", "e": "<ERROR>", "obp": "<ON-BASE-PCT>", "po": "<PUTOUT>",
                     "pos": "<POS>", "sb": "<STOLEN-BASE>", "sf": "<SAC-FLY>", "slg": "<SLUG>",
                     "so": "<STRIKEOUT>"
                             }
player_verbalization_map = dict(pitcher_verbalization_map, **batter_verbalization_map)
team_verbalization_map = {"team_errors": "<TEAM_ERRORS>", "team_hits": "<TEAM_HITS>", "team_runs": "<TEAM_RUNS>"}
HIGH_NUMBER = -100


def get_team_line_attributes(entry, name):
    """

    :param entry:
    :param name:
    :return:
    """
    if name == entry["home_line"]["team_name"]:
        line = entry["home_line"]
        type = "home"
    elif name == entry["vis_line"]["team_name"]:
        line = entry["vis_line"]
        type = "vis"
    else:
        assert False

    city = line["team_city"]
    name = line["team_name"]
    result = line["result"]
    updated_type = "<"+type.upper()+">"
    team_tup = (updated_type, name, city, result)
    team_line = "%s <TEAM> %s <CITY> %s <TEAM-RESULT> %s"
    sentence1 = team_line %(team_tup)
    other_attributes = []
    attributes = ["team_runs", "team_hits", "team_errors"]
    for attrib in attributes:
        template_string = " ".join([team_verbalization_map[attrib], "%s"])
        other_attributes.append(template_string % line[attrib])
    other_attributes = " ".join(other_attributes)
    team_info = sentence1
    if len(other_attributes) > 0:
        team_info = " ".join([sentence1, other_attributes])
    return team_info


def get_player_line(bs, input_player_name):
    """
    :param bs:
    :param input_player_name:
    :return:
    """
    player_line = "<PLAYER> %s <TEAM> %s <POS> %s"
    player_names = list(bs["full_name"].items())
    player_found = False
    player_info = ""
    for (pid, name) in player_names:
        if name == input_player_name:
            player_tup = (tokenize_initials(name), bs["team"][pid], bs["pos"][pid])
            player_basic_info = player_line %(player_tup)
            other_attributes = []
            for attrib in ["r", "h", "hr", "rbi", "e", "ab", "avg", "cs", "hbp", "bb", "sb", "sf", "so", "a", "po",
                           "p_ip1", "p_ip2", "p_w", "p_l", "p_h", "p_r", "p_er", "p_bb", "p_so", "p_hr", "p_np", "p_s",
                           "p_era", "p_win", "p_loss", "p_save", "p_sv", "p_bf", "p_out", "p_bs"]:
                if bs[attrib][pid] == "N/A":
                    continue
                if attrib in ['sb', 'sf', 'e', 'po', 'a', 'cs', 'hbp', 'hr', 'so', 'bb', "p_hr", "p_sv",
                              "p_bs"] and int(bs[attrib][pid]) == 0:
                    continue
                if attrib in ['avg']  and bs[attrib][pid] == ".000":
                    continue
                template_string = " ".join([player_verbalization_map[attrib], "%s"])
                other_attributes.append(template_string %(bs[attrib][pid]))
            player_other_attributes = " ".join(other_attributes)
            player_info = " ".join([player_basic_info, player_other_attributes])
            player_found = True
    assert player_found
    return player_info


def get_play_by_play_all_entities_inning(entry, home, away, inning, side):
    """
    Method to get play by play for all entities given an inning and side
    :param entry:
    :param home:
    :param away:
    :param inning:
    :param entities:
    :return:
    """
    plays = entry["play_by_play"]
    play_by_play_desc = []
    if str(inning) not in plays:
        return play_by_play_desc

    play_index = 1
    inning_plays = plays[str(inning)][side]
    entities_found = []
    for inning_play in inning_plays:
        other_attrib_desc = get_play_by_play_desc(entities_found, home, away, inning, inning_play, play_index, side)
        other_attrib_desc = " ".join(other_attrib_desc)
        play_index += 1
        play_by_play_desc.append(other_attrib_desc)
    return play_by_play_desc, entities_found


def get_play_by_play_desc(entities_found, home, away, inning, inning_play, play_index,
                          top_bottom):
    inning_line = " ".join(["<INNING> %d", pbyp_verbalization_map[top_bottom], "<BATTING> %s <PITCHING> %s <PLAY> %d"])
    if top_bottom == "top":
        inning_attrib = (inning, away, home, play_index)
    else:
        inning_attrib = (inning, home, away, play_index)
    inning_desc = inning_line % (inning_attrib)
    other_attrib_desc = [inning_desc]
    other_attrib_desc.extend(get_runs_desc(inning_play))
    other_attrib_desc.extend(get_obs_desc(inning_play))
    for attrib in ["batter", "pitcher", "fielder_error"]:
        if attrib in inning_play:
            get_name_desc(attrib, inning_play, other_attrib_desc)
            entities_found.append(inning_play[attrib])
    for attrib in ["scorers", "b2", "b3"]:
        if attrib in inning_play and len(inning_play[attrib]) > 0 and inning_play[attrib][0] != "N/A":
            for baserunner_instance in inning_play[attrib]:
                get_name_desc_entity(attrib, baserunner_instance, other_attrib_desc)
    get_attrib_value_desc("event", inning_play, other_attrib_desc)
    get_attrib_value_desc("event2", inning_play, other_attrib_desc)
    get_team_scores_desc(away, home, inning_play, other_attrib_desc)
    return other_attrib_desc


def get_inning_side_entities(entry, inning, entities):
    """
    Method to get side of the inning described in the summary
    :param entry:
    :param inning:
    :param entities:
    :return:
    """
    plays = entry["play_by_play"]
    if str(inning) not in plays:
        return None, None

    entities_so_far_side = []
    total_non_batting_entities = []
    for top_bottom in ["top", "bottom"]:
        non_batting_entities = []
        inning_plays = plays[str(inning)][top_bottom]
        all_entities_found = set()
        for inning_play in inning_plays:
            entities_found, non_batting = get_entities_in_play(entities, inning_play)
            non_batting_entities.append(non_batting)
            all_entities_found.update(entities_found)
        entities_so_far_side.append(all_entities_found)
        total_non_batting_entities.append(non_batting_entities)

    if not entities_so_far_side[0] and not entities_so_far_side[1]:  # no entities;
        return None, None
    if len(entities_so_far_side[0]) > len(entities_so_far_side[1]):
        return "top", entities_so_far_side[0]
    elif len(entities_so_far_side[0]) < len(entities_so_far_side[1]):
        return "bottom", entities_so_far_side[1]
    else:
        if any(total_non_batting_entities[0]):
            return "top", entities_so_far_side[0]
        elif any(total_non_batting_entities[1]):
            return "bottom", entities_so_far_side[1]
        else:
            return "both", None


def match_in_candidate_innings(entry, innings, summary_innings, entities):
    """
    :param entry:
    :param innings: innings to be searched in
    :param summary_innings: innings mentioned in the summary segment
    :param entities: total entities in the segment
    :return:
    """
    entities_in_summary_inning = set()
    for summary_inning in summary_innings:
        intersection = get_matching_entities_in_inning(entry, summary_inning, entities)
        entities_in_summary_inning.update(intersection)
    entities_not_found = entities.difference(entities_in_summary_inning)
    matched_inning = -1
    if len(entities_not_found) > 1:
        remaining_inings = set(innings).difference(set(summary_innings))
        orderered_remaining_innings = [inning for inning in innings if inning in remaining_inings]
        matched_inning = get_inning_all_entities_set_intersection(entry, orderered_remaining_innings, entities_not_found)
    return matched_inning


def get_entities_in_play(entities, inning_play):
    non_batting = False
    entities_found = set()
    for attrib in ["batter", "pitcher", "fielder_error"]:
        if attrib in inning_play and inning_play[attrib] in entities:
            entities_found.add(inning_play[attrib])
            if attrib in ["fielder_error", "pitcher"]:
                non_batting = True
    for attrib in ["scorers", "b1", "b2", "b3"]:
        if attrib in inning_play and len(inning_play[attrib]) > 0 and inning_play[attrib][0] != "N/A":
            for baserunner_instance in inning_play[attrib]:
                if baserunner_instance in entities:
                    entities_found.add(baserunner_instance)
    return entities_found, non_batting


def get_matching_entities_in_inning(entry, inning, entities):
    """
    Method to get matching entities in an inning with the summary
    :param entry:
    :param inning:
    :param entities:
    :return:
    """
    plays = entry["play_by_play"]
    entities_in_inning = set()
    for top_bottom in ["top", "bottom"]:
        if str(inning) in plays:  # inning may be of a previous match like "He got the victory Friday when he got David Ortiz to hit into an inning-ending double play in the 11th inning"
            inning_plays = plays[str(inning)][top_bottom]
            for inning_play in inning_plays:
                for attrib in ["batter", "pitcher", "fielder_error"]:
                    if attrib in inning_play:
                        entities_in_inning.add(inning_play[attrib])
                for attrib in ["scorers", "b1", "b2", "b3"]:
                    if attrib in inning_play and len(inning_play[attrib]) > 0 and inning_play[attrib][0] != "N/A":
                        for baserunner_instance in inning_play[attrib]:
                            entities_in_inning.add(baserunner_instance)
    intersection = entities_in_inning.intersection(entities)
    return intersection


def get_inning_all_entities_set_intersection(entry, innings, entities):
    """
    Method to get inning
    :param entry:
    :param innings:
    :param entities:
    :return:
    """
    max_intersection = 1
    matched_inning = -1
    for inning in innings:
        intersection = get_matching_entities_in_inning(entry, inning, entities)
        if max_intersection < len(intersection):
            max_intersection = len(intersection)
            matched_inning = inning

    return matched_inning


def get_team_scores_desc(away, home, inning_play, obs_desc):
    if "home_team_runs" in inning_play and "away_team_runs" in inning_play:
        desc = "<TEAM-SCORES> %s %d %s %d" % (
            home, int(inning_play["home_team_runs"]), away, int(inning_play["away_team_runs"]))
        obs_desc.append(desc)


def get_attrib_value_desc(attrib, inning_play, obs_desc):
    if attrib in inning_play:
        desc = " ".join([pbyp_verbalization_map[attrib], "%s"])
        obs_desc.append(desc % (inning_play[attrib]))


def get_name_desc(attrib, inning_play, obs_desc):
    if attrib in inning_play:
        desc = " ".join([pbyp_verbalization_map[attrib], "%s"])
        attrib_value = tokenize_initials(inning_play[attrib])
        obs_desc.append(desc % (attrib_value))


def get_name_desc_entity(attrib, entity_name, obs_desc):
    desc = " ".join([pbyp_verbalization_map[attrib], "%s"])
    attrib_value = tokenize_initials(entity_name)
    obs_desc.append(desc % (attrib_value))


def get_runs_desc(inning_play):
    obs_desc = []
    for attrib in ["runs", "rbi", "error_runs"]:
        if attrib in inning_play and int(inning_play[attrib]) > 0:
            desc = " ".join([pbyp_verbalization_map[attrib], "%d"])
            obs_desc.append(desc % (int(inning_play[attrib])))
    return obs_desc


def get_obs_desc(inning_play):
    obs_desc = []
    for attrib in ["o", "b", "s"]:
        if attrib in inning_play:
            desc = " ".join([pbyp_verbalization_map[attrib], "%d"])
            obs_desc.append(desc % (int(inning_play[attrib])))
    return obs_desc


def tokenize_initials(value):
    attrib_value = re.sub(r"(\w)\.(\w)\.", r"\g<1>. \g<2>.", value)
    return attrib_value


def get_all_paragraph_plans(entry, for_macroplanning=False):
    output = []
    if entry["home_line"]["result"] == "win":
        win_team_name = entry["home_name"]
        lose_team_name = entry["vis_name"]
    else:
        win_team_name = entry["vis_name"]
        lose_team_name = entry["home_name"]
    box_score = entry["box_score"]
    top_home_players = get_players(entry["box_score"], entry["home_name"])
    top_vis_players = get_players(entry["box_score"], entry["vis_name"])
    total_players = top_home_players + top_vis_players
    # teams
    output.append(get_team_line_attributes(entry, win_team_name))
    output.append(get_team_line_attributes(entry, lose_team_name))
    # both teams together
    output.append(" ".join(
        [get_team_line_attributes(entry, win_team_name),
         get_team_line_attributes(entry, lose_team_name)]))
    # opening statement
    for player_index, player in enumerate(total_players):
        output.append(" ".join(
            [get_player_line(box_score, player),
             get_team_line_attributes(entry, win_team_name),
             get_team_line_attributes(entry, lose_team_name)]))
    # each player
    for player_index, player in enumerate(total_players):
        output.append(get_player_line(box_score, player))
    # team and player
    for team, players in zip([entry["home_name"], entry["vis_name"]], [top_home_players, top_vis_players]):
        for player in players:
            desc = " ".join(
                [get_team_line_attributes(entry, team),
                 get_player_line(box_score, player)])
            output.append(desc)
    for inning in range(1, len(entry['home_line']['innings']) + 1):
        for side in ["top", "bottom"]:
            pbyp_desc, entities_found = get_play_by_play_all_entities_inning(entry, entry["home_line"]["team_name"],
                                                                             entry["vis_line"]["team_name"], inning,
                                                                             side)
            desc = []
            if not for_macroplanning:
                entities_found = list(OrderedDict.fromkeys(entities_found))
                desc.append(get_team_line_attributes(entry, entry["home_line"]["team_name"]))
                desc.append(get_team_line_attributes(entry, entry["vis_line"]["team_name"]))
                desc.extend(
                    [get_player_line(entry["box_score"], player_name) for player_name in entities_found])
            desc.extend(pbyp_desc)
            if pbyp_desc:
                output.append(" ".join(desc))
    return output


def get_players(bs, team):
    keys = bs["team"].keys()
    player_lists = []
    for key in keys:
        if bs["pos"][key]== "N/A":
            continue
        player_lists.append((key, get_attrib_value(bs, key, "r"), get_attrib_value(bs, key, "rbi"), get_attrib_value(bs, key, "p_ip1")))
    player_lists.sort(key=lambda x: (-int(x[1]), -int(x[2]), -int(x[3])))
    players = []
    for (pid, _, _, _) in player_lists:
        if bs["team"][pid]  == team:
            players.append(bs["full_name"][pid])
    return players


def get_attrib_value(bs, key, attrib):
    return bs[attrib][key] if key in bs[attrib] and bs[attrib][key] != "N/A" else HIGH_NUMBER


def get_play_by_play_all_entities_inning_gen(entry, home, away, inning, entities, side):
    """
    Method to get play by play for all entities given an inning and side
    :param entry:
    :param home:
    :param away:
    :param inning:
    :param entities:
    :return:
    """
    plays = entry["play_by_play"]
    play_by_play_desc = []
    if str(inning) not in plays:
        return play_by_play_desc, None

    play_index = 1
    inning_plays = plays[str(inning)][side]
    entities_found = []
    for inning_play in inning_plays:
        entity_found, other_attrib_desc = get_play_by_play_desc_gen(entities_found, entities, home,
                                                                away, inning, inning_play, play_index, side)
        other_attrib_desc = " ".join(other_attrib_desc)
        play_index += 1
        if entity_found:
            play_by_play_desc.append(other_attrib_desc)
    return play_by_play_desc, entities_found


def get_play_by_play_desc_gen(entities_found, entities_so_far, home, away, inning, inning_play, play_index,
                          top_bottom):
    entity_found = False
    inning_line = " ".join(["<INNING> %d", pbyp_verbalization_map[top_bottom], "<BATTING> %s <PITCHING> %s <PLAY> %d"])
    if top_bottom == "top":
        inning_attrib = (inning, away, home, play_index)
    else:
        inning_attrib = (inning, home, away, play_index)
    inning_desc = inning_line % (inning_attrib)
    other_attrib_desc = [inning_desc]
    other_attrib_desc.extend(get_runs_desc(inning_play))
    other_attrib_desc.extend(get_obs_desc(inning_play))
    for attrib in ["batter", "pitcher", "fielder_error"]:
        if attrib in inning_play and inning_play[attrib] in entities_so_far:
            entity_found = True
            entities_found.append(inning_play[attrib])
            get_name_desc(attrib, inning_play, other_attrib_desc)
    for attrib in ["scorers", "b2", "b3"]:
        if attrib in inning_play and len(inning_play[attrib]) > 0 and inning_play[attrib][0] != "N/A":
            for baserunner_instance in inning_play[attrib]:
                if baserunner_instance in entities_so_far:
                    entity_found = True
                    entities_found.append(baserunner_instance)
                    get_name_desc_entity(attrib, baserunner_instance, other_attrib_desc)
    get_attrib_value_desc("event", inning_play, other_attrib_desc)
    get_attrib_value_desc("event2", inning_play, other_attrib_desc)
    get_team_scores_desc(away, home, inning_play, other_attrib_desc)
    return entity_found, other_attrib_desc


def get_team_information(thing, home):
    teams = set()
    if home:
        team_type = "home_"
    else:
        team_type = "vis_"

    teams.add(thing[team_type + "name"])
    teams.add(" ".join([thing[team_type + "city"], thing[team_type + "name"]]))

    alternate_names = {"D-backs": "Diamondbacks", "Diamondbacks": "D-backs", "Athletics": "A 's"}
    for key in alternate_names:
        if thing[team_type + "name"] == key:
            teams.add(" ".join([thing[team_type + "city"], alternate_names[key]]))
            teams.add(alternate_names[key])
    return teams


def get_city_information(thing, home):
    cities = set()
    if home:
        team_type = "home_"
    else:
        team_type = "vis_"
    cities.add(thing[team_type + "city"])

    alternate_names = {"Chi Cubs": ["Chicago"], "LA Angels": ["Los Angeles", "LA"], "LA Dodgers": ["Los Angeles", "LA"],
                       "NY Yankees": ["New York", "NY"], "NY Mets": ["New York", "NY"], "Chi White Sox": ["Chicago"]}
    for key in alternate_names:
        if thing[team_type + "city"] == key:
            for val in alternate_names[key]:
                cities.add(val)
    return cities


def get_ents(thing):
    players = set()
    teams = set()
    cities = set()
    teams.update(get_team_information(thing, home=False))
    teams.update(get_team_information(thing, home=True))
    cities.update(get_city_information(thing, home=False))
    cities.update(get_city_information(thing, home=True))
    players.update(thing["box_score"]["full_name"].values())
    players.update(thing["box_score"]["last_name"].values())
    for entset in [players, teams, cities]:
        for k in list(entset):
            pieces = k.split()
            for piece_index in range(len(pieces)):
                entset.add(" ".join(pieces[:piece_index]))
    all_ents = players | teams | cities
    return all_ents, players, teams, cities


def get_team_idx(thing, entname):
    home_teams = set()
    home_cities = set()
    vis_teams = set()
    vis_cities = set()
    vis_teams.update(get_team_information(thing, home=False))
    home_teams.update(get_team_information(thing, home=True))
    vis_cities.update(get_city_information(thing, home=False))
    home_cities.update(get_city_information(thing, home=True))

    for entset in [home_teams, home_cities, vis_teams, vis_cities]:
        for k in list(entset):
            pieces = k.split()
            for piece_index in range(len(pieces)):
                entset.add(" ".join(pieces[:piece_index]))
    if entname in home_teams or entname in home_cities:
        team_name = (thing["home_name"], "home")
    elif entname in vis_teams or entname in vis_cities:
        team_name = (thing["vis_name"], "vis")
    else:
        assert False

    return team_name


def get_player_idx(players, entname, names_map, prev_word, prev_second_word):
    keys = []
    matched_player_name = None
    for index, v in enumerate(players):
        if entname == v[0]:
            names_map[v[1]] = entname
            matched_player_name = v[0]  # (full name, last name, first name)
    if len(keys) == 0:
        for index, v in enumerate(players):  # handling special cases
            if prev_second_word + prev_word == v[2] and entname == v[1]:  # handling tokenization of C.J as C. J.
                matched_player_name = v[0]
                names_map[v[1]] = v[0]
    if len(keys) == 0:
        for index, v in enumerate(players):
            if entname in names_map:
                if names_map[entname] == v[0]:
                    matched_player_name = v[0]  # matching for coreferent mention of second name
            elif entname == v[1]:  # matching second name
                matched_player_name = v[0]
        if len(keys) > 1:
            print("prev_word", prev_word)
            print("more than one match", entname)
            matched_player_name = None
    return matched_player_name


def get_ordinal_adjective_map(ordinal_adjective_map_file_name):
    ordinal_adjective_map_file = open(ordinal_adjective_map_file_name, mode="r", encoding="utf-8")
    ordinal_adjective_map_lines = ordinal_adjective_map_file.readlines()
    ordinal_adjective_map_lines = [line.strip() for line in ordinal_adjective_map_lines]
    ordinal_adjective_map = {}
    for line in ordinal_adjective_map_lines:
        ordinal_adjective_map[line.split("\t")[0]] = line.split("\t")[1]
    return ordinal_adjective_map


def get_inning(sent, prev_sent_context, ordinal_adjective_map):
    inning_identifier = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                         "7th", "8th", "9th", "10th", "11th", "12th", "13th", "14th", "15th"}
    inning_identifier_map = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
                             "eighth": 8, "ninth": 9, "tenth": 10, "7th": 7, "8th": 8, "9th": 9, "10th": 10, "11th": 11,
                             "12th": 12, "13th": 13, "14th": 14, "15th": 15}
    stops = stopwords.words('english')
    innings = []
    upd_sent = " ".join(sent)
    upd_sent = upd_sent.replace("-", " ").split()  # handles cases such as these: pitched out of a second-inning jam
    intersected = set(upd_sent).intersection(inning_identifier)
    if len(intersected) > 0:
        # candidate present
        for i in range(len(sent)):
            if sent[i] in inning_identifier and i+1 < len(sent) and sent[i+1] in ["inning", "innings"]:
                innings.append((inning_identifier_map[sent[i]], i))
            elif "-" in sent[i]  and sent[i].split("-")[0] in inning_identifier and sent[i].split("-")[1]  == "inning":
                innings.append((inning_identifier_map[sent[i].split("-")[0]], i))
            elif (" ".join(sent[:i]).endswith("in the") or " ".join(sent[:i]).endswith("in the top of the") or " ".join(
                    sent[:i]).endswith("in the bottom of the")) and sent[i] in inning_identifier and (
                    (i + 1 < len(sent) and (sent[i + 1] in [".", ","] or sent[i + 1] in stops)) or i + 1 == len(sent)):
                innings.append((inning_identifier_map[sent[i]], i))
            elif sent[i] in inning_identifier and ((i+1 < len(sent) and (sent[i+1] in [".", ","] or sent[i+1] in stops)) or i+1 == len(sent)):
                # i+1 == len(sent) handles the case such as "Kapler also doubled in a run in the first "; no full stop at the end
                expanded_context = prev_sent_context + sent[:i+1]
                expanded_context = " ".join(expanded_context)
                assert expanded_context in ordinal_adjective_map
                if ordinal_adjective_map[expanded_context] == "True":
                    innings.append((inning_identifier_map[sent[i]], i))
    return innings


def sort_files_key(x):
    if "train" in x:
        file_index = int(x[5:7].strip("."))  # get the index of the train file
    else:
        file_index = -1  # valid and test
    return file_index


def filter_summaries(summary_entry, seen_output, test_seen_output):
    match_words = {"rain", "rains", "rained", "snow"}
    filter = False
    if len(summary_entry["summary"]) < 100:
        filter = True
    elif 100 < len(summary_entry["summary"]) < 300:
        if len(match_words.intersection(set(summary_entry["summary"]))) > 0:
            filter = True
    elif "_".join(summary_entry["summary"][:50]) in seen_output:  # retaining only one instance
        filter = True
    elif "_".join(summary_entry["summary"][:50]) in test_seen_output:  # retaining only one instance
        filter = True
    return filter


def extract_entities(entry, sent, all_ents, players=None, teams=None, cities=None, players_list=None, names_map=None):
    sent_ents = []
    sequential_entities = []
    matched_player_name = None
    team_name = None
    i = 0
    while i < len(sent):
        if sent[i] in all_ents:  # finds longest spans
            j = 1
            while i + j <= len(sent) and " ".join(sent[i:i + j]) in all_ents:
                j += 1
            candidate_entity = " ".join(sent[i:i + j - 1])
            if candidate_entity in teams or candidate_entity in cities:
                team_name = get_team_idx(entry, candidate_entity)
            elif candidate_entity in players:
                matched_player_name = get_player_idx(players_list, candidate_entity, names_map, sent[i - 1],
                                                     sent[i - 2])
            if matched_player_name is not None or team_name is not None:
                sent_ents.append((i, i + j - 1, candidate_entity))
                if matched_player_name is not None:
                    sequential_entities.append((matched_player_name, None))
                    matched_player_name = None
                elif team_name is not None:
                    sequential_entities.append(team_name)
                    team_name = None
            i += j - 1
        else:
            i += 1
    return sent_ents, sequential_entities


def chunks(input_list, chunk_size):
    for index in range(0, len(input_list), chunk_size):
        yield input_list[index: index + chunk_size]


def get_players_with_map(entry):
    player_team_map = {}
    bs = entry["box_score"]
    full_names = bs["full_name"]
    first_names = bs["first_name"]
    second_names = bs["last_name"]
    teams = bs["team"]
    players = []
    for k in full_names:
        players.append((full_names[k], second_names[k], first_names[k]))
        player_team_map[full_names[k]] = teams[k]
    return players, player_team_map

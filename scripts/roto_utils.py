import re
import numpy as np
player_line = "<PLAYER> %s <TEAM> %s <POS> %s <RANK> %s <MIN> %d <PTS> %d <FG> %d %d %d <FG3> %d %d %d " \
              "<FT> %d %d %d <REB> %d <AST> %d <STL> %s " \
              "<BLK> %d <DREB> %d <OREB> %d <TO> %d"
NUM_PLAYERS = 13

#################################
# template gen
def tokenize_initials(value):
    attrib_value = re.sub(r"(\w)\.(\w)\.", r"\g<1>. \g<2>.", value)
    return attrib_value


def handle_na(value):
    return "0" if value == "N/A" else value


def get_player_line(bs, input_player_name, player_team_map, home_player_pts, vis_player_pts, home_team_map, vis_team_map, result, for_macroplanning=True):
    if for_macroplanning:
        rank = get_rank(input_player_name, home_player_pts, vis_player_pts, home_team_map, vis_team_map, result)
    else:
        rank = get_rank_for_generation(input_player_name, home_player_pts, vis_player_pts, home_team_map, vis_team_map, result)
    player_names = list(bs["PLAYER_NAME"].items())
    player_found = False
    player_tup = None
    for (pid, name) in player_names:
        if name == input_player_name:
            player_tup = (tokenize_initials(name), player_team_map[input_player_name],
                          bs["START_POSITION"][pid],
                          rank,
                          int(handle_na(bs["MIN"][pid])),
                          int(handle_na(bs["PTS"][pid])),
                          int(handle_na(bs["FGM"][pid])),
                          int(handle_na(bs["FGA"][pid])), int(handle_na(bs["FG_PCT"][pid])),
                          int(handle_na(bs["FG3M"][pid])), int(handle_na(bs["FG3A"][pid])),
                          int(handle_na(bs["FG3_PCT"][pid])),
                          int(handle_na(bs["FTM"][pid])), int(handle_na(bs["FTA"][pid])),
                          int(handle_na(bs["FT_PCT"][pid])),
                          int(handle_na(bs["REB"][pid])), int(handle_na(bs["AST"][pid])),
                          int(handle_na(bs["STL"][pid])),
                          int(handle_na(bs["BLK"][pid])), int(handle_na(bs["DREB"][pid])),
                          int(handle_na(bs["OREB"][pid])), int(handle_na(bs["TO"][pid])))
            player_found = True
            break
    assert player_found
    return player_line % (player_tup)


def get_team_line(line, result, type, for_macroplanning=True):
    if for_macroplanning:
        team_line = "<TEAM> %s <CITY> %s <TEAM-RESULT> %s <TEAM-PTS> %d <WINS-LOSSES> %d %d <QTRS> %d %d %d %d " \
                    "<TEAM-AST> %d <3PT> %d <TEAM-FG> %d <TEAM-FT> %d <TEAM-REB> %d <TEAM-TO> %d"
    else:
        team_line = "%s <TEAM> %s <CITY> %s <TEAM-RESULT> %s <TEAM-PTS> %d <WINS-LOSSES> %d %d <QTRS> %d %d %d %d " \
                    "<TEAM-AST> %d <3PT> %d <TEAM-FG> %d <TEAM-FT> %d <TEAM-REB> %d <TEAM-TO> %d"
    city = line["TEAM-CITY"]
    name = line["TEAM-NAME"]
    wins = int(line["TEAM-WINS"])
    losses = int(line["TEAM-LOSSES"])
    pts = int(line["TEAM-PTS"])
    ast = int(line["TEAM-AST"])
    three_pointers_pct = int(line["TEAM-FG3_PCT"])
    field_goals_pct = int(line["TEAM-FG_PCT"])
    free_throws_pct = int(line["TEAM-FT_PCT"])
    pts_qtr1 = int(line["TEAM-PTS_QTR1"])
    pts_qtr2 = int(line["TEAM-PTS_QTR2"])
    pts_qtr3 = int(line["TEAM-PTS_QTR3"])
    pts_qtr4 = int(line["TEAM-PTS_QTR4"])
    reb = int(line["TEAM-REB"])
    tov = int(line["TEAM-TOV"])
    team_tup = (name, city, result, pts, wins, losses, pts_qtr1, pts_qtr2, pts_qtr3, pts_qtr4, ast,
                three_pointers_pct, field_goals_pct, free_throws_pct, reb, tov)
    if not for_macroplanning:
        updated_type = "<" + type.upper() + ">"
        team_tup = (updated_type, name, city, result, pts, wins, losses, pts_qtr1, pts_qtr2, pts_qtr3, pts_qtr4, ast,
                    three_pointers_pct, field_goals_pct, free_throws_pct, reb, tov)

    return team_line %(team_tup)


def sort_points(entry):
    home_team_map = {}
    vis_team_map = {}
    bs = entry["box_score"]
    nplayers = 0
    for k,v in bs["PTS"].items():
        nplayers += 1

    num_home, num_vis = 0, 0
    home_pts = []
    vis_pts = []
    for i in range(nplayers):
        player_city = entry["box_score"]["TEAM_CITY"][str(i)]
        player_name = bs["PLAYER_NAME"][str(i)]
        if player_city == entry["home_city"]:
            if num_home < NUM_PLAYERS:
                home_team_map[player_name] = bs["PTS"][str(i)]
                if bs["PTS"][str(i)] != "N/A":
                    home_pts.append(int(bs["PTS"][str(i)]))
                num_home += 1
        else:
            if num_vis < NUM_PLAYERS:
                vis_team_map[player_name] = bs["PTS"][str(i)]
                if bs["PTS"][str(i)] != "N/A":
                    vis_pts.append(int(bs["PTS"][str(i)]))
                num_vis += 1
    if entry["home_city"] == entry["vis_city"] and entry["home_city"] == "Los Angeles":
        num_home, num_vis = 0, 0
        for i in range(nplayers):
            player_name = bs["PLAYER_NAME"][str(i)]
            if num_vis < NUM_PLAYERS:
                vis_team_map[player_name] = bs["PTS"][str(i)]
                if bs["PTS"][str(i)] != "N/A":
                    vis_pts.append(int(bs["PTS"][str(i)]))
                num_vis += 1
            elif num_home < NUM_PLAYERS:
                home_team_map[player_name] = bs["PTS"][str(i)]
                if bs["PTS"][str(i)] != "N/A":
                    home_pts.append(int(bs["PTS"][str(i)]))
                num_home += 1
    home_seq = sorted(home_pts, reverse=True)
    vis_seq = sorted(vis_pts, reverse=True)
    return home_team_map, vis_team_map, home_seq, vis_seq


def get_rank(player_name, home_seq, vis_seq, home_team_map, vis_team_map, result):
    if player_name in home_team_map:
        if home_team_map[player_name] == 'N/A':
            rank = result.upper()+'-DIDNTPLAY'
        else:
            rank = result.upper() + '-'+str(home_seq.index(int(home_team_map[player_name])))
    elif player_name in vis_team_map:
        if vis_team_map[player_name] == 'N/A':
            rank = result.upper() + '-DIDNTPLAY'
        else:
            rank = result.upper() + '-'+str(vis_seq.index(int(vis_team_map[player_name])))
    else:
        print("player_name", player_name)
        assert False
    return rank


def get_rank_for_generation(player_name, home_seq, vis_seq, home_team_map, vis_team_map, result):
    if player_name in home_team_map:
        if home_team_map[player_name] == 'N/A':
            rank = 'HOME-DIDNTPLAY'
        else:
            rank = 'HOME-'+str(home_seq.index(int(home_team_map[player_name])))
    elif player_name in vis_team_map:
        if vis_team_map[player_name] == 'N/A':
            rank = 'VIS-DIDNTPLAY'
        else:
            rank = 'VIS-'+str(vis_seq.index(int(vis_team_map[player_name])))
    else:
        print("player_name", player_name)
        assert False
    return rank


def sort_player_and_points(entry):
    bs = entry["box_score"]
    nplayers = 0
    for k,v in bs["PTS"].items():
        nplayers += 1

    num_home, num_vis = 0, 0
    home_pts = []
    vis_pts = []
    for i in range(nplayers):
        player_city = entry["box_score"]["TEAM_CITY"][str(i)]
        player_name = bs["PLAYER_NAME"][str(i)]
        if player_city == entry["home_city"]:
            if num_home < NUM_PLAYERS:
                if bs["PTS"][str(i)] != "N/A":
                    home_pts.append((player_name, int(bs["PTS"][str(i)])))
                else:
                    home_pts.append((player_name, -1))
                num_home += 1
        else:
            if num_vis < NUM_PLAYERS:
                if bs["PTS"][str(i)] != "N/A":
                    vis_pts.append((player_name, int(bs["PTS"][str(i)])))
                else:
                    vis_pts.append((player_name, -1))
                num_vis += 1
    if entry["home_city"] == entry["vis_city"] and entry["home_city"] == "Los Angeles":
        num_home, num_vis = 0, 0
        for i in range(nplayers):
            player_name = bs["PLAYER_NAME"][str(i)]
            if num_vis < NUM_PLAYERS:
                if bs["PTS"][str(i)] != "N/A":
                    vis_pts.append((player_name, int(bs["PTS"][str(i)])))
                else:
                    vis_pts.append((player_name, -1))
                num_vis += 1
            elif num_home < NUM_PLAYERS:
                if bs["PTS"][str(i)] != "N/A":
                    home_pts.append((player_name, int(bs["PTS"][str(i)])))
                else:
                    home_pts.append((player_name, -1))
                num_home += 1
    home_seq = sorted(home_pts, key=lambda x: -x[1])
    vis_seq = sorted(vis_pts, key=lambda x: -x[1])
    return home_seq, vis_seq


##########################
#construct_plan

def get_all_paragraph_plans(entry, for_macroplanning=True):
    box_score_ = entry["box_score"]
    if int(entry["home_line"]["TEAM-PTS"]) > int(entry["vis_line"]["TEAM-PTS"]):
        home_won = True
    else:
        home_won = False
    descs = [""]  # empty segment
    desc = []
    if home_won:
        home_line = get_team_line(entry["home_line"], "won", "home", for_macroplanning=for_macroplanning)
        vis_line = get_team_line(entry["vis_line"], "lost", "vis", for_macroplanning=for_macroplanning)
        desc.append(home_line)
        desc.append(vis_line)
    else:
        vis_line = get_team_line(entry["vis_line"], "won", "vis", for_macroplanning=for_macroplanning)
        home_line = get_team_line(entry["home_line"], "lost", "home", for_macroplanning=for_macroplanning)
        desc.append(vis_line)
        desc.append(home_line)
    descs.append(" ".join(desc))  # include line discussing both teams
    descs.extend(desc)  # include individual teams
    players_list, player_team_map = get_players(entry)
    home_team_map, vis_team_map, home_player_pts, vis_player_pts = sort_points(entry)
    home_player_seq, vis_player_seq = sort_player_and_points(entry)
    desc = []
    for player_name, _ in home_player_seq + vis_player_seq:
        result = get_result_player(player_name, entry["home_city"] + " " + entry["home_line"]["TEAM-NAME"],
                                   entry["vis_city"] + " " + entry["vis_line"]["TEAM-NAME"], home_won, player_team_map)
        player_line = get_player_line(box_score_, player_name, player_team_map, home_player_pts,
                                      vis_player_pts, home_team_map, vis_team_map, result, for_macroplanning=for_macroplanning)
        desc.append(player_line)
    descs.extend(desc)
    # add paragraph plan for team and players
    for line, player_seq in zip([home_line, vis_line], [home_player_seq, vis_player_seq]):

        desc = add_team_and_player_desc(entry, line, player_seq, home_player_pts, home_team_map,
                                        vis_player_pts, vis_team_map, player_team_map, home_won, for_macroplanning=for_macroplanning)
        descs.extend(desc)

    # add paragraph plan for pairs of players in the same team
    desc = []
    for player_seq in [home_player_seq, vis_player_seq]:
        for player_index, (player_name, _) in enumerate(player_seq):
            result = get_result_player(player_name, entry["home_city"] + " " + entry["home_line"]["TEAM-NAME"],
                                   entry["vis_city"] + " " + entry["vis_line"]["TEAM-NAME"],
                                       home_won, player_team_map)
            player_line = get_player_line(box_score_, player_name, player_team_map, home_player_pts,
                                          vis_player_pts, home_team_map, vis_team_map, result, for_macroplanning=for_macroplanning)
            for player_name_2, _ in player_seq[player_index + 1:]:
                player_line_2 = get_player_line(box_score_, player_name_2, player_team_map, home_player_pts,
                                                vis_player_pts, home_team_map, vis_team_map, result, for_macroplanning=for_macroplanning)
                desc.append(" ".join([player_line, player_line_2]))
    descs.extend(desc)
    return descs


def add_team_and_player_desc(entry, line, player_seq, home_seq, home_team_map, vis_seq,
                             vis_team_map, player_team_map, home_won, for_macroplanning=True):
    desc = []
    box_score_ = entry["box_score"]
    for player_name, _ in player_seq:
        result = get_result_player(player_name, entry["home_city"] + " " + entry["home_line"]["TEAM-NAME"],
                                   entry["vis_city"] + " " + entry["vis_line"]["TEAM-NAME"], home_won, player_team_map)
        player_line = get_player_line(box_score_, player_name, player_team_map, home_seq, vis_seq, home_team_map,
                                      vis_team_map, result, for_macroplanning=for_macroplanning)
        desc.append(" ".join([line, player_line]))
    return desc


def get_result_player(player_name, home_name, vis_name, home_won, player_team_map):
    if player_team_map[player_name] == home_name:
        result = "won" if home_won else "lost"
    elif player_team_map[player_name] == vis_name:
        result = "lost" if home_won else "won"
    else:
        assert False
    return result


#############################
# create target data
def get_team_information(thing, home):
    teams = set()
    if home:
        team_type = "home_"
    else:
        team_type = "vis_"

    teams.add(thing[team_type + "name"])
    teams.add(thing[team_type + "line"]["TEAM-NAME"])
    teams.add(thing[team_type + "city"] + " " + thing[team_type + "name"])
    teams.add(thing[team_type + "city"] + " " + thing[team_type + "line"]["TEAM-NAME"])
    assert thing[team_type + "line"]["TEAM-NAME"] == thing[team_type + "name"]
    alternate_names = {"76ers": "Sixers", "Mavericks": "Mavs", "Cavaliers": "Cavs", "Timberwolves": "Wolves"}
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
    alternate_names = {"LA": "Los Angeles", "Los Angeles": "LA"}
    for key in alternate_names:
        if thing[team_type + "city"] == key:
            cities.add(alternate_names[key])
    return cities


def get_ents(thing):
    players = set()
    teams = set()
    cities = set()

    teams.update(get_team_information(thing, home=False))
    teams.update(get_team_information(thing, home=True))

    # sometimes team_city is different
    cities.update(get_city_information(thing, home=False))
    cities.update(get_city_information(thing, home=True))
    cities.update(thing["box_score"]["TEAM_CITY"].values())
    players.update(thing["box_score"]["PLAYER_NAME"].values())
    players.update(thing["box_score"]["SECOND_NAME"].values())
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

    # sometimes team_city is different
    home_cities.update(get_city_information(thing, home=True))
    vis_cities.update(get_city_information(thing, home=False))
    for entset in [home_teams, home_cities, vis_teams, vis_cities]:
        for k in list(entset):
            pieces = k.split()
            for piece_index in range(len(pieces)):
                entset.add(" ".join(pieces[:piece_index]))
    team_index = np.full(28, False)
    if entname in home_teams or entname in home_cities:
        team_index[26] = True
        team_name = (thing["home_line"]["TEAM-NAME"], "home")
    elif entname in vis_teams or entname in vis_cities:
        team_index[27] = True
        team_name = (thing["vis_line"]["TEAM-NAME"], "vis")
    else:
        assert False

    return team_index, team_name


def get_players(entry):
    player_team_map = {}
    bs = entry["box_score"]
    nplayers = 0
    home_players, vis_players = [], []
    for k,v in entry["box_score"]["PTS"].items():
        nplayers += 1

    num_home, num_vis = 0, 0
    for i in range(nplayers):
        player_city = entry["box_score"]["TEAM_CITY"][str(i)]
        player_name = bs["PLAYER_NAME"][str(i)]
        second_name = bs["SECOND_NAME"][str(i)]
        first_name = bs["FIRST_NAME"][str(i)]
        if player_city == entry["home_city"]:
            if len(home_players) < NUM_PLAYERS:
                home_players.append((player_name, second_name,
                                     first_name))
                player_team_map[player_name] = " ".join(
                    [player_city, entry["home_line"]["TEAM-NAME"]])
                num_home += 1
        else:
            if len(vis_players) < NUM_PLAYERS:
                vis_players.append((player_name, second_name,
                                    first_name))
                player_team_map[player_name] = " ".join(
                    [player_city, entry["vis_line"]["TEAM-NAME"]])
                num_vis += 1

    if entry["home_city"] == entry["vis_city"] and entry["home_city"] == "Los Angeles":
        home_players, vis_players = [], []
        num_home, num_vis = 0, 0
        for i in range(nplayers):
            player_name = bs["PLAYER_NAME"][str(i)]
            second_name = bs["SECOND_NAME"][str(i)]
            first_name = bs["FIRST_NAME"][str(i)]
            if len(vis_players) < NUM_PLAYERS:
                vis_players.append((player_name, second_name,
                                    first_name))
                player_team_map[player_name] = " ".join(
                    ["Los Angeles", entry["vis_line"]["TEAM-NAME"]])
                num_vis += 1
            elif len(home_players) < NUM_PLAYERS:
                home_players.append((player_name, second_name,
                                     first_name))
                player_team_map[player_name] = " ".join(
                    ["Los Angeles", entry["home_line"]["TEAM-NAME"]])
                num_home += 1

    players = []
    for ii, player_list in enumerate([home_players, vis_players]):
        for j in range(NUM_PLAYERS):
            players.append(player_list[j] if j < len(player_list) else ("N/A","N/A","N/A"))
    return players, player_team_map


def get_player_idx(entry, players, entname, names_map, prev_word, prev_second_word):
    bs = entry["box_score"]
    keys = []
    matched_player_name = None
    for index, v in enumerate(players):
        if entname == v[0]:
            keys.append(str(index))
            if entname in ["Marcus Morris", "Markieff Morris", "Nene"]:
                names_map[v[2]] = entname  # handling match for Marcus, Markieff, Nene; matching for first name
            elif entname == "Michael Carter-Williams":
                names_map["Williams"] = entname
            else:
                names_map[v[1]] = entname  # matching for second name
            matched_player_name = v[0]
    if len(keys) == 0:
        for index, v in enumerate(players):  # handling special cases
            if prev_second_word + prev_word == v[2] and entname == v[1]:  # handling tokenization of C.J. as C. J.
                keys.append(str(index))
                matched_player_name = v[0]
                names_map[v[1]] = v[0]  # matching for second name
            elif prev_word == "Louis" and entname == "Williams" and v[0] == "Lou Williams":
                keys.append(str(index))
                matched_player_name = v[0]
            elif prev_second_word == "J." and prev_word == "R." and entname == "Smith" and v[0] == "JR Smith":
                keys.append(str(index))
                matched_player_name = v[0]
            elif prev_word == "Kelvin" and entname == "Martin" and v[0] == "Kevin Martin":
                keys.append(str(index))
                matched_player_name = v[0]
            elif prev_word == "Steph" and entname == "Curry" and v[0] == "Stephen Curry":
                keys.append(str(index))
                matched_player_name = v[0]
            elif prev_word == "James" and entname == "Ennis" and v[0] == "James Ennis III":
                keys.append(str(index))
                matched_player_name = v[0]
            elif entname == "Williams" and entname in names_map and names_map[entname] == "Michael Carter-Williams" and \
                            v[0] == "Michael Carter-Williams":
                keys.append(str(index))
                matched_player_name = v[0]
    if len(keys) == 0:
        for index, v in enumerate(players):
            if entname in names_map:
                if names_map[entname] == v[0]:
                    keys.append(str(index))
                    matched_player_name = v[0]
            elif entname == v[1]:
                keys.append(str(index))
                matched_player_name = v[0]
        if len(keys) > 1:
            print("prev_word", prev_word)
            print("more than one match", entname + " : " + str(bs["PLAYER_NAME"].values()))
            keys = []
            matched_player_name = None
    if len(keys) == 0:
        for index, v in enumerate(players):
            if entname == v[2]:
                keys.append(str(index))
                matched_player_name = v[0]
        if len(keys) > 1:
            print("multiple matches found for first name", entname + " : " + str(bs["PLAYER_NAME"].values()))
            keys = []
            matched_player_name = None
    player_index = np.full(28, False)
    if len(keys) > 0:
        player_index[int(keys[0])] = True
    return player_index, matched_player_name


def extract_entities(entry, sent, all_ents, players=None, teams=None, cities=None, players_list=None, names_map=None):
    sent_ents = []
    combined_entity_array = np.full(28, False)
    sequential_entities = []
    matched_player_name = None
    team_name = None
    i = 0
    while i < len(sent):
        if sent[i] in all_ents:
            j = 1
            while i+j <= len(sent) and " ".join(sent[i:i+j]) in all_ents:
                j += 1
            candidate_entity = " ".join(sent[i:i + j - 1])
            if candidate_entity in players:
                pidx, matched_player_name = get_player_idx(entry, players_list, candidate_entity, names_map, sent[i - 1], sent[i - 2])
            elif candidate_entity in teams:
                pidx, team_name = get_team_idx(entry, candidate_entity)
            elif candidate_entity in cities:
                pidx, team_name = get_team_idx(entry, candidate_entity)
            if pidx is not None:
                sent_ents.append((i, i + j - 1, candidate_entity))
                combined_entity_array = np.logical_or(combined_entity_array, pidx)
                if matched_player_name is not None:
                    sequential_entities.append((matched_player_name, None))
                    matched_player_name = None
                elif team_name is not None:
                    sequential_entities.append(team_name)
                    team_name = None
            i += j-1
        else:
            i += 1
    return sent_ents, combined_entity_array, sequential_entities


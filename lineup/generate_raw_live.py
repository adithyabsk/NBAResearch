#!/usr/bin/env python

from copy import deepcopy
import itertools
import multiprocessing
from multiprocessing import Pool
import time

import numpy as np
import pandas as pd
from tqdm import tqdm


def set_pandas_options(max_columns=None, max_rows=None):
    pd.set_option("display.max_columns", max_columns)
    pd.set_option("display.max_rows", max_rows)


def patch_raw_data():
    """Fix all errors found in raw data
    Error 1:
        Game_id: 20300231
        Description: A substitution was recorded incorrectly in on_court_df. The
                     player (2549) was substituted onto the other team. Change 
                     on_court_df record to match play by play data. Events 
                     493, 494, 495, 497 need to be fixed.
    Error 2:
        Game_id: 20500121
        Description: Erraneous free throw recorded after the end of the game. Remove
                     it. Event 541
    Error 3:
        Game_id: 21201224
        Descripton: Subbing error, wrong player id. (203191) --> (203006) on events
                    518-540
    Error 4:
        Game_id: 21400610
        Description: Subbing error, wrong player id. (203254) --> (203210). All valid
                     events.
    Error 5:
        Game_id: 21400947
        Description: Entire 2nd quarter starting players is incorrect and missing. 
                     Fixed using assumptions from play by play data.
    Error 6:
        Game_id: 21601203
        Description: Not exactly an error in the data but the source is incomplete.
                     A player is subbed in just to take a flagrant free throw but
                     they don't ever become "active" so they don't show up in the 
                     players list. Add and remove them from "active" to fix this.
    Error 7:
        Game_id: 21300140
        Description: Wrong substitution (2430) --> (203513) Events 527-533. Also
                     drop event 535, happens after end of quarter according to
                     Wall clock.
    """

    on_court_df = pd.read_csv(
        "../data/Hackathon_Players_on_Court.csv", compression="gzip"
    )
    pbp_df = pd.read_csv("../data/Hackathon_Play_by_Play.csv", compression="gzip")
    
    # Error 1
    on_court_df.loc[(on_court_df['Game_id'] == 20300231)
                    & (on_court_df['Event_Num'].isin([493, 494, 495, 497]))
                    & (on_court_df['Team_id'] == 1610612743), 'Player5'] = 2403
    on_court_df.loc[(on_court_df['Game_id'] == 20300231)
                    & (on_court_df['Event_Num'].isin([493, 494, 497]))
                    & (on_court_df['Team_id'] == 1610612746), 'Player4'] = 2549
    on_court_df.loc[(on_court_df['Game_id'] == 20300231)
                    & (on_court_df['Event_Num'] == 495)
                    & (on_court_df['Team_id'] == 1610612746), 'Player5'] = 2549
        
    # Error 2
    pbp_df = pbp_df[~((pbp_df['Game_id'] == 20500121) 
                      & (pbp_df['Event_Num'] == 541))]
    
    # Error 3
    on_court_df.loc[(on_court_df['Game_id'] == 21201224)
                    & (on_court_df['Event_Num'].isin(range(518, 541)))
                    & (on_court_df['Team_id'] == 1610612742), 'Player5'] = 203006
                    
    # Error 4
    on_court_df.loc[(on_court_df['Game_id'] == 21400610)] = on_court_df.loc[
        (on_court_df['Game_id'] == 21400610)].replace({203254:203210})
    
    # Error 5
    on_court_df.loc[(on_court_df['Game_id'] == 21400947)
                    & (on_court_df['Event_Num'] == 103)
                    & (on_court_df['Team_id'] == 1610612751), 
                    ['Player1', 'Player2', 'Player3', 'Player4', 'Player5']] = [
                        101187, 203486, 203900, 101127, 203928]
    on_court_df.loc[(on_court_df['Game_id'] == 21400947)
                    & (on_court_df['Event_Num'] == 103)
                    & (on_court_df['Team_id'] == 1610612740), 
                    ['Player1', 'Player2', 'Player3', 'Player4', 'Player5']] = [
                        203076, 201569, 201582, 201967, 202343]
    
    # Error 6
    on_court_df.loc[(on_court_df['Game_id'] == 21601203)
                & (on_court_df['Event_Num'].isin(range(486, 496)))
                & (on_court_df['Team_id'] == 1610612754), 'Player4'] = 1626176
    
    # Error 7
    on_court_df.loc[(on_court_df['Game_id'] == 21300140)
                    & (on_court_df['Event_Num'].isin(range(527, 534)))
                    & (on_court_df['Team_id'] == 1610612741), 'Player5'] = 203513
    pbp_df = pbp_df[~((pbp_df['Game_id'] == 21300140) 
                      & (pbp_df['Event_Num'] == 535))]

    on_court_df = on_court_df.to_csv(
        "../data/Hackathon_Players_on_Court.csv", compression="gzip", index=False
    )
    pbp_df.to_csv("../data/Hackathon_Play_by_Play.csv", compression="gzip", index=False)


def load_datasets():
    """
    Loads all of the provided datasets into RAM

    Summary of datasets:
        box_score_df: Per game box score statistics for each player
        msg_type_df: English descriptions of encoded event codes
        game_mapping_df: additional information per game (teams, date, time, etc...)
        hustle_df: pbp of hustle stats
        hustle_box_df: box score summaries of pbp hustle stats
        pbp_df: Playbyplay actions in each game
        on_court_df: the list of active players per timestamp
        team_mapping_df: other team info
    """
    box_score_df = pd.read_csv("../data/Hackathon_Box_Scores.csv", compression="gzip")
    msg_type_df = pd.read_csv(
        "../data/Hackathon_Event_Msg_Type.csv", compression="gzip"
    )
    game_mapping_df = pd.read_csv(
        "../data/Hackathon_Game_Mapping.csv", compression="gzip"
    )
    hustle_df = pd.read_csv("../data/Hackathon_Hustle_Stats.csv", compression="gzip")
    hustle_box_df = pd.read_csv("../data/Hackathon_Hustle_Stats_Box_Scores.csv")
    pbp_df = pd.read_csv("../data/Hackathon_Play_by_Play.csv", compression="gzip")
    on_court_df = pd.read_csv(
        "../data/Hackathon_Players_on_Court.csv", compression="gzip"
    )
    team_mapping_df = pd.read_csv(
        "../data/Hackathon_Team_Mapping.csv", compression="gzip"
    )

    return (
        box_score_df,
        msg_type_df,
        game_mapping_df,
        hustle_df,
        hustle_box_df,
        pbp_df,
        on_court_df,
        team_mapping_df,
    )


def clean_pbp_df(pbp_df, on_court_df):
    """Cleans up the pbp_df to be ready for processing

    Removes timeouts, jump balls, validations, memos, final scores, instant replays, 
        stoppages, deadball rebound

    Additionally it validates the team_id for the player involved in each event

    Args:
        pbp_df (pd.Dataframe): a dataframe consisting of the play-by-play 
            info for a single game
        on_court_df (pd.Dataframe): a dataframe consisting of the active members on
            the court for each event in a game
    Returns:
        A cleaned up version of the pbp_df
    """

    def add_team_mapping(x):
        # Adds team -> team mapping per game/team combo
        prev = x.iloc[-1].copy()
        prev["Player_id"] = prev["Team_id"]
        return x.append(prev)

    # Event cleaning
    pbp_df = pbp_df[~pbp_df["Event_Msg_Type"].isin([0, 7, 9, 10, 14, 15, 16, 18, 20])]
    pbp_df = pbp_df[~((pbp_df["Event_Msg_Type"] == 4) & (pbp_df["Action_Type"] == 2))]

    # Generate player to team keys
    on_court_df = on_court_df[
        ["Game_id", "Team_id", "Player1", "Player2", "Player3", "Player4", "Player5"]
    ]
    cols = [c for c in on_court_df if c.startswith("Player")]
    melted = (
        pd.melt(
            on_court_df,
            id_vars=["Game_id", "Team_id"],
            value_vars=cols,
            value_name="Player_id",
        )
        .drop(columns=["variable"])
        .drop_duplicates()
    )
    melted = (
        melted.groupby(["Game_id", "Team_id"])
        .apply(add_team_mapping)
        .reset_index(drop=True)
    )

    # Return values
    pbp_df = pbp_df.merge(
        melted,
        how="left",
        left_on=["Game_id", "Person1"],
        right_on=["Game_id", "Player_id"],
    )
    pbp_df = pbp_df.fillna(0)
    pbp_df = pbp_df.drop(columns=["Team_id_x", "Player_id"]).rename(
        columns={"Team_id_y": "Team_id"}
    )
    pbp_df["Team_id"] = pbp_df["Team_id"].astype("int64")
    mapping = melted.set_index(["Game_id", "Player_id"])
    mapping = {l: mapping.xs(l)["Team_id"].to_dict() for l in mapping.index.levels[0]}

    pbp_df = pbp_df.sort_values(
        by=["Period", "PC_Time", "WC_Time", "Event_Num"],
        ascending=[True, False, True, True],
    )

    return pbp_df, mapping


def process_game(args):
    """Takes a game_df and creates 'realtime' box scores for that game
    Args:
        game_df (pd.Dataframe): game specific pbp_df info
        active_df (pd.DataFrame): game specific on_court_df info
        game_mapping (dict): a dictionary with players an the team they are on
    """
    game_df, active_df, game_mapping = args

    def shot_update(stats_df, shot_idx, points, made, ast_idx, blk_idx=0):
        if points == 1:
            stats_df.loc[shot_idx, "FT"] += 1 if made else 0
            stats_df.loc[shot_idx, "PTS"] += points if made else 0
            stats_df.loc[shot_idx, "FTA"] += 1
        elif points in [2, 3]:
            stats_df.loc[shot_idx, "FG"] += 1 if made else 0
            stats_df.loc[shot_idx, "PTS"] += points if made else 0
            if made and ast_idx != 0:
                stats_df.loc[ast_idx, "AST"] += 1
            stats_df.loc[shot_idx, "FGA"] += 1
            if points == 3:
                stats_df.loc[shot_idx, "3P"] += 1 if made else 0
                stats_df.loc[shot_idx, "3PA"] += 1
            if made is False and blk_idx != 0:
                stats_df.loc[blk_idx, "BLK"] += 1
        else:
            raise ValueError(
                "Invalid point type {}. Points must " "be 1, 2, or 3".format(points)
            )

    def reb_update(stats_df, idx, curr_team, prev_shot_team):
        if prev_shot_team == curr_team:
            stats_df.loc[idx, "OREB"] += 1
        else:
            stats_df.loc[idx, "DREB"] += 1

    def to_update(stats_df, idx, steal_idx):
        stats_df.loc[idx, "TO"] += 1
        if steal_idx != 0:
            stats_df.loc[steal_idx, "STL"] += 1

    def foul_update(stats_df, idx):
        stats_df.loc[idx, "PF"] += 1

    def update_mins(stats_df, active_dict, diff):
        players = list(itertools.chain(*active_dict.values()))
        if len(players) != 10:
            raise Exception("Invalid number of players current list: {}".format(players))
        stats_df.loc[players, "MIN"] += diff

    def update_pm(stats_df, active_dict, score_team, score):
        for team, players in active_dict.items():
            stats_df.loc[players, "+/-"] += score if team == score_team else -score

    def recompute_pcts(stats_df):
        stats_df["FG%"] = stats_df["FG"] / stats_df["FGA"]
        stats_df["3P%"] = stats_df["3P"] / stats_df["3PA"]
        stats_df["FT%"] = stats_df["FT"] / stats_df["FTA"]
        stats_df["FG%"] = np.where(np.isfinite(stats_df["FG%"].values),
                                   stats_df["FG%"].values,
                                   0)
        stats_df["3P%"] = np.where(np.isfinite(stats_df["3P%"].values),
                                   stats_df["3P%"].values,
                                   0)
        stats_df["FT%"] = np.where(np.isfinite(stats_df["FT%"].values),
                                   stats_df["FT%"].values,
                                   0)

    tracked_stats = [
        "MIN",
        "FG",
        "FGA",
        "FG%",
        "3P",
        "3PA",
        "3P%",
        "FT",
        "FTA",
        "FT%",
        "OREB",
        "DREB",
        "AST",
        "PF",
        "STL",
        "TO",
        "BLK",
        "PTS",
        "+/-",
    ]
    team_df_list = []
    player_df_list = []

    TEAMS = pd.unique(active_df["Team_id"])
    PLAYERS = pd.unique(
        active_df[
            ["Player1", "Player2", "Player3", "Player4", "Player5"]
        ].values.ravel()
    )

    # overall stat holding dfs
    teams_df = pd.DataFrame(
        np.zeros((len(TEAMS), len(tracked_stats))),
        index=pd.Index(TEAMS, name="Team_id"),
        columns=tracked_stats,
    )
    midx_arr = [[game_mapping[p] for p in PLAYERS], PLAYERS]
    players_df = pd.DataFrame(
        np.zeros((len(PLAYERS), len(tracked_stats))),
        index=pd.MultiIndex.from_arrays(midx_arr, names=("Team_id", "Player_id")),
        columns=tracked_stats,
    )
    players_df = players_df.reset_index(level=[0])

    prev_time = None
    prev_active = {}

    # rebound var
    prev_shot_team = None

    # +/- related vars
    # snapshot of who was active during last foul
    prev_foul_active = {}

    # Error Tracking
    ERROR = False

    # two dataframes generated from this:
    # * per team live stats
    # * per player live stats
    
    try:
        for i, event in game_df.iterrows():
            mtype = event["Event_Msg_Type"]
            evnum = event["Event_Num"]
            pctime = event["PC_Time"]
            tid = event["Team_id"]
            op1 = event["Option1"]
            p1 = event["Person1"]
            p2 = event["Person2"]
            p3 = event["Person3"]
            # print('Entered ({}): {} {}'.format(multiprocessing.current_process(),
            #                                    event['Event_Num'],
            #                                    mtype))
    
            # Update minutes if not start of quarter
            if mtype != 12 and int((prev_time - pctime)/10) > 0:
                update_mins(players_df, prev_active, (prev_time - pctime) / 10 / 60)
    
            if mtype == 12:  # start quarter
                prev_time = pctime
                cols = ["Player1", "Player2", "Player3", "Player4", "Player5"]
                curr_active = active_df.loc[active_df["Event_Num"] == evnum]
                for i, row in curr_active.iterrows():
                    prev_active[row["Team_id"]] = row[cols].values.tolist()
            elif mtype == 13:  # end quarter
                pass
            elif mtype == 8:  # substitution
                # temp_team is necessary because sometimes a player isn't
                # active but still gets substituted
                if p1 in PLAYERS and p1 in prev_active[game_mapping[p1]]:
                    temp_team = game_mapping[p1]
                    prev_active[temp_team].remove(p1)
                if p2 in PLAYERS:
                    temp_team = game_mapping[p2]
                    prev_active[temp_team].append(p2)
            elif mtype == 1:  # made shot (2 or 3)
                shot_update(teams_df, tid, op1, True, (tid if p2 != 0 else 0))
                shot_update(players_df, p1, op1, True, p2)
                update_pm(players_df, prev_active, tid, op1)
            elif mtype == 2:  # missed shot (w/ BLK info)
                prev_shot_team = tid
                shot_update(teams_df, tid, op1, False, 0, (tid if p3 != 0 else 0))
                shot_update(players_df, p1, op1, False, 0, (p3 if p3 != 0 else 0))
            elif mtype == 3:  # free throw (made or missed)
                made = True if op1 == 1 else False
                shot_update(teams_df, tid, 1, made, 0)
                shot_update(players_df, p1, 1, made, 0)
                if made:  # assumes foul before FT
                    update_pm(players_df, prev_foul_active, tid, 1)
                else:
                    prev_shot_team = tid
            elif mtype == 4:  # rebound
                reb_update(teams_df, tid, tid, prev_shot_team)
                if tid != p1:  # not team rebound
                    reb_update(players_df, p1, tid, prev_shot_team)
            elif mtype == 5:
                to_update(teams_df, tid, (game_mapping[p2] if p2 != 0 else 0))
                if tid != p1:  # not team turnover
                    to_update(players_df, p1, p2 if p2 in PLAYERS else 0)
            elif mtype == 6:
                if not tid == 0:  # coach fouls :/
                    foul_update(teams_df, tid)
                if p1 in PLAYERS:
                    foul_update(players_df, p1)
                prev_foul_active = deepcopy(prev_active)
            
            # recompute % based stats
            recompute_pcts(teams_df)
            recompute_pcts(players_df)
            
            prev_time = pctime  # set prev time
            
            # snapshot stats
            cols = ["Game_id", "Event_Num", "Period", "PC_Time", "WC_Time"]
            active_cols = ["Active{}".format(i) for i in range(10)]
            active_players = list(itertools.chain(*prev_active.values()))
            teams_df = teams_df.assign(**{k: event[k] for k in cols})
            teams_df = teams_df.assign(**{c: p for p, c in zip(active_players, active_cols)})
            players_df = players_df.assign(**{k: event[k] for k in cols})
            players_df = players_df.assign(**{c: p for p, c in zip(active_players, active_cols)})
            team_df_list.append(teams_df.reset_index())
            player_df_list.append(players_df.reset_index())

    except (Exception, Warning) as e:
        print(
            "Game_id ({}): {}".format(
                multiprocessing.current_process(), game_df["Game_id"].iloc[-1]
            )
        )
        print('Entered ({}): {} {}'.format(multiprocessing.current_process(),
                                           event['Event_Num'],
                                           mtype))
        print('Error: \n', e)
        print('\n')
        ERROR = True

    if not ERROR:
        ret_team_df = pd.concat(team_df_list, ignore_index=True)
        ret_player_df = pd.concat(player_df_list, ignore_index=True)
    else:
        ret_team_df = pd.DataFrame()
        ret_player_df = pd.DataFrame()

    return ret_team_df, ret_player_df

def single_game_live_box_scores(game_id):
    print("Processing single game...")
    print("Loading datasets...")
    pbp_df = pd.read_csv("../data/Hackathon_Play_by_Play.csv", compression="gzip")
    on_court_df = pd.read_csv("../data/Hackathon_Players_on_Court.csv", compression="gzip")
    print("Finished loading datasets...")

    game_ids = [game_id]
    pbp_df = pbp_df.loc[pbp_df['Game_id'] == game_id]
    on_court_df = on_court_df.loc[on_court_df['Game_id'] == game_id]
    
    print('Cleaning...')
    pbp_df, mapping = clean_pbp_df(pbp_df, on_court_df)
    print('Processing pbp...')
    ret_team_df, ret_player_df = process_game((pbp_df, on_court_df, mapping[game_id]))

    return ret_team_df, ret_player_df

def generate_raw_live_box_scores():
    print("Processing all games...")
    print("Loading datasets...")
    pbp_df = pd.read_csv("../data/Hackathon_Play_by_Play.csv", compression="gzip")
    on_court_df = pd.read_csv("../data/Hackathon_Players_on_Court.csv", compression="gzip")
    print("Finished loading datasets...")

    game_ids = pd.unique(pbp_df["Game_id"])
    game_ids = [g for g in game_ids if (g in range(41700000, 41800000))
                                        or (g in range(21700000, 21800000))]
    game_len = len(game_ids)

    print("Cleaning play by play dataset...")
    start1 = time.time()
    pbp_df, mapping = clean_pbp_df(pbp_df, on_court_df)
    print("Cleaned {} games in {}s".format(game_len, time.time() - start1))

    print("Divide dataframe into games...")
    start2 = time.time()
    pbp_group = pbp_df.groupby('Game_id')
    on_court_group = on_court_df.groupby('Game_id')
    process_args = [(pbp_group.get_group(g_id), on_court_group.get_group(g_id), mapping[g_id]) for g_id in game_ids]
    print("Divided {} games in {}s".format(game_len, time.time() - start2))
    
    print("Generating live data for play by play games...")
    start3 = time.time()
    with open('../data/output/LiveTeamBoxScores.h5', 'w'):
        pass # wipe file
    with open('../data/output/LivePlayerBoxScores.h5', 'w'):
        pass # wipe file
    with Pool() as pool:
        for gt_df, gp_df in tqdm(pool.imap_unordered(process_game, process_args), total=len(process_args)):
            gt_df.to_hdf('../data/output/LiveTeamBoxScores161718.h5', 'gt_df', format='table', append=True)
            gp_df.to_hdf('../data/output/LivePlayerBoxScores161718.h5', 'gt_df', format='table', append=True)

    print("Processed {} games in {}s".format(game_len, time.time() - start3))


if __name__ == "__main__":
    # patch_raw_data()
    generate_raw_live_box_scores()
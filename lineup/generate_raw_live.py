#!/usr/bin/env python

from copy import deepcopy
import itertools
import multiprocessing
import time

import numpy as np
import pandas as pd
from tqdm import tqdm


def set_pandas_options(max_columns=None, max_rows=None):
    pd.set_option("display.max_columns", max_columns)
    pd.set_option("display.max_rows", max_rows)


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
    pbp_df = pbp_df[~pbp_df["Event_Msg_Type"].isin([9, 10, 14, 15, 16, 18, 20])]
    pbp_df = pbp_df[~((pbp_df["Event_Msg_Type"] == 4) & (pbp_df["Action_Type"] == 2))]
    pbp_df = pbp_df[pbp_df["Event_Num"] > 0]

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


def process_game(game_df, active_df, game_mapping):
    """Takes a game_df and creates 'realtime' box scores for that game
    Args:
        game_df (pd.Dataframe): game specific pbp_df info
        active_df (pd.DataFrame): game specific on_court_df info
        game_mapping (dict): a dictionary with players an the team they are on
    """

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
        stats_df.loc[players, "MIN"] += diff

    def update_pm(stats_df, active_dict, score_team, score):
        for team, players in active_dict.items():
            stats_df.loc[players, "+/-"] += score if team == score_team else -score

    def recompute_pcts(stats_df):
        stats_df["FG%"] = stats_df["FG"] / stats_df["FGA"]
        stats_df["3P%"] = stats_df["3P"] / stats_df["3PA"]
        stats_df["FT%"] = stats_df["FT"] / stats_df["FTA"]
        rep_dict = {np.nan: 0, np.inf: 0, -np.inf: 0}
        stats_df["FG%"] = stats_df["FG%"].map(lambda x: rep_dict.get(x, x))
        stats_df["3P%"] = stats_df["3P%"].map(lambda x: rep_dict.get(x, x))
        stats_df["FT%"] = stats_df["FT%"].map(lambda x: rep_dict.get(x, x))

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

    # two dataframes generated from this:
    # * per team live stats
    # * per player live stats
    print(
        "Game_id ({}): {}".format(
            multiprocessing.current_process(), game_df["Game_id"].iloc[-1]
        )
    )
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
        if mtype == 12:  # start quarter
            prev_time = pctime
            cols = ["Player1", "Player2", "Player3", "Player4", "Player5"]
            curr_active = active_df.loc[active_df["Event_Num"] == evnum]
            for i, row in curr_active.iterrows():
                prev_active[row["Team_id"]] = row[cols].values.tolist()
        elif mtype == 13:  # end quarter
            update_mins(players_df, prev_active, (prev_time - pctime) / 10. / 60.)
        elif mtype == 8:  # substitution
            update_mins(players_df, prev_active, (prev_time - pctime) / 10. / 60.)
            prev_active[tid].remove(p1)
            prev_active[tid].append(p2)
        elif mtype == 1:  # made shot (2 or 3)
            shot_update(teams_df, tid, op1, True, (tid if p2 != 0 else 0))
            shot_update(players_df, p1, op1, True, p2)
            update_pm(players_df, prev_active, tid, op1)
            update_mins(players_df, prev_active, (prev_time - pctime) / 10. / 60.)
        elif mtype == 2:  # missed shot (w/ BLK info)
            prev_shot_team = tid
            shot_update(teams_df, tid, op1, False, 0, (tid if p3 != 0 else 0))
            shot_update(players_df, p1, op1, False, 0, (p3 if p3 != 0 else 0))
            update_mins(players_df, prev_active, (prev_time - pctime) / 10. / 60.)
        elif mtype == 3:  # free throw (made or missed)
            made = True if op1 == 1 else False
            shot_update(teams_df, tid, 1, made, 0)
            shot_update(players_df, p1, 1, made, 0)
            if made:  # assumes foul before FT
                update_pm(players_df, prev_foul_active, tid, 1)
            else:
                prev_shot_team = tid
            update_mins(players_df, prev_active, (prev_time - pctime) / 10. / 60.)
        elif mtype == 4:  # rebound
            reb_update(teams_df, tid, tid, prev_shot_team)
            if tid != p1:  # not team rebound
                reb_update(players_df, p1, tid, prev_shot_team)
            update_mins(players_df, prev_active, (prev_time - pctime) / 10. / 60.)
        elif mtype == 5:
            to_update(teams_df, tid, (game_mapping[p2] if p2 != 0 else 0))
            if tid != p1:  # not team turnover
                to_update(players_df, p1, p2)
            update_mins(players_df, prev_active, (prev_time - pctime) / 10. / 60.)
        elif mtype == 6:
            if not tid == 0:  # coach fouls :/
                foul_update(teams_df, tid)
            if p1 in PLAYERS:
                foul_update(players_df, p1)
            prev_foul_active = deepcopy(prev_active)
            update_mins(players_df, prev_active, (prev_time - pctime) / 10. / 60.)
        elif mtype == 7:
            foul_update(teams_df, tid)  # not perfect will suffice
            update_mins(players_df, prev_active, (prev_time - pctime) / 10. / 60.)

        # recompute % based stats
        recompute_pcts(teams_df)
        recompute_pcts(players_df)

        prev_time = pctime  # set prev time

        # snapshot stats
        cols = ["Game_id", "Event_Num", "Period", "PC_Time"]
        teams_df = teams_df.assign(**{k: event[k] for k in cols})
        players_df = players_df.assign(**{k: event[k] for k in cols})
        team_df_list.append(teams_df.reset_index())
        player_df_list.append(players_df.reset_index())

    ret_team_df = pd.concat(team_df_list)
    ret_player_df = pd.concat(player_df_list)

    return ret_team_df, ret_player_df


def generate_raw_live_box_scores():
    print("Loading datasets...")
    pbp_df = pd.read_csv("../data/Hackathon_Play_by_Play.csv", compression="gzip")
    on_court_df = pd.read_csv(
        "../data/Hackathon_Players_on_Court.csv", compression="gzip"
    )
    print("Finished loading datasets...")

    game_ids = pd.unique(pbp_df["Game_id"])
    game_len = len(game_ids)

    print("Cleaning play by play dataset...")
    start1 = time.time()
    pbp_df, mapping = clean_pbp_df(pbp_df, on_court_df)
    print("Cleaned {} games in {}s".format(game_len, time.time() - start1))

    process_args = [
        (
            pbp_df.loc[pbp_df["Game_id"] == g_id],
            on_court_df.loc[on_court_df["Game_id"] == g_id],
            mapping[g_id],
        )
        for g_id in game_ids
    ]

    print("Generating live data for play by play games...")
    start2 = time.time()
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.starmap(process_game, process_args)
    print("Processed {} games in {}s".format(game_len, time.time() - start2))

    print("Concatenating datasets...")
    ret_team_df_list, ret_player_df_list = zip(*results)
    pd.concat(ret_team_df_list).to_csv(
        "../data/output/LiveTeamBoxScores.csv", index=0, compression="gzip"
    )
    pd.concat(ret_player_df_list).to_csv(
        "../data/output/LivePlayerBoxScores.csv", index=0, compression="gzip"
    )


if __name__ == "__main__":
    generate_raw_live_box_scores()

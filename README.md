# NBAResearch

## Set up repo
* install using requirements.txt
* setup datadir
  * folder named data in root with the following files
    - Hackathon_Play_by_Play.csv
    - Hackathon_Players_on_Court.csv
* directory structure:
```
data
├── Hackathon_Play_by_Play.csv
├── Hackathon_Players_on_Court.csv
└── output
```

* run generate_raw_live.py

## Dataset info
* Hustle stats play by play is only from 2015
* Hustle stats box score data is only from 2016
  * There are 7 types of hustle stats with different action types
  * ['screen_assists', 'charges_drawn',
       'loose_balls_recovered', 'deflections', 'contested_shots',
       'contested_2_shots', 'contested_3_shots', 'box_outs',
       'offensive_loose_ball_recoveries', 'defensive_loose_ball_recoveries',
       'defensive_box_outs', 'offensive_box_outs', 'pts_off_screen_assists']

## Game parser
Event id decision map
1. made shot
    * increment FG
    * increment FGA
    * recompute FG%
    * increment AST
    * if 3P:
        - increment 3P
        - increment 3PA
        - recompute 3P%
    * update points
    * update plus minus
    * update minutes
2. missed shot
    * increment FGA
    * recompute FG%
    * if 3P:
        - increment 3PA
        - recompute 3P%
    * if person3 != 0:
        - update BLK
    * update minutes
3. free throw
    * if made:
        - increment FT
    * increment FTA
    * recompute FG%
    * if 3P:
        - increment 3PA
        - recompute 3P%
    * update minutes
4. rebound (drop all deadball rebounds)
    * if action_type != 1 (team rebound):
        - increment reb for player
    * increment team rebounds
5. turnover
    * if action_type != 11 (team turnover):
        - increment player TO
    * if player2 != 0:
        - increment STL
    * increment team TO
6. foul
    * increment PF
7. violation
    * increment team PF count (know this isn't perfect)
8. substitution
    * complex as shit
    * if foul queue up free throw updates

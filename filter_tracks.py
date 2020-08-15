#!/usr/bin/env python3 

import pandas as pd
import numpy as np

filename = "~/Desktop/projects/dgd_lyric_generation/tracklist.csv"

track_df = pd.read_csv(filename)

#remove duplicate tracks
#track_df = track_df.drop_duplicates(subset="Tracks").reindex([i for i in range(0, len(track_df))])
track_df = track_df.drop_duplicates(subset="Tracks", ignore_index=True).reset_index()

#remove instrumental tracks
titles = track_df["Tracks"]
for e, t in enumerate(titles):
    if "Instrumental" in t:
        track_df = track_df.drop(e)
track_df = track_df.reset_index()
del track_df["level_0"]
del track_df["index"]
del track_df["Unnamed: 0"]
track_df.to_csv("lyrical_tracklist.csv")
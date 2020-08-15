#!/usr/bin/env python3

import pandas as pd
import numpy as np
from spotify_account import SpotifyAccount

acc = SpotifyAccount()
sp = acc.spotify
artist = "Dance Gavin Dance"
artist_uri = "6guC9FqvlVboSKTI77NG2k"

def get_records(artist_uri, album_type, limit=50):
    """Get a list of the records (of a specific album_type) for a given artist

    Args:
        artist_uri (str): The artist URI 
        album_type (str): The type of record to retrieve data for
        limit (int, optional): Limiting the number of items to return. Defaults to 50.

    Returns:
        list: A list of dictionaries with record data
    """
    return sp.artist_albums(artist_uri, album_type=album_type, limit=limit)["items"]

def match_record_to_id(record_data):
    """Match the record title to its URI

    Args:
        record_data (dict): The dictionary with Spotify catalog information about artist's albums

    Returns:
        tuple: The pair (album name, album URI)
    """
    return (record_data["name"], record_data["id"])


def get_tracks(record_name):
    """Get the track names for a specific record

    Args:
        record_name (str): The name of the record as seen on Spotify Desktop App. Not robust to spelling or capitalization errors.

    Returns:
        list: List of tracks that appear on a specific album as strings
    """
    record_id = records_to_ids[record_name]
    results = pd.DataFrame(data=sp.album_tracks(record_id))["items"]
    n_tracks = len(results)
    return [results[i]["name"] for i in range(0, n_tracks)]

def get_track_id(track):
    """Get the URI for a single track

    Args:
        track (str): The name of the track to lookup as seen on Spotify Desktop App. Not robust to spelling or capitalization errors.

    Returns:
        str: The URI identifying a track
    """
    results = sp.search(track, type="track")["tracks"]["items"][0]
    return results["id"]

#full length albums from Dance Gavin Dance
full_albums = get_records(artist_uri, "album")
#singles and EPs from Dance Gavin Dance
singles = get_records(artist_uri, "single")
#one list of full length albums, singles, and EPs
all_records = full_albums+singles

#mapping album names to album id
records_to_ids = {}
for d in all_records:
    result = match_record_to_id(d)
    records_to_ids[result[0]] = result[1]

tracks_to_ids = []
for album_name, uri in records_to_ids.items():
    tracks = get_tracks(album_name)
    for track in tracks:
        tracks_to_ids.append((track, get_track_id(track)))

ntracks = len(tracks_to_ids)
tracks_df = pd.DataFrame(data={"Tracks":[tracks_to_ids[i][0] for i in range(0, ntracks)], "URI":[tracks_to_ids[i][1] for i in range(0, ntracks)]})

#write tracklist to file on disk
tracks_df.to_csv("tracklist.csv")
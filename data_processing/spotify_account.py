#!/usr/bin/env python3

import spotipy
from credentials import Credentials

class SpotifyAccount:

    def __init__(self):
        self.client_id = Credentials["client_id"]
        self.client_secret = Credentials["client_secret"]
        self.username = Credentials["username"]
        self.token = Credentials["token"]
        self.spotify = spotipy.Spotify(auth=self.token)
        return
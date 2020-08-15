#!/usr/bin/env python3 

import pandas as pd
import requests

class SongLyrics:
    
    url_template = "https://genius.com/Dance-gavin-dance-{}-lyrics"
    lyrical_tracklist = pd.read_csv("~/Desktop/projects/dgd_lyric_generation/lyrical_tracklist.csv")

    def __init__(self, title):
        self.title = title
        self.lyrics_url = "https://genius.com/Dance-gavin-dance-{}-lyrics".format(self.fmt_title(self.title))
        try:
            self.loc = self.lyrical_tracklist[self.lyrical_tracklist["Tracks"]==self.title].index[0]
        except:
            print(self.title)
        return 

    def fmt_title(self, title):
        title = title.lower()
        replace_characters = {'&':"and", ' - ':' ', '.':' ', ':':'',',':'', '. ':' ', '/':'-', "'":'','!':'' ,' ':'-'} 
        for k, v in replace_characters.items():
            title = title.replace(k, v)
        return title

        

if __name__=="__main__":

    foo_songs = ["Story Of My Bros - Acoustic", 
                "Tidal Waves: Breakfast, Lunch And Dinner",
                "The Robot vs. Heroin Battle Of Vietnam", 
                "The Robot with Human Hair Pt.2 1/2",
                "The Robot With Human Hair, Pt. 2",
                "Doom & Gloom"
                ]
    for song in foo_songs:
        bar_obj = SongLyrics(song)
        r = requests.get(bar_obj.lyrics_url)
        if r.status_code != 200:
            print(bar_obj.lyrics_url)

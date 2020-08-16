#!/usr/bin/env python3 

import re
import bs4
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

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

    def remove_feature(self, title):
        feature_pattern = re.compile("\(feat-.*\)")
        mo = re.search(feature_pattern, title)
        if mo is not None:
            mo = str(mo.group())
            feat_idx = title.index(mo)
            title = title[:feat_idx].strip('-')
        return title

    def fmt_title(self, title):
        title = title.lower().replace('&', "and")
        
        removable = [':', '!', '-', '.', '/', ',', '. ']
        #delete single, non-letter characters
        wordlist = title.split(' ')
        for i, word in enumerate(wordlist):
            if word in removable:
                wordlist.remove(wordlist[i])

        wordlist = np.reshape(np.array(wordlist), (len(wordlist), 1))
        for sublist in wordlist:
            for char in removable:
                #remove non-letter characters from start/end
                sublist[0] = sublist[0].strip(char)
                #replace apostrophe
                sublist[0] = sublist[0].replace("’s", 's')
                sublist[0] = sublist[0].replace("'", '')
                #replace accented letters
                sublist[0] = sublist[0].replace('é', 'e')
                #replace non-letter characters with hyphens
                sublist[0] = sublist[0].replace(char, '-')

        wordlist = wordlist.flatten()
        title = '-'.join(wordlist)
        title = self.remove_feature(title)
        return title

    def download_lyrics(self):
        r = requests.get(self.lyrics_url)
        soup = BeautifulSoup(r.text, features="html.parser")
        main = soup.find("div", class_="lyrics")
        lyrics = main.get_text(' ', strip=True)

        return 


if __name__=="__main__":

    
    #foo_songs = ["Story Of My Bros - Acoustic", 
    #            "Tidal Waves: Breakfast, Lunch And Dinner",
    #            "The Robot vs. Heroin Battle Of Vietnam", 
    #            "The Robot with Human Hair Pt.2 1/2",
    #            "The Robot With Human Hair, Pt. 2",
    #            "Doom & Gloom"
    #            ]
    
    
    #foo_songs = pd.read_csv("lyrical_tracklist.csv")["Tracks"]
    prisoner = SongLyrics("Prisoner")
    lyrics = prisoner.download_lyrics() 
    print(lyrics)

        
    
#!/usr/bin/env python3


"""
Count how many launches each SpaceX rocket has made.

Prints one line per rocket in the format "<rocket name>: <launch count>",
sorted by launch count (highest first) and then by name (A to Z).
"""


import requests


# SpaceX API v4 endpoints
BASE_LAUNCH_URL = 'https://api.spacexdata.com/v4/launches'
BASE_ROCKET_URL = "https://api.spacexdata.com/v4/rockets/"

if __name__ == '__main__':

    # Maps a rocket name to the number of launches it has made
    rockets = {}

    # Fetch every launch ever recorded
    launches = requests.get(BASE_LAUNCH_URL).json()

    for launch in launches:
        # Each launch stores only the id of its rocket, so look up the name
        rocket_id = launch.get('rocket')
        rocket = requests.get(BASE_ROCKET_URL + rocket_id).json()
        rocket_name = rocket.get('name')

        # Count this launch
        if rocket_name in rockets.keys():
            rockets[rocket_name] += 1
        else:
            rockets[rocket_name] = 1

    # Sort by name first; the next sort is stable, so ties stay alphabetical
    sort = sorted(rockets.items(), key=lambda x: x[0])
    # Then sort by count, highest first
    sort = sorted(sort, key=lambda x: x[1], reverse=True)

    for i in sort:
        print("{}: {}".format(i[0], i[1]))

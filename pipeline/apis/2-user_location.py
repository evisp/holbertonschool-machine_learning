#!/usr/bin/env python3


"""
Print the location of a GitHub user.

Usage: ./2-user_location.py https://api.github.com/users/<username>

Prints the user's location, "Not found" if the user does not exist, or the
number of minutes until the rate limit resets if the limit was exceeded.
"""


import requests
import sys
import time


# Ask GitHub for version 3 of its API
DEFAULT_HEADERS = {"Accept": "application/vnd.github.v3+json"}


if __name__ == "__main__":
    # Exactly one argument is expected: the full API URL of the user
    if len(sys.argv) != 2:
        exit()
    url = sys.argv[1]
    r = requests.get(url, headers=DEFAULT_HEADERS)

    # User found: print the location field
    if r.status_code == 200:
        print(r.json()["location"])

    # No such user
    elif r.status_code == 404:
        print("Not found")

    # Rate limit exceeded: the header holds the reset time as a Unix timestamp
    elif r.status_code == 403:
        rate_limit = int(r.headers["X-Ratelimit-Reset"])
        now = int(time.time())
        # Convert the seconds left into whole minutes
        minutes = int((rate_limit - now) / 60)
        print("Reset in {} min".format(minutes))

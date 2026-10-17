#!/usr/bin/env python3

"""
Find the Star Wars starships that can carry a given number of passengers.

Uses the SWAPI mirror hosted by Holberton, which returns results in pages.
"""

import requests


def availableShips(passengerCount):
    """
    Return the names of the starships that can hold enough passengers.

    Args:
        passengerCount: the minimum number of passengers a ship must carry.

    Returns:
        A list with the name of every starship whose passenger capacity is
        greater than or equal to passengerCount. Ships whose capacity is
        missing, "n/a" or "unknown" are skipped.
    """
    # Start at the first page of the paginated starships endpoint
    url = "https://swapi-api.hbtn.io/api/starships"
    available_ships = []
    # Each page links to the next one through "next", which is None on the
    # last page. The last page is processed before the loop ends.
    while url is not None:
        data = requests.get(url).json()
        starships = data.get("results")
        for starship in starships:
            passengers = starship.get("passengers")
            # Ignore ships with no usable capacity
            if passengers is None or passengers in ["n/a", "unknown"]:
                continue
            # Capacities such as "1,000" use commas, so remove them first
            if int(passengers.replace(",", "")) < passengerCount:
                continue
            available_ships.append(starship.get("name"))
        # Move on to the next page
        url = data.get("next")

    return available_ships

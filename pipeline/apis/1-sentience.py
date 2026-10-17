#!/usr/bin/env python3

"""
Find the home planets of every sentient species in Star Wars.

Uses the SWAPI mirror hosted by Holberton, which returns results in pages.
"""

import requests


def sentientPlanets():
    """
    Return the names of the home planets of all sentient species.

    A species counts as sentient when its designation or its classification
    is "sentient".

    Returns:
        A list of planet names, one per sentient species that has a
        homeworld. Species without a homeworld are skipped.
    """
    # Start at the first page of the species endpoint
    url = "https://swapi-api.hbtn.io/api/species"
    sentient_planets = []
    while url is not None:
        data = requests.get(url).json()
        species = data.get("results")
        # Check both fields, since the data stores "sentient" in either one
        for specie in species:
            if specie.get('designation') == 'sentient' or \
                    specie.get('classification') == 'sentient':
                # A homeworld is a link to a planet, not a name
                homeworld_url = specie.get("homeworld")

                if homeworld_url is None:
                    continue
                # Follow the link to get the planet name
                sentient_planets.append(
                    requests.get(homeworld_url).json().get("name")
                )

        # "next" is None on the last page, which ends the loop
        url = data.get("next")

    return sentient_planets

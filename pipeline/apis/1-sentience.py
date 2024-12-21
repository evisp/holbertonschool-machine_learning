#!/usr/bin/env python3
""" SWAPI API:
    Method that returns the list of names
    of the home planets of all sentient species
"""

import requests


def sentientPlanets():
    """
    Returns the list of names of the home planets of all sentient species
    """
    url = "https://swapi-api.hbtn.io/api/species"
    sentient_planets = []
    while url is not None:
        data = requests.get(url).json()
        species = data.get("results")
        for specie in species:
            if specie.get('designation') == 'sentient' or \
                    specie.get('classification') == 'sentient':
                homeworld_url = specie.get("homeworld")

                if homeworld_url is None:
                    continue
                sentient_planets.append(
                    requests.get(homeworld_url).json().get("name")
                )

        url = data.get("next")

    return sentient_planets


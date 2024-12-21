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
    planets = set()
    while url:  # Loop until there's no more pages
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to retrieve data: {response.status_code}")
            return list(planets)  # Return as a list

        try:
            data = response.json()
        except ValueError:
            print("Error decoding JSON")
            return list(planets)  # Return as a list

        for species in data['results']:
            # Check for sentient species
            if 'sentient' in species.get('classification', '').lower() or \
               'sentient' in species.get('designation', '').lower():

                planet_url = species.get('homeworld')
                if planet_url:
                    # Make a request to get the planet name
                    planet_response = requests.get(planet_url)
                    if planet_response.status_code == 200:
                        planet_data = planet_response.json()
                        planets.add(planet_data['name'])
                    else:
                        print(f"Error: {planet_response.status_code}")

        # Update the URL for the next set of species
        url = data['next']

    return list(planets)

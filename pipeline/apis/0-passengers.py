#!/usr/bin/env python3
""" SWAPI API
    Returns the list of ships that can hold a given number of passengers
"""
import requests


def availableShips(passengerCount):
    """ Return the list of ships that hold 'passengerCount' passangers"""
    ships = []  # List to hold ships that meet the criteria
    url = "https://swapi-api.hbtn.io/api/starships"

    while url:  # Loop until there's no more pages
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code != 200:
            print(f"Failed to retrieve data: {response.status_code}")
            return ships  # Return empty list if there was an error

        try:
            data = response.json()
        except ValueError:
            print("Error decoding JSON")
            return ships  # Return empty list if JSON is invalid

        for ship in data['results']:
            passengers = ship['passengers']
            if passengers.isdigit() and int(passengers) >= passengerCount:
                ships.append(ship['name'])  # Add ship name to the list

        # Check for the next page
        url = data['next']

    return ships

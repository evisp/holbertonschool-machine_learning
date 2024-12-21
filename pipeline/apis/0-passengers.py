#!/usr/bin/env python3
""" SWAPI API
    Returns the list of ships that can hold a given number of passengers
"""
import requests


def availableShips(passengerCount):
    """ Return the list of ships that hold 'passengerCount' passangers"""
    data = requests.get("https://swapi-api.hbtn.io/api/starships").json()
    available_ships = []
    while data.get("next") is not None:
        starships = data.get("results")
        for starship in starships:
            passengers = starship.get("passengers")
            if passengers is None or passengers in ["n/a", "unknown"]:
                continue
            if int(passengers.replace(",", "")) < passengerCount:
                continue
            available_ships.append(starship.get("name"))
        data = requests.get(data.get("next")).json()

    return available_ships

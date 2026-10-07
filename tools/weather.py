import requests


# Coordinates for commonly used cities
CITY_COORDINATES = {
    "hyderabad": (17.3850, 78.4867),
    "bangalore": (12.9716, 77.5946),
    "bengaluru": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707),
    "mumbai": (19.0760, 72.8777),
    "delhi": (28.6139, 77.2090),
    "kolkata": (22.5726, 88.3639),
    "pune": (18.5204, 73.8567),
    "new york": (40.7128, -74.0060),
    "london": (51.5074, -0.1278),
}


def get_weather(city: str):
    """
    Gets current weather information for a city.
    """

    city = city.strip().lower()

    if city not in CITY_COORDINATES:
        return {
            "success": False,
            "error": (
                f"Weather coordinates are not available for '{city}'. "
                "Try Hyderabad, Bangalore, Chennai, Mumbai, Delhi, "
                "Kolkata, Pune, New York, or London."
            )
        }

    latitude, longitude = CITY_COORDINATES[city]

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "apparent_temperature,"
            "wind_speed_10m,"
            "weather_code"
        )
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()
        current = data["current"]

        return {
            "success": True,
            "city": city.title(),
            "temperature": current["temperature_2m"],
            "humidity": current["relative_humidity_2m"],
            "feels_like": current["apparent_temperature"],
            "wind_speed": current["wind_speed_10m"],
            "weather_code": current["weather_code"]
        }

    except requests.RequestException as e:
        return {
            "success": False,
            "error": f"Weather service error: {str(e)}"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }
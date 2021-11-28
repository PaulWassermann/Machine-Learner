from abc import ABC

from .levenshtein import levenshtein_nearest


class BaseFactory(ABC):

    _instance = ""
    _map = {}

    @classmethod
    def create(cls, name: str) -> ...:

        name = name.lower()

        try:
            return cls._map[name]

        except KeyError:
            print(f"\"{name}\" is not a valid {cls._instance} name. Possible values are: "
                  f"""\"{'", "'.join(cls._map.keys())}\"""")
            name = levenshtein_nearest(name, cls._map.keys())
            print(f"Fallback to \"{name}\"\n")
            return cls._map[name]

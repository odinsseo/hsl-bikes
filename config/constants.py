# HSL city bikes: service is typically suspended November–March; exclude from modeling.
HSL_CITY_BIKE_INACTIVE_MONTHS: frozenset[int] = frozenset({1, 2, 3, 11, 12})

# Renamed station names (old -> new) for matching with stations coordinates.
# Based on reference/constants.py for Helsinki City Bikes.
RENAMED_STATIONS = {
    "Kampin metroasema": "Kamppi (M)",
    "Sörnäisten metroasema": "Sörnäinen (M)",
    "Hakaniemen metroasema": "Hakaniemi (M)",
    "Erottaja": "Erottajan aukio",
    "Museokatu": "Apollonkatu",
    "Kauppatori": "Unioninkatu",
    "Eiran Sairaala": "Kapteeninpuistikko",
    "Marian Sairaala": "Marian sairaala",
    "Munkkivuoren ostoskeskus": "Huopalahdentie",
    "Hylkeenpyytäjänkatu": "Hernesaarenranta",
    "Kaironkatu": "Verkatehtaanpuisto",
    "Veturitori": "Diakoniapuisto",
    "Tiurintie": "Lumivaarantie",
    "Porkkalankatu": "Itämerentori",
    "O'Bike Station": "Kaisaniemenpuisto",
    "Ruomelantie***": "Kuunkatu",
    "Kalasataman metroasema": "Kalasatama (M)",
    "Myllypuron metroasema": "Myllypuro (M)",
    "Herttoniemen metroasema": "Herttoniemi (M)",
    "Itäkeskuksen metroasema": "Itäkeskus (M)",
    "Puotilan metroasema": "Puotila (M)",
    "Cygnauksenkatu": "Cygnaeuksenkatu",
    "Siilitien metroasema": "Siilitie (M)",
    "Kulosaaren metroasema": "Kulosaari (M)",
    "Herttoniemen ranta": "Herttoniemenranta",
    "Mäntytie": "Paciuksenkatu",
    "Kalastajantie": "Etuniementie",
    "Adjutantinkatu": "Postipuun koulu",
    "Mestarinkatu": "Läkkitori",
    "Leiritori": "Säteri",
    "Olarinkatu": "Auringonkatu",
    "Ulvilanpuisto": "Ulvilantie",
    "Friisinkalliontie": "Avaruuskatu",
    "Professorintie": "Laajalahden keskus",
    "Lahnalahdentie": "Puistokaari",
    "Niemenmäenkuja": "Huopalahdentie",
    "Messitytönkatu": "Länsisatamankatu",
    "Armas Launiksen katu": "Leppävaarankäytävä",
    "Kauppakartanonkuja": "Petter Wetterin tie",
}

# Maintenance / non-public stations to drop from analysis
STATIONS_TO_DROP_PREFIXES = ("Workshop", " ", "Bike Production", "Pop-Up")

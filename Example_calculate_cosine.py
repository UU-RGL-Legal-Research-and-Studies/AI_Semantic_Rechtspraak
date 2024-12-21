import numpy as np
import pandas as pd

# Woordenboek van de vectorparen waarvoor de cosinusgelijkenis wordt berekend
vectoren = {
    "Voorbeeld 0": (np.array([2, 5]), np.array([5,2])),
    "Voorbeeld 1": (np.array([2, 4]), np.array([5, 2])),
    "Voorbeeld 2": (np.array([2, 5]), np.array([4, 3])),
    "Voorbeeld 3": (np.array([2, 4]), np.array([4, 3])),
    "Voorbeeld 4": (np.array([4, 3]), np.array([5, 2])),
    "Voorbeeld 5": (np.array([2, 5]), np.array([2, 4]))
}

# Woordenboek om de resultaten van de cosinusgelijkenis op te slaan
resultaten = {}

# Itereer over elk vectorpaar en bereken de cosinusgelijkenis
for naam, (A, B) in vectoren.items():
    # Stap 1: Bereken het inwendig product (dot product) van A en B
    # Dit geeft een enkele waarde die de mate van overeenkomst in richting (hoekpercentage) van A en B weergeeft
    inwendig_product = np.dot(A, B)
    
    # Stap 2: Bereken de Euclidische norm (lengte) van elke vector
    # Voor een vector A = (A_1, A_2) is de norm ||A|| = sqrt(A_1^2 + A_2^2)
    # Dit geeft de afstand van de vector tot de oorsprong en maakt het mogelijk de lengte te normaliseren
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    
    # Stap 3: Bereken de cosinusgelijkenis
    # De cosinusgelijkenis is het inwendig product gedeeld door het product van de normen
    # Hierdoor krijgen we een waarde tussen -1 en 1 die aangeeft hoe gelijk de richtingen zijn
    cosinus_gelijkenis = inwendig_product / (norm_A * norm_B)
    
    # Voeg het resultaat toe aan het woordenboek met de resultaten
    resultaten[naam] = cosinus_gelijkenis

# Zet de resultaten om naar een DataFrame voor overzichtelijke weergave
resultaten_df = pd.DataFrame.from_dict(resultaten, orient='index', columns=['Cosinusgelijkenis'])
print(resultaten_df)

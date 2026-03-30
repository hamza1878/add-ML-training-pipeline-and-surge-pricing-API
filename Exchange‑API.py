

from __future__ import annotations

import requests


API_KEY         = "ca0c0c8d9cca7488c9a4f80f"
BASE_URL        = "https://v6.exchangerate-api.com/v6"

CURRENCY_ALIASES = {
    "DT":  "TND",  
    "TND": "TND",
    "EUR": "EUR",
    "USD": "USD",
}


# ─────────────────────────────────────────────────────────────────
# FONCTIONS
# ─────────────────────────────────────────────────────────────────

def get_exchange_rate(base: str, target: str) -> float | None:
    """
    Retourne le taux de change base → target.
    Retourne None si l'API échoue ou si le code est introuvable.
    """
    base   = CURRENCY_ALIASES.get(base.upper(),   base.upper())
    target = CURRENCY_ALIASES.get(target.upper(), target.upper())

    url = f"{BASE_URL}/{API_KEY}/latest/{base}"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("result") != "success":
            print(f"  ⚠️  Erreur API : {data.get('error-type', 'inconnu')}")
            return None

        rate = data["conversion_rates"].get(target)
        if rate is None:
            print(f"  ⚠️  Code devise '{target}' introuvable.")
        return rate

    except requests.RequestException as e:
        print(f"  ⚠️  Requête échouée : {e}")
        return None


def convert(amount: float, base: str, target: str) -> float | None:
    """
    Convertit ``amount`` de ``base`` vers ``target``.
    Retourne le montant converti, ou None en cas d'erreur.
    """
    rate = get_exchange_rate(base, target)
    if rate is None:
        return None
    return round(amount * rate, 3)


# ─────────────────────────────────────────────────────────────────
# DEMO CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Exemple : 100 EUR → TND
    pairs = [
        (100.0, "EUR", "TND"),
        (100.0, "EUR", "DT"),   # alias DT accepté
        (50.0,  "USD", "TND"),
    ]

    print("=" * 45)
    print("  MOVIROO — Conversion de devises")
    print("=" * 45)

    for amount, base, target in pairs:
        result = convert(amount, base, target)
        if result is not None:
            iso_target = CURRENCY_ALIASES.get(target.upper(), target.upper())
            print(f"  {amount:>8.2f} {base} = {result:>10.3f} {iso_target}")
        else:
            print(f"  ❌ Conversion {base}→{target} impossible")
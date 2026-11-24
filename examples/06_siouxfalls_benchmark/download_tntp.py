"""Downloads the Sioux Falls network and demand files from the TNTP repository at a pinned commit."""
import hashlib
import os
import sys
import urllib.request

COMMIT = "d1639b4ef218c17928ba573e806ddf8ba5e7ae6d"
URL = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/{}/SiouxFalls/{}"
FILES = {"SiouxFalls_net.tntp": "ace99b24cec69c273ff0cf3d6d074110177f0cc0ae24b0c7a9f4f4cb5e27635c",
         "SiouxFalls_trips.tntp": "56f9566857f3f66730fd5c4232258d7ee3ac2931a476526331afd062f4958de7"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads", "SiouxFalls")


def sha256(fn):
    return hashlib.sha256(open(fn, "rb").read()).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, expected in FILES.items():
        fn = os.path.join(OUT, name)
        if not (os.path.exists(fn) and sha256(fn) == expected):
            urllib.request.urlretrieve(URL.format(COMMIT, name), fn)
        if sha256(fn) != expected:
            sys.exit(f"{name}: sha256 mismatch, expected {expected}")
        print(f"{name}: ok")
    print("TNTP data (github.com/bstabler/TransportationNetworks) is for academic research purposes only; "
          "it is not redistributed with this repository.")


if __name__ == "__main__":
    main()
